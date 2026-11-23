# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import gc
import operator
import warnings
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import Any, Callable

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map, TreeSpec
from torch.utils.weak import WeakIdKeyDictionary
from torch_cudagraph_fx._cudagraph_annotations import (
    _USE_TORCH_GRAPH_ANNOTATIONS,
    annotations_available,
    get_kernel_annotations,
    mark_kernels,
    maybe_get_kernel_py_stacks,
)
from torch_cudagraph_fx._events import trace_cuda_event_constructors
from torch_cudagraph_fx._export_json import export_graph
from torch_cudagraph_fx._indexed_graph import IndexedGraph
from torch_cudagraph_fx._interpreter import CudaGraphFxInterpreter, is_graph_variable
from torch_cudagraph_fx._metadata import (
    bind_node_metadata,
    FREE_AFTER_KEY,
    FX_NODE_ANNOTATION_KEY,
    FX_NODE_ID_KEY,
    MetadataManager,
    node_fx_id,
    node_fx_id_from_metadata,
    node_metadata,
    node_stream,
    node_tensor_meta,
    NodeMetadata,
    StackTraceMetadataProvider,
    StreamMetadataProvider,
)

_FX_LITERAL_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    Enum,
    torch.device,
    torch.dtype,
    torch.layout,
    torch.memory_format,
)


@dataclass(frozen=True)
class _UncapturedKernel:
    name: str | None
    node_type: str
    py_stack: str | None = None


@dataclass
class CudaGraphState:
    cuda_graph: torch.cuda.CUDAGraph
    flat_inputs: list[Any]
    flat_outputs: list[Any]


def _print_readable_with_streams(
    module: fx.GraphModule,
    module_name: str,
    include_stride: bool = False,
    include_device: bool = False,
    colored: bool = False,
    include_stack_trace: bool = False,
    expanded_def: bool = False,
) -> str:
    graph = module.graph
    if graph is None or not isinstance(graph, torch.fx.Graph):
        raise AssertionError("print_readable must be used on a module with a graph")

    nodes = list(graph.nodes)

    def add_stream_comments(
        current_transform: Callable[[list[str]], list[str]] | None,
    ) -> Callable[[list[str]], list[str]]:
        def transform(body: list[str]) -> list[str]:
            lines: list[str] = []
            for line in body:
                lines.append(line)
                if line.startswith("# COUNTER: "):
                    node = nodes[int(line.removeprefix("# COUNTER: ").strip())]
                    if (
                        node.op == "call_function"
                        and (stream := node_stream(node)) is not None
                    ):
                        lines.append(f"# stream {stream}\n")
            if current_transform is not None:
                return current_transform(lines)
            return lines

        return transform

    with graph.on_generate_code(add_stream_comments):
        python_code = graph.python_code(
            root_module="self",
            verbose=include_stack_trace,
            include_stride=include_stride,
            include_device=include_device,
            colored=colored,
            expanded_def=expanded_def,
        )

    module_code = python_code.src.lstrip("\n")
    module_code = f"class {module_name}(torch.nn.Module):\n" + module_code
    lines = module_code.split("\n")
    lines = [lines[0], *[f"    {line}" if line.strip() else "" for line in lines[1:]]]
    return "\n".join(lines).rstrip()


def _has_cudagraph_fx_node_annotation(
    annotations: list[dict[str, Any]],
) -> bool:
    return any(annotation.get(FX_NODE_ANNOTATION_KEY) for annotation in annotations)


@contextmanager
def annotated_capture(
    cuda_graph: torch.cuda.CUDAGraph, **kwargs: Any
) -> Iterator[None]:
    kwargs.setdefault("capture_error_mode", "relaxed")
    if not annotations_available():
        with torch.cuda.graph(cuda_graph, **kwargs):
            yield
        return

    if _USE_TORCH_GRAPH_ANNOTATIONS:
        if maybe_get_kernel_py_stacks() is not None:
            kwargs["annotation_config"] = {
                **(kwargs.get("annotation_config") or {}),
                "record_py_stacks": True,
            }
        with torch.cuda.graph(cuda_graph, enable_annotations=True, **kwargs):
            yield
        return

    from torch_cudagraph_fx._cudagraph_annotations_polyfill import (
        capture_with_annotations_enabled,
    )

    with capture_with_annotations_enabled(cuda_graph, **kwargs):
        yield


def _walk_graph_nodes(
    cuda_graph: torch.cuda.CUDAGraph,
) -> Iterator[tuple[str, int, str | None]]:
    if _USE_TORCH_GRAPH_ANNOTATIONS:
        if not cuda_graph._has_graph_exec:
            cuda_graph.instantiate()
        for node in cuda_graph.get_graph_data()["nodes"]:
            yield node["node_type"], node["tools_id"], node["kernel_name"]
        return

    from torch_cudagraph_fx._cudagraph_annotations_polyfill import walk_graph_nodes

    yield from walk_graph_nodes(cuda_graph)


def _check_uncaptured_kernels(
    cuda_graph: torch.cuda.CUDAGraph,
    *,
    allow_missing_kernels: bool = False,
) -> None:
    if not annotations_available():
        warnings.warn(
            "torch_cudagraph_fx: CUDA graph kernel annotations are not available; "
            "the traced graph might not capture all kernel launches. Run with "
            "LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat to enable kernel annotations "
            "and verify capture completeness.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    py_stacks = maybe_get_kernel_py_stacks()
    annotations = get_kernel_annotations()
    uncaptured: list[_UncapturedKernel] = []
    for node_type, tools_id, kernel_name in _walk_graph_nodes(cuda_graph):
        # Only torch's annotatable node types can carry a mark_kernels toolsId --
        # _get_annotatable_types() == {KERNEL, MEMCPY, MEMSET, BATCH_MEM_OP} --
        # so the filter mirrors that set.
        if node_type not in ("kernel", "memcpy", "memset", "batch_mem_op"):
            continue
        if _has_cudagraph_fx_node_annotation(annotations.get(tools_id, [])):
            continue
        uncaptured.append(
            _UncapturedKernel(
                name=kernel_name,
                node_type=node_type,
                py_stack=py_stacks.get(tools_id) if py_stacks is not None else None,
            )
        )
    if uncaptured:
        message = _format_uncaptured_kernels(
            uncaptured, allow_missing_kernels=allow_missing_kernels
        )
        if allow_missing_kernels:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        else:
            raise RuntimeError(message)


def _format_uncaptured_kernels(
    uncaptured: list[_UncapturedKernel],
    *,
    allow_missing_kernels: bool,
) -> str:
    no_py_stack = "Python launch stack not captured."
    names = [k.name or k.node_type for k in uncaptured]
    lines = [
        f"torch_cudagraph_fx: {len(uncaptured)} kernel launch(es) had no fx node "
        f"and are missing from the trace: {names}",
        "These launches bypassed __torch_dispatch__. To capture them, wrap the "
        'function that launches them in @torch_cudagraph_fx.trace_leaf("ns::op") '
        "(set CGFX_TRACING=1 before it is imported) or in a custom op; see "
        "'Tracing leaves' in the torch_cudagraph_fx README.",
    ]
    for kernel in uncaptured:
        lines.append(
            f"  {kernel.name or kernel.node_type} ({kernel.node_type}) launched from:"
        )
        stack = kernel.py_stack or no_py_stack
        lines.extend(f"    {frame}" for frame in stack.splitlines())
    if allow_missing_kernels:
        lines.append(
            "Continuing with an incomplete FX graph; replay may produce incorrect "
            "results."
        )
    else:
        lines.append(
            "To return the incomplete trace with a warning, pass "
            "allow_missing_kernels=True to make_cudagraph_fx."
        )
    return "\n".join(lines)


def _address(tensor: torch.Tensor) -> int:
    """The address of the storage of ``tensor``, or 0 if it holds no memory or
    forbids reading its address, as a placeholder for freed memory can."""
    storage = tensor.untyped_storage()
    if not storage.nbytes():
        return 0
    try:
        return storage.data_ptr()
    except RuntimeError:
        return 0


class _CudagraphFxDispatchMode(TorchDispatchMode):
    """Record each dispatched op as an fx node, tagged with its current CUDA stream.

    Not built on ``make_fx`` because we want to start with a minimal complexity. Three
    key implementation differences with ``make_fx``:

    1. Stream annotation: We'll record the CUDA stream current at dispatch, recorded per node.
    2. No ``proxy`` layer. ``__torch_dispatch__`` already hands us the op and its
       args, so we build fx nodes directly.
    3. No decompositions. One fx node per dispatched op (the kernel launch it
       represents), no decompositions.
    4. No dynamic shape support.

    Kernel launches are recorded into the capture (see :func:`make_cudagraph_fx`),
    not executed. Tensors are still real objects (valid identity/shape/stride/
    dtype); the graph is wired by object identity and op, never by tensor data.
    Streams get small stable ids in first-seen order -- the stream current at
    trace start is id ``0`` (the default lane).
    """

    def __init__(self, graph: fx.Graph) -> None:
        super().__init__()
        self.graph = graph
        self.root: torch.nn.Module = torch.nn.Module()
        self._value_to_node: WeakIdKeyDictionary = WeakIdKeyDictionary()
        # The latest input or result of each storage, by address and dtype.
        self._storages: dict[
            tuple[int, torch.dtype], tuple[StorageWeakRef, fx.Node]
        ] = {}
        # Root attributes keep constants alive, so their ids cannot be reused.
        self._constant_nodes: dict[int, fx.Node] = {}
        self._next_constant_index = 0
        self._stream_ids: dict[int, int] = {}
        self._newest: fx.Node | None = None
        # Operator results, watched so replay can free each where tracing did.
        self._lifetimes: list[tuple[fx.Node, weakref.finalize]] = []
        self.metadata = MetadataManager(
            graph,
            [
                StreamMetadataProvider(self.current_stream_id),
                StackTraceMetadataProvider(),
            ],
        )

    def current_stream_id(self) -> int:
        ptr = torch.cuda.current_stream().cuda_stream
        if ptr not in self._stream_ids:
            self._stream_ids[ptr] = len(self._stream_ids)
        return self._stream_ids[ptr]

    def bind_inputs(self, flat_args: list[Any]) -> None:
        # A placeholder per graph variable; literals are embedded where used.
        for i, value in enumerate(flat_args):
            if is_graph_variable(value):
                placeholder = self._create_node("placeholder", f"arg_{i}")
                self._bind_value(value, placeholder)

    def _create_node(
        self,
        op: str,
        target: fx.node.Target,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        *,
        node_meta: NodeMetadata | None = None,
    ) -> fx.Node:
        if node_meta is None:
            node_meta = self.metadata.new_node_metadata()
        node = self.graph.create_node(op, target, args, kwargs)
        bind_node_metadata(node, node_meta)
        self._newest = node
        return node

    def _bind_value(
        self,
        value: Any,
        node: fx.Node,
        *,
        created: bool = False,
    ) -> None:
        """Register a runtime value and enrich its already attached node metadata.
        ``created`` marks an operator result that tracing allocated."""
        if is_graph_variable(value):
            # An in-place result is a tensor that already has a binding.
            created = created and value not in self._value_to_node
            self._value_to_node[value] = node
            if created and isinstance(value, torch.Tensor):
                watch = weakref.finalize(value, self._record_free, node)
                self._lifetimes.append((node, watch))
        else:
            self._constant_nodes[id(value)] = node
        if isinstance(value, torch.Tensor):
            self.metadata.add_tensor_meta(node, value)
            # Constants and weights persist, so a tensor sharing their storage
            # is one too.
            persistent = node_tensor_meta(node)["type"] in ("constant", "weight")
            if not persistent and (address := _address(value)):
                storage = StorageWeakRef(value.untyped_storage())
                self._storages[address, value.dtype] = (storage, node)

    def _record_free(self, node: fx.Node) -> None:
        assert self._newest is not None
        node_metadata(node)[FREE_AFTER_KEY] = node_fx_id(self._newest)

    def finish(self, *, release_escaped: bool) -> list[tuple[fx.Node, torch.Tensor]]:
        """Stop watching results. With ``release_escaped``, return the results
        still referenced after tracing, with their storage freed: tracing never
        computed them."""
        if release_escaped:
            gc.collect()
        escaped = []
        for node, watch in self._lifetimes:
            alive = watch.detach()
            # A tensor whose storage the function freed holds nothing to rebind.
            if release_escaped and alive and alive[0].untyped_storage().nbytes():
                escaped.append((node, alive[0]))
        self._lifetimes.clear()
        with torch.no_grad():
            for _, value in escaped:
                value.set_()
        return escaped

    def _lookup_value(
        self,
        value: Any,
    ) -> fx.Node | None:
        if is_graph_variable(value):
            return self._value_to_node.get(value)
        return self._constant_nodes.get(id(value))

    def project(self, node: fx.Node, path: tuple[Any, ...]) -> fx.Node:
        # Native operators can return nested sequences, e.g. (Tensor[], Work).
        source = node
        for key in path:
            assert isinstance(key, pytree.SequenceKey), (
                f"expected a sequence output path, got {path!r}"
            )
            node = self._create_node(
                "call_function",
                operator.getitem,
                (node, key.idx),
                node_meta=self.metadata.new_projection_metadata(source),
            )
        return node

    def lift(self, value: Any) -> Any:
        if isinstance(value, _FX_LITERAL_TYPES):
            return value
        if (node := self._lookup_value(value)) is not None:
            return node
        if isinstance(value, torch.Tensor) and (node := self._view_of(value)):
            return node
        node = self._lift_constant(value)
        self._bind_value(value, node)
        return node

    def _view_of(self, value: torch.Tensor) -> fx.Node | None:
        """A tensor made without dispatch, such as one autograd or DLPack wraps
        around existing memory, that shares the live storage of an input or
        result is a view of it; as a constant, it would keep the traced storage
        and its values."""
        entry = self._storages.get((_address(value), value.dtype))
        if entry is None or entry[0].expired():
            return None
        args = (value.size(), value.stride(), value.storage_offset())
        self._record_call_function(
            torch.ops.aten.as_strided.default,
            (entry[1], *args),
            {},
            value,
            self.metadata.before_call((value, *args), {}),
        )
        return self._value_to_node[value]

    def _lift_constant(self, value: Any) -> fx.Node:
        name = f"_constant_{self._next_constant_index}"
        self._next_constant_index += 1
        setattr(self.root, name, value)
        return self._create_node("get_attr", name)

    def _register_outputs(
        self,
        node: fx.Node,
        out: Any,
    ) -> None:
        # ATen results are new tensors or inputs; a custom operator can return a
        # tensor it keeps, such as a cached workspace.
        created = isinstance(
            node.target, torch._ops.OpOverload
        ) and node.target.namespace in ("aten", "prims")
        for path, value in pytree.tree_flatten_with_path(out)[0]:
            if is_graph_variable(value):
                self._bind_value(value, self.project(node, path), created=created)

    def _record_call_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        out: Any,
        node_meta: NodeMetadata,
    ) -> None:
        node = self._create_node(
            "call_function",
            func,
            args,
            kwargs,
            node_meta=node_meta,
        )
        self.metadata.after_call(node, out)
        self._register_outputs(node, out)

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        lifted_args = tree_map(self.lift, args)
        lifted_kwargs = tree_map(self.lift, kwargs)
        node_meta = self.metadata.before_call(args, kwargs)
        with mark_kernels(
            {
                FX_NODE_ANNOTATION_KEY: True,
                FX_NODE_ID_KEY: node_fx_id_from_metadata(node_meta),
            }
        ):
            out = func(*args, **kwargs)
        self._record_call_function(
            func,
            lifted_args,
            lifted_kwargs,
            out,
            node_meta,
        )
        return out


class CudaGraphFxTrace:
    def __init__(
        self,
        graph_module: fx.GraphModule,
        in_spec: TreeSpec,
        out_spec: TreeSpec,
        *,
        cudagraph_state: CudaGraphState | None = None,
        escaped: Sequence[torch.Tensor] = (),
    ) -> None:
        self._gm = graph_module
        self._in_spec = in_spec
        self._out_spec = out_spec
        self._cudagraph_state = cudagraph_state
        self._escaped = tuple(escaped)

    @property
    def escaped(self) -> tuple[torch.Tensor, ...]:
        """Tensors that tracing created and its caller still references. The
        graph returns them after the function's outputs, and replay makes each
        one alias its new value."""
        return self._escaped

    @property
    def graph_module(self) -> fx.GraphModule:
        return self._gm

    @property
    def graph(self) -> fx.Graph:
        return self._gm.graph

    @functools.cached_property
    def indexed_graph(self) -> IndexedGraph:
        return IndexedGraph(self._gm)

    @property
    def in_spec(self) -> TreeSpec:
        return self._in_spec

    @property
    def out_spec(self) -> TreeSpec:
        return self._out_spec

    def cudagraph_states(self) -> CudaGraphState | None:
        return self._cudagraph_state

    def print_readable(
        self,
        print_output: bool = True,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        include_stack_trace: bool = False,
        *,
        expanded_def: bool = False,
    ) -> str:
        graph_text = _print_readable_with_streams(
            self._gm,
            self._gm._get_name(),
            include_stride=include_stride,
            include_device=include_device,
            colored=colored,
            include_stack_trace=include_stack_trace,
            expanded_def=expanded_def,
        )
        if print_output:
            print(graph_text)
        return graph_text

    def _interpreter(self) -> CudaGraphFxInterpreter:
        return CudaGraphFxInterpreter(
            self._gm,
            in_spec=self._in_spec,
            out_spec=self._out_spec,
            escaped=self._escaped,
        )

    def export_json(self, path: str | PathLike[str]) -> None:
        """Write operator and tensor metadata as JSON without executing the graph.

        Operator returns describe tensors; graph outputs also include literals.
        CUDA graph retention is not required.
        """
        export_graph(self.graph, path)

    def replay(self, *args: Any, **kwargs: Any) -> Any:
        return self._interpreter().run_fn(*args, **kwargs)

    def release_cudagraph_states(self) -> None:
        if self._cudagraph_state is not None:
            self._cudagraph_state.cuda_graph.reset()
            self._cudagraph_state = None


def make_cudagraph_fx(
    fn: Callable[..., Any],
    *,
    retain_cudagraph: bool = False,
    allow_missing_kernels: bool = False,
    pool: tuple[int, int] | None = None,
    stream: torch.cuda.Stream | None = None,
) -> Callable[..., CudaGraphFxTrace]:
    """``pool`` and ``stream`` set the memory pool and stream of the tracing
    capture, as in ``torch.cuda.graph``."""

    def traced(*args: Any, **kwargs: Any) -> CudaGraphFxTrace:
        graph = fx.Graph()
        mode = _CudagraphFxDispatchMode(graph)
        flat_args, in_spec = tree_flatten((args, kwargs))
        cuda_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with (
            torch.autograd.grad_mode.set_multithreading_enabled(False),
            annotated_capture(cuda_graph, pool=pool, stream=stream),
            trace_cuda_event_constructors(),
            mode.metadata.track_modules(),
            mode,
            mode.metadata.enable_autograd_hooks(),
        ):
            mode.current_stream_id()  # reserve id 0 for the capturing stream
            mode.bind_inputs(flat_args)
            out = fn(*args, **kwargs)
            out_flat, out_spec = tree_flatten(out)
            # Older FX codegen uses repr() for outputs, which is invalid for enums.
            outputs = tuple(
                mode._lift_constant(o) if isinstance(o, Enum) else mode.lift(o)
                for o in out_flat
            )
        _check_uncaptured_kernels(
            cuda_graph, allow_missing_kernels=allow_missing_kernels
        )
        state = None
        if retain_cudagraph:
            # The retained graph computes the traced tensors themselves.
            state = CudaGraphState(cuda_graph, flat_args, out_flat)
        del out, out_flat
        escaped = mode.finish(release_escaped=not retain_cudagraph)
        graph.output((*outputs, *(node for node, _ in escaped)))
        mode.metadata.post_process()
        gm = fx.GraphModule(mode.root, graph)
        return CudaGraphFxTrace(
            gm,
            in_spec,
            out_spec,
            cudagraph_state=state,
            escaped=[value for _, value in escaped],
        )

    return traced
