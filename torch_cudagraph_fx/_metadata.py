# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from threading import get_ident
from typing import Any, cast, TypeAlias

import torch
import torch.fx as fx
from torch._C import _SchemaArgType, _SchemaArgument, _SchemaInfo
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._pytree import tree_flatten
from torch.utils._traceback import CapturedTraceback
from torch.utils.hooks import RemovableHandle
from torch.utils.weak import WeakIdRef

CUDAGRAPH_FX_METADATA_KEY = "cudagraph_fx"
FX_NODE_ANNOTATION_KEY = "cudagraph_fx.fx_node"
FX_NODE_ID_KEY = "fx_node_id"

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
NodeMetadata: TypeAlias = dict[str, JsonValue]

_STACKTRACE_INTERNAL_FILE_MARKERS = (
    "/site-packages/torch/",
    "/torch_cudagraph_fx/custom_ops/",
    "/torch_cudagraph_fx/_metadata.py",
    "/torch_cudagraph_fx/_tracer.py",
    "/torch/_compile.py",
    "/torch/_dynamo/eval_frame.py",
    "/torch/_ops.py",
    "/torch/_tensor.py",
    "/torch/utils/_python_dispatch.py",
)


def node_metadata(node: fx.Node) -> NodeMetadata:
    metadata = node.meta.get(CUDAGRAPH_FX_METADATA_KEY)
    if not isinstance(metadata, dict):
        raise AssertionError(f"FX node {node.name!r} has no cudagraph_fx metadata")
    return cast(NodeMetadata, metadata)


def node_autograd_meta(node: fx.Node) -> NodeMetadata:
    return cast(NodeMetadata, node_metadata(node).get("autograd", {}))


def node_tensor_meta(node: fx.Node) -> NodeMetadata:
    return cast(NodeMetadata, node_metadata(node).get("tensor", {}))


def node_fx_id(node: fx.Node) -> int:
    value = node_metadata(node)[FX_NODE_ID_KEY]
    if not isinstance(value, int):
        raise AssertionError(f"FX node {node.name!r} has an invalid fx_node_id")
    return value


def node_stream(node: fx.Node) -> int | None:
    execution = node_metadata(node).get("execution")
    if not isinstance(execution, dict):
        return None
    stream = execution.get("stream")
    return stream if isinstance(stream, int) else None


def node_stack_trace(node: fx.Node) -> str | None:
    execution = node_metadata(node).get("execution")
    if not isinstance(execution, dict):
        return None
    stack_trace = execution.get("stack_trace")
    return stack_trace if isinstance(stack_trace, str) else None


def bind_node_metadata(node: fx.Node, node_meta: NodeMetadata) -> None:
    if CUDAGRAPH_FX_METADATA_KEY in node.meta:
        raise AssertionError(f"FX node {node.name!r} already has cudagraph_fx metadata")
    nodes = node.graph.nodes
    if next(iter(reversed(nodes))) is not node:
        raise AssertionError("metadata can only be bound to the newest node")
    if node_fx_id_from_metadata(node_meta) != len(nodes) - 1:
        raise AssertionError("fx_node_id does not match graph insertion order")
    node.meta[CUDAGRAPH_FX_METADATA_KEY] = node_meta
    if (stack_trace := node_stack_trace(node)) is not None:
        node.stack_trace = stack_trace


def _is_internal_stack_frame(frame: traceback.FrameSummary) -> bool:
    return any(marker in frame.filename for marker in _STACKTRACE_INTERNAL_FILE_MARKERS)


def _format_user_stack_summary(summary: traceback.StackSummary) -> str:
    frames = [frame for frame in summary if not _is_internal_stack_frame(frame)]
    if not frames:
        frames = list(summary)
    return "".join(traceback.StackSummary.from_list(frames).format())


def node_fx_id_from_metadata(node_meta: NodeMetadata) -> int:
    value = node_meta[FX_NODE_ID_KEY]
    if not isinstance(value, int):
        raise AssertionError("cudagraph_fx fx_node_id is not an integer")
    return value


class MetadataProvider:
    """Mutate metadata in place; node hooks receive already initialized nodes."""

    def before_call(
        self,
        node_meta: NodeMetadata,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        pass

    def after_call(
        self,
        node: fx.Node,
        returns: Any,
    ) -> None:
        pass

    def post_process(self, graph: fx.Graph) -> None:
        pass


class StreamMetadataProvider(MetadataProvider):
    def __init__(self, current_stream_id: Callable[[], int]) -> None:
        self._current_stream_id = current_stream_id

    def before_call(
        self,
        node_meta: NodeMetadata,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        execution = node_meta.setdefault("execution", {})
        assert isinstance(execution, dict)
        execution["stream"] = self._current_stream_id()


class StackTraceMetadataProvider(MetadataProvider):
    def before_call(
        self,
        node_meta: NodeMetadata,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        execution = node_meta.setdefault("execution", {})
        assert isinstance(execution, dict)
        execution["stack_trace"] = _format_user_stack_summary(
            CapturedTraceback.extract(skip=1).summary()
        )


class AutogradMetadataProvider(MetadataProvider):
    """Record backward sequence IDs and infer forward associations for every op.

    ``executing_seq_nr`` identifies the active autograd node, or is ``None``
    outside backward. ``creating_seq_nrs`` records inferred graph construction
    associations, including recomputation and higher-order differentiation.

    Each ``after_call`` samples the thread-local next-sequence counter as
    ``after_nr``, assigns ``creating_seq_nrs``, then updates ``_previous_after_nr``:

    * Counter advanced: assign ``range(_previous_after_nr, after_nr)``, allowing
      one op to be associated with one or more IDs.
    * Counter unchanged in the eligible context: reuse ``[after_nr - 1]``,
      allowing multiple ops to share one ID.
    * Otherwise: assign ``[]``.

    ``_creating_seq_context`` records the graph task and active backward node
    eligible to reuse the last ID, or ``None`` when the last call recorded no
    IDs. Reuse cannot cross into another backward node or invocation.

    Attribution remains heuristic: intervals include allocations between calls,
    and the fallback can associate no-grad ops with unrelated IDs in a context.
    """

    def __init__(self) -> None:
        self._previous_after_nr = torch.autograd._get_sequence_nr()
        self._creating_seq_context: (
            tuple[int, torch.autograd.graph.Node | None] | None
        ) = None

    def before_call(
        self,
        node_meta: NodeMetadata,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        active_backward_node = torch._C._current_autograd_node()
        node_meta["autograd"] = {
            "phase": "backward" if active_backward_node is not None else "forward",
            "forward_node_ids": [],
        }

    def after_call(
        self,
        node: fx.Node,
        returns: Any,
    ) -> None:
        autograd = cast(NodeMetadata, node_metadata(node)["autograd"])
        active_backward_node = torch._C._current_autograd_node()
        if active_backward_node is not None:
            autograd["executing_seq_nr"] = int(active_backward_node._sequence_nr())
        else:
            autograd["executing_seq_nr"] = None

        # The node prevents ID reuse across backward nodes; the task ID prevents
        # reuse across repeated or nested backward runs through the same node.
        context = (torch._C._current_graph_task_id(), active_backward_node)
        after_nr = torch.autograd._get_sequence_nr()
        if after_nr < self._previous_after_nr:
            raise AssertionError("autograd sequence number moved backwards")
        creating_seq_nrs = list(range(self._previous_after_nr, after_nr))
        if not creating_seq_nrs and context == self._creating_seq_context:
            creating_seq_nrs = [after_nr - 1]
        self._previous_after_nr = after_nr
        autograd["creating_seq_nrs"] = creating_seq_nrs
        self._creating_seq_context = context if creating_seq_nrs else None

    def post_process(self, graph: fx.Graph) -> None:
        """Link backward ops to operators that recorded their autograd sequence IDs."""
        sequence_to_forward_ids: dict[int, list[int]] = {}
        for node in graph.nodes:
            if node.op != "call_function" or node.target is operator.getitem:
                continue
            autograd = node_autograd_meta(node)
            creating_seq_nrs = cast(list[int], autograd.get("creating_seq_nrs", []))
            for creating_seq_nr in creating_seq_nrs:
                sequence_to_forward_ids.setdefault(creating_seq_nr, []).append(
                    node_fx_id(node)
                )

        for node in graph.nodes:
            if node.op == "output":
                continue
            autograd = node_autograd_meta(node)
            if autograd.get("phase") != "backward":
                continue
            executing_seq_nr = cast(int | None, autograd.get("executing_seq_nr"))
            if executing_seq_nr is None:
                continue
            autograd["forward_node_ids"] = sequence_to_forward_ids.get(
                executing_seq_nr, []
            ).copy()


class GradientDependencyMetadataProvider(MetadataProvider):
    def __init__(self) -> None:
        self._gradient_inputs: set[WeakIdRef] = set()
        self._autograd_hooks: dict[torch.autograd.graph.Node, RemovableHandle] = {}

    def _register_autograd_node(self, node: torch.autograd.graph.Node) -> None:
        pending = [node]
        while pending:
            current = pending.pop()
            if current in self._autograd_hooks:
                continue

            def prehook(
                grad_outputs: tuple[torch.Tensor | None, ...],
            ) -> None:
                self._gradient_inputs.update(
                    WeakIdRef(value) for value in grad_outputs if value is not None
                )

            self._autograd_hooks[current] = current.register_prehook(prehook)
            pending.extend(
                next_node
                for next_node, _output_index in current.next_functions
                if next_node is not None
            )

    def before_call(
        self,
        node_meta: NodeMetadata,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        autograd = cast(NodeMetadata, node_meta["autograd"])
        autograd["depends_on_gradient"] = (
            any(
                WeakIdRef(value) in self._gradient_inputs
                for value in tree_flatten((args, kwargs))[0]
                if isinstance(value, torch.Tensor)
            )
            if autograd.get("phase") == "backward"
            else None
        )

    def post_process(self, graph: fx.Graph) -> None:
        """Propagate direct gradient-input observations through backward FX edges."""
        gradient_nodes: set[fx.Node] = set()
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            autograd = cast(NodeMetadata, node_metadata(node)["autograd"])
            if autograd.get("phase") != "backward":
                continue
            if any(input_node in gradient_nodes for input_node in node.all_input_nodes):
                autograd["depends_on_gradient"] = True
            if autograd.get("depends_on_gradient") is True:
                gradient_nodes.add(node)


def _alias_tensor_type(node: fx.Node, storage_id: int) -> str | None:
    producer = node
    return_index = 0
    while producer.target is operator.getitem:
        return_index = cast(int, producer.args[1])
        producer = cast(fx.Node, producer.args[0])

    inputs = producer.all_input_nodes
    schema = getattr(producer.target, "_schema", None)
    if schema is not None and schema.returns:
        return_index = return_index if len(schema.returns) > 1 else 0
        if schema.returns[return_index].alias_info is not None:
            schema_info = _SchemaInfo(schema)
            output = _SchemaArgument(_SchemaArgType.output, return_index)
            inputs = tree_flatten(
                [
                    producer.args[i]
                    if i < len(producer.args)
                    else producer.kwargs.get(arg.name)
                    for i, arg in enumerate(schema.arguments)
                    if schema_info.may_contain_alias(
                        _SchemaArgument(_SchemaArgType.input, i), output
                    )
                ]
            )[0]

    types = {
        cast(str, meta["type"])
        for input_node in inputs
        if isinstance(input_node, fx.Node)
        if (meta := node_tensor_meta(input_node)).get("storage_id") == storage_id
    }
    return types.pop() if len(types) == 1 else None


class MetadataManager:
    def __init__(self, graph: fx.Graph, providers: list[MetadataProvider]) -> None:
        self._graph = graph
        self._autograd_metadata = AutogradMetadataProvider()
        self._gradient_dependency_metadata = GradientDependencyMetadataProvider()
        self._providers = [
            *providers,
            self._autograd_metadata,
            self._gradient_dependency_metadata,
        ]
        self._storage_ids: dict[StorageWeakRef, int] = {}

    def new_node_metadata(self) -> NodeMetadata:
        return {FX_NODE_ID_KEY: len(self._graph.nodes)}

    def before_call(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> NodeMetadata:
        node_meta = self.new_node_metadata()
        for provider in self._providers:
            provider.before_call(node_meta, args, kwargs)
        return node_meta

    def after_call(
        self,
        node: fx.Node,
        returns: Any,
    ) -> None:
        """Finalize call metadata before output projections copy it."""
        for provider in self._providers:
            provider.after_call(node, returns)

    def add_tensor_meta(
        self,
        node: fx.Node,
        tensor: torch.Tensor,
    ) -> None:
        storage_ref = StorageWeakRef(tensor.untyped_storage())
        storage_id = self._storage_ids.setdefault(storage_ref, len(self._storage_ids))
        autograd = node_autograd_meta(node)
        if isinstance(tensor, torch.nn.Parameter):
            recorded_type = "weight"
        elif node.op == "placeholder":
            recorded_type = "input"
        elif node.op == "get_attr":
            recorded_type = "constant"
        elif (alias_type := _alias_tensor_type(node, storage_id)) is not None:
            recorded_type = alias_type
        elif autograd.get("phase") == "backward":
            recorded_type = "backward_tensor"
        else:
            recorded_type = "forward_tensor"
        with torch._C._DisableTorchDispatch():
            node_metadata(node)["tensor"] = {
                "shape": list(tensor.shape),
                "stride": (
                    list(tensor.stride()) if tensor.layout is torch.strided else None
                ),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "device": str(tensor.device),
                "layout": str(tensor.layout).removeprefix("torch."),
                "element_size": tensor.element_size(),
                "nbytes": tensor.numel() * tensor.element_size(),
                "requires_grad": tensor.requires_grad,
                "is_leaf": tensor.is_leaf,
                "storage_id": storage_id,
                "type": recorded_type,
            }

    def new_projection_metadata(self, source: fx.Node) -> NodeMetadata:
        source_meta = node_metadata(source)
        node_meta = self.new_node_metadata()
        autograd = source_meta.get("autograd")
        if isinstance(autograd, dict):
            node_meta["autograd"] = deepcopy(autograd)
        return node_meta

    def post_process(self) -> None:
        for provider in self._providers:
            provider.post_process(self._graph)

    @contextmanager
    def enable_autograd_hooks(self) -> Iterator[None]:
        provider = self._gradient_dependency_metadata
        original_run_backward = torch.autograd._engine_run_backward
        original_graph_run_backward = torch.autograd.graph._engine_run_backward
        tracing_thread = get_ident()

        def run_backward(
            outputs: Sequence[torch.Tensor | torch.autograd.graph.GradientEdge],
            grad_outputs: Sequence[torch.Tensor | None],
            *args: Any,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, ...]:
            if get_ident() != tracing_thread:
                return original_run_backward(outputs, grad_outputs, *args, **kwargs)
            provider._gradient_inputs.update(
                WeakIdRef(value) for value in grad_outputs if value is not None
            )
            for output in outputs:
                grad_node = (
                    output.node
                    if isinstance(output, torch.autograd.graph.GradientEdge)
                    else output.grad_fn
                )
                if grad_node is not None:
                    provider._register_autograd_node(grad_node)
            return original_run_backward(outputs, grad_outputs, *args, **kwargs)

        try:
            # Each engine entry sees its complete graph, including nested backward.
            torch.autograd._engine_run_backward = run_backward
            torch.autograd.graph._engine_run_backward = run_backward
            yield
        finally:
            torch.autograd._engine_run_backward = original_run_backward
            torch.autograd.graph._engine_run_backward = original_graph_run_backward
            for handle in provider._autograd_hooks.values():
                handle.remove()
            provider._autograd_hooks.clear()
