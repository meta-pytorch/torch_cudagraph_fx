# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

import torch
import torch.fx as fx
from torch._opaque_base import OpaqueBase
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torch_cudagraph_fx._cudagraph_annotations import mark_kernels
from torch_cudagraph_fx._metadata import (
    CUDAGRAPH_FX_METADATA_KEY,
    FREE_AFTER_KEY,
    FX_NODE_ANNOTATION_KEY,
    FX_NODE_ID_KEY,
    node_metadata,
    node_stream,
)
from torch_cudagraph_fx._references import is_trace_reference


def _cgfx_meta(node: fx.Node) -> dict[str, Any]:
    return node.meta.get(CUDAGRAPH_FX_METADATA_KEY) or {}


def _stream_id(node: fx.Node) -> int:
    execution = _cgfx_meta(node).get("execution") or {}
    return execution.get("stream") or 0


def is_graph_variable(value: Any) -> bool:
    """Whether a value gets a graph binding; enums are embedded as literals."""
    return (
        isinstance(value, (torch.Tensor, OpaqueBase)) or is_trace_reference(value)
    ) and not isinstance(value, Enum)


# Kernels that `substitute_with_kernels` loaded, by absolute cache file path.
loaded_kernels: dict[str, Callable[..., Any]] = {}


def fused_op(
    name: str, region: fx.GraphModule, *args: Any, cache_path: str | None = None
) -> Any:
    """Run the fused region ``name``, whose operators are the graph ``region``, or
    the ``run`` of its kernel cache file ``cache_path``.

    The region is interpreted: its generated code would resolve custom operators
    to overload packets, which return unflattened outputs.
    """
    if cache_path is not None:
        return loaded_kernels[cache_path](*args)
    return fx.Interpreter(region).run(*args)


class CudaGraphFxInterpreter(fx.Interpreter):
    """Execute a ``torch_cudagraph_fx.trace`` graph with the traced stream topology.

    Semantic stream id ``0`` intentionally stays on the caller's current stream:
    the capture stream during recording, or the default stream during plain
    replay. Higher ids map to freshly allocated side streams, created lazily and
    cached.

    Cross-stream ordering is preserved because the graph also contains the
    ``new_event`` / ``record_event`` / ``wait_event`` nodes (captured by the
    tracer); ``new_event`` mints a fresh event each replay and record/wait share
    it by dataflow, so multi-stream graphs replay correctly, not just with the
    right stream assignment.

    Each value is freed after its last use. A value that another stream uses
    is freed no earlier than tracing freed it: the traced program ordered its
    streams before freeing such memory, so allocator reuse stays safe without
    ``record_stream``, which would hold the memory until a capture ends.
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        *,
        in_spec: TreeSpec | None = None,
        out_spec: TreeSpec | None = None,
        escaped: Sequence[torch.Tensor] = (),
    ) -> None:
        super().__init__(gm)
        self._in_spec = in_spec
        self._out_spec = out_spec
        self._escaped = tuple(escaped)
        self._side_streams: dict[int, torch.cuda.Stream] = {}
        # fx.Interpreter frees a value after its last user runs, so a result no
        # node reads would stay alive until the graph finishes; eager drops it
        # immediately.
        for node in self.graph.nodes:
            if node.op == "call_function" and not node.users:
                self.user_to_last_uses.setdefault(node, []).append(node)
        self._free_no_earlier_than_tracing()

    def _free_no_earlier_than_tracing(self) -> None:
        nodes = list(self.graph.nodes)
        position: dict[int, int] = {}  # fx_node_id -> index of the node running it
        for i, node in enumerate(nodes):
            members = [node]
            if node.target is fused_op:
                members = self.fetch_attr(node.args[1].target).graph.nodes
            for member in members:
                fx_id = _cgfx_meta(member).get(FX_NODE_ID_KEY)
                if isinstance(fx_id, int):
                    position[fx_id] = i
        traced_ids = sorted(position)
        last_use = {
            n: user for user, freed in self.user_to_last_uses.items() for n in freed
        }
        index = {node: i for i, node in enumerate(nodes)}
        for node, user in last_use.items():
            free_after = _cgfx_meta(node).get(FREE_AFTER_KEY)
            streams = {_stream_id(u) for u in node.users if u.op != "output"}
            if not isinstance(free_after, int) or streams <= {_stream_id(node)}:
                continue
            k = bisect.bisect_left(traced_ids, free_after)
            later = nodes[position[traced_ids[k]]] if k < len(traced_ids) else None
            if later is None or index[later] > index[user]:
                self.user_to_last_uses[user].remove(node)
                if later is not None:
                    self.user_to_last_uses.setdefault(later, []).append(node)

    def _stream_for(self, stream_id: int) -> torch.cuda.Stream | None:
        if stream_id == 0:
            return None  # run on whatever stream is current
        stream = self._side_streams.get(stream_id)
        if stream is None:
            stream = torch.cuda.Stream()
            self._side_streams[stream_id] = stream
        return stream

    def run_node(self, n: fx.Node) -> Any:
        if n.op != "call_function":
            return super().run_node(n)
        stream = self._stream_for(node_stream(n) or 0)
        annotation = {FX_NODE_ANNOTATION_KEY: True}
        if isinstance(node_id := node_metadata(n).get(FX_NODE_ID_KEY), int):
            annotation[FX_NODE_ID_KEY] = node_id
        with torch.cuda.stream(stream), mark_kernels(annotation):
            with self._set_current_node(n):
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                return getattr(self, n.op)(n.target, args, kwargs)

    def run_fn(self, *args: Any, **kwargs: Any) -> Any:
        flat, in_spec = tree_flatten((args, kwargs))
        if self._in_spec is not None and in_spec != self._in_spec:
            raise RuntimeError("replay inputs don't match the traced pytree structure")
        # One arg per placeholder bind_inputs created, in flattened order.
        inputs = [a for a in flat if is_graph_variable(a)]
        out = self.run(*inputs)
        if self._escaped:
            out, values = out[: -len(self._escaped)], out[-len(self._escaped) :]
            with torch.no_grad():
                for escaped, value in zip(self._escaped, values, strict=True):
                    escaped.set_(value)
        if self._out_spec is not None:
            return tree_unflatten(list(out), self._out_spec)
        return out
