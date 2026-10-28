# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
import torch.fx as fx
from torch._opaque_base import OpaqueBase
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torch_cudagraph_fx._cudagraph_annotations import mark_kernels
from torch_cudagraph_fx._metadata import (
    FX_NODE_ANNOTATION_KEY,
    FX_NODE_ID_KEY,
    node_fx_id,
    node_stream,
)
from torch_cudagraph_fx._references import is_trace_reference


def is_graph_variable(value: Any) -> bool:
    """Whether a value gets a graph binding; enums are embedded as literals."""
    return (
        isinstance(value, (torch.Tensor, OpaqueBase)) or is_trace_reference(value)
    ) and not isinstance(value, Enum)


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
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        *,
        in_spec: TreeSpec | None = None,
        out_spec: TreeSpec | None = None,
    ) -> None:
        super().__init__(gm)
        self._in_spec = in_spec
        self._out_spec = out_spec
        self._side_streams: dict[int, torch.cuda.Stream] = {}

    def _stream_for(self, stream_id: int) -> torch.cuda.Stream | None:
        if stream_id == 0:
            return None  # run on whatever stream is current
        stream = self._side_streams.get(stream_id)
        if stream is None:
            stream = torch.cuda.Stream()
            self._side_streams[stream_id] = stream
        return stream

    def _record_tensor_streams(self, value: Any, stream: torch.cuda.Stream) -> None:
        # Tell the caching allocator each tensor is also used on ``stream`` (it may
        # have been allocated on another one), so its memory is not recycled while
        # this stream still has pending work touching it. Memory lifetime only --
        # cross-stream ordering comes from the event nodes.
        for leaf in tree_flatten(value)[0]:
            if isinstance(leaf, torch.Tensor) and leaf.device.type == "cuda":
                leaf.record_stream(stream)

    def run_node(self, n: fx.Node) -> Any:
        if n.op != "call_function":
            return super().run_node(n)
        stream = self._stream_for(node_stream(n) or 0)
        annotation = {
            FX_NODE_ANNOTATION_KEY: True,
            FX_NODE_ID_KEY: node_fx_id(n),
        }
        with torch.cuda.stream(stream), mark_kernels(annotation):
            replay_stream = torch.cuda.current_stream()
            with self._set_current_node(n):
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                result = getattr(self, n.op)(n.target, args, kwargs)
            self._record_tensor_streams((args, kwargs, result), replay_stream)
            return result

    def run_fn(self, *args: Any, **kwargs: Any) -> Any:
        flat, in_spec = tree_flatten((args, kwargs))
        if self._in_spec is not None and in_spec != self._in_spec:
            raise RuntimeError("replay inputs don't match the traced pytree structure")
        # One arg per placeholder bind_inputs created, in flattened order.
        inputs = [a for a in flat if is_graph_variable(a)]
        out = self.run(*inputs)
        if self._out_spec is not None:
            return tree_unflatten(list(out), self._out_spec)
        return out
