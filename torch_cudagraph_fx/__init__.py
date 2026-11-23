# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch_cudagraph_fx._compile import cgfx_compile, CompiledFunction
from torch_cudagraph_fx._events import (
    CudaGraphEvent,
    new_event,
    record_event,
    wait_event,
)
from torch_cudagraph_fx._indexed_graph import IndexedGraph, Op, OpGroup, Scope
from torch_cudagraph_fx._interpreter import fused_op
from torch_cudagraph_fx._kernel_cache import substitute_with_kernels
from torch_cudagraph_fx._references import trace_reference
from torch_cudagraph_fx._schedule import SchedulePlan
from torch_cudagraph_fx._tracer import (
    CudaGraphFxTrace,
    CudaGraphState,
    make_cudagraph_fx,
)
from torch_cudagraph_fx.custom_ops import (
    custom_op,
    OpaqueRef,
    register_dataclass_pytree,
    trace_dataclass,
    trace_leaf,
)

__all__ = [
    "cgfx_compile",
    "CompiledFunction",
    "OpaqueRef",
    "trace_dataclass",
    "register_dataclass_pytree",
    "CudaGraphEvent",
    "CudaGraphFxTrace",
    "CudaGraphState",
    "IndexedGraph",
    "fused_op",
    "Op",
    "OpGroup",
    "SchedulePlan",
    "Scope",
    "custom_op",
    "trace_leaf",
    "substitute_with_kernels",
    "trace_reference",
    "make_cudagraph_fx",
    "new_event",
    "record_event",
    "wait_event",
]
