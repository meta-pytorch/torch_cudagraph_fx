# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch_cudagraph_fx._events import (
    CudaGraphEvent,
    new_event,
    record_event,
    wait_event,
)
from torch_cudagraph_fx._references import trace_reference
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
    "OpaqueRef",
    "trace_dataclass",
    "register_dataclass_pytree",
    "CudaGraphEvent",
    "CudaGraphFxTrace",
    "CudaGraphState",
    "custom_op",
    "trace_leaf",
    "trace_reference",
    "make_cudagraph_fx",
    "new_event",
    "record_event",
    "wait_event",
]
