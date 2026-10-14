# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Traceable CUDA event ops.

A CUDA event is modeled as an **opaque object** (``CudaGraphEvent``, a torch
opaque *reference* type). During tracing we monkey-patch event construction to
return a ``CudaGraphEvent``, whose methods route to traceable custom ops that the
tracer can capture; stream APIs desugar into these same patched event calls.
Three custom ops are necessary: ``new_event``, ``record_event`` and ``wait_event``.

TODO: use dispatchless pytree custom ops to remove the dispatching overhead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch._library.opaque_object import is_opaque_type, register_opaque_type
from torch._opaque_base import OpaqueBase

_ORIGINAL_CUDA_EVENT = torch.cuda.Event


class CudaGraphEvent(OpaqueBase):
    """Reference-typed opaque object so event identity travels by FX dataflow."""

    def __init__(
        self,
        *,
        enable_timing: bool = False,
        blocking: bool = False,
        interprocess: bool = False,
        external: bool = False,
    ) -> None:
        self.event: torch.cuda.Event = _ORIGINAL_CUDA_EVENT(
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
            external=external,
        )

    def record(self, stream: torch.cuda.Stream | None = None) -> None:
        with torch.cuda.stream(stream):
            record_event(self)

    def wait(self, stream: torch.cuda.Stream | None = None) -> None:
        with torch.cuda.stream(stream):
            wait_event(self)

    def elapsed_time(self, end_event: CudaGraphEvent) -> float:
        return self.event.elapsed_time(_expect_cudagraph_event(end_event).event)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.event, name)


if not is_opaque_type(CudaGraphEvent):
    register_opaque_type(CudaGraphEvent, typ="reference")


@torch.library.custom_op("torch_cudagraph_fx::new_event", mutates_args=())
def new_event(
    enable_timing: bool = False,
    blocking: bool = False,
    interprocess: bool = False,
    external: bool = False,
) -> CudaGraphEvent:
    return CudaGraphEvent(
        enable_timing=enable_timing,
        blocking=blocking,
        interprocess=interprocess,
        external=external,
    )


@torch.library.custom_op("torch_cudagraph_fx::record_event", mutates_args=())
def record_event(event: CudaGraphEvent) -> None:
    _expect_cudagraph_event(event).event.record()


@torch.library.custom_op("torch_cudagraph_fx::wait_event", mutates_args=())
def wait_event(event: CudaGraphEvent) -> None:
    _expect_cudagraph_event(event).event.wait()


def _expect_cudagraph_event(event: object) -> CudaGraphEvent:
    if not isinstance(event, CudaGraphEvent):
        raise TypeError(
            "expected a CudaGraphEvent from torch_cudagraph_fx.new_event(); "
            f"got {type(event).__name__}. Native torch.cuda.Event objects are "
            "not traceable."
        )
    return event


@contextmanager
def trace_cuda_event_constructors() -> Iterator[None]:
    """Make Python CUDA event constructors return traceable events."""

    original_cuda_event = torch.cuda.Event
    original_cuda_streams_event = torch.cuda.streams.Event
    try:
        torch.cuda.Event = new_event
        torch.cuda.streams.Event = new_event
        yield
    finally:
        torch.cuda.Event = original_cuda_event
        torch.cuda.streams.Event = original_cuda_streams_event
