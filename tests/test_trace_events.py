# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import expecttest
import torch
from torch_cudagraph_fx import (
    CudaGraphEvent,
    make_cudagraph_fx,
    new_event,
    record_event,
    wait_event,
)
from torch_cudagraph_fx._events import trace_cuda_event_constructors
from utils import requires_gpus


class TestCudaGraphFxEvents(expecttest.TestCase):
    @requires_gpus(1)
    def test_events_are_captured_as_nodes_with_streams(self) -> None:
        side = torch.cuda.Stream()

        def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
            a = x * 2.0
            done = new_event()
            record_event(done)
            with torch.cuda.stream(side):
                wait_event(done)
                b = torch.relu(a) + 1.0
                side_done = new_event()
                record_event(side_done)
            wait_event(side_done)
            return (b,)

        x = torch.randn(128, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        mul_tensor = torch.ops.aten.mul.Tensor(arg_0, 2.0);  arg_0 = None
        # stream 0
        new_event_default = torch.ops.torch_cudagraph_fx.new_event.default()
        # stream 0
        record_event_default = torch.ops.torch_cudagraph_fx.record_event.default(new_event_default);  record_event_default = None
        # stream 1
        wait_event_default = torch.ops.torch_cudagraph_fx.wait_event.default(new_event_default);  new_event_default = wait_event_default = None
        # stream 1
        relu_default = torch.ops.aten.relu.default(mul_tensor);  mul_tensor = None
        # stream 1
        add_tensor = torch.ops.aten.add.Tensor(relu_default, 1.0);  relu_default = None
        # stream 1
        new_event_default_1 = torch.ops.torch_cudagraph_fx.new_event.default()
        # stream 1
        record_event_default_1 = torch.ops.torch_cudagraph_fx.record_event.default(new_event_default_1);  record_event_default_1 = None
        # stream 0
        wait_event_default_1 = torch.ops.torch_cudagraph_fx.wait_event.default(new_event_default_1);  new_event_default_1 = wait_event_default_1 = None
        return (add_tensor,)""",
        )

        (expected,) = fn(x)
        (got,) = fx_trace.replay(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @requires_gpus(1)
    def test_patched_cuda_event_methods_are_traced(self) -> None:
        side = torch.cuda.Stream()

        def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
            a = x * 2.0
            ready = torch.cuda.Event()
            ready.record()
            with torch.cuda.stream(side):
                ready.wait()
                b = torch.relu(a) + 1.0
                done = torch.cuda.Event()
                done.record()
            done.wait()
            return (b,)

        x = torch.randn(64, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        mul_tensor = torch.ops.aten.mul.Tensor(arg_0, 2.0);  arg_0 = None
        # stream 0
        new_event_default = torch.ops.torch_cudagraph_fx.new_event.default()
        # stream 0
        record_event_default = torch.ops.torch_cudagraph_fx.record_event.default(new_event_default);  record_event_default = None
        # stream 1
        wait_event_default = torch.ops.torch_cudagraph_fx.wait_event.default(new_event_default);  new_event_default = wait_event_default = None
        # stream 1
        relu_default = torch.ops.aten.relu.default(mul_tensor);  mul_tensor = None
        # stream 1
        add_tensor = torch.ops.aten.add.Tensor(relu_default, 1.0);  relu_default = None
        # stream 1
        new_event_default_1 = torch.ops.torch_cudagraph_fx.new_event.default()
        # stream 1
        record_event_default_1 = torch.ops.torch_cudagraph_fx.record_event.default(new_event_default_1);  record_event_default_1 = None
        # stream 0
        wait_event_default_1 = torch.ops.torch_cudagraph_fx.wait_event.default(new_event_default_1);  new_event_default_1 = wait_event_default_1 = None
        return (add_tensor,)""",
        )

        (expected,) = fn(x)
        (got,) = fx_trace.replay(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @requires_gpus(1)
    def test_side_stream_is_recorded_in_metadata(self) -> None:
        side = torch.cuda.Stream()

        def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
            a = x + 1
            ev = new_event()
            record_event(ev)
            with torch.cuda.stream(side):
                wait_event(ev)
                b = a.sin()
                ev2 = new_event()
                record_event(ev2)
            wait_event(ev2)
            return (b,)

        x = torch.randn(64, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(arg_0, 1);  arg_0 = None
        # stream 0
        new_event_default = torch.ops.torch_cudagraph_fx.new_event.default()
        # stream 0
        record_event_default = torch.ops.torch_cudagraph_fx.record_event.default(new_event_default);  record_event_default = None
        # stream 1
        wait_event_default = torch.ops.torch_cudagraph_fx.wait_event.default(new_event_default);  new_event_default = wait_event_default = None
        # stream 1
        sin_default = torch.ops.aten.sin.default(add_tensor);  add_tensor = None
        # stream 1
        new_event_default_1 = torch.ops.torch_cudagraph_fx.new_event.default()
        # stream 1
        record_event_default_1 = torch.ops.torch_cudagraph_fx.record_event.default(new_event_default_1);  record_event_default_1 = None
        # stream 0
        wait_event_default_1 = torch.ops.torch_cudagraph_fx.wait_event.default(new_event_default_1);  new_event_default_1 = wait_event_default_1 = None
        return (sin_default,)""",
        )

    @requires_gpus(1)
    def test_cross_stream_replay_is_numerically_correct(self) -> None:
        side = torch.cuda.Stream()

        def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
            a = x * 2.0
            done = new_event()
            record_event(done)
            with torch.cuda.stream(side):
                wait_event(done)
                b = torch.relu(a) + 1.0
                side_done = new_event()
                record_event(side_done)
            wait_event(side_done)
            return (b * 3.0,)

        x = torch.randn(4096, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        mul_tensor = torch.ops.aten.mul.Tensor(arg_0, 2.0);  arg_0 = None
        # stream 0
        new_event_default = torch.ops.torch_cudagraph_fx.new_event.default()
        # stream 0
        record_event_default = torch.ops.torch_cudagraph_fx.record_event.default(new_event_default);  record_event_default = None
        # stream 1
        wait_event_default = torch.ops.torch_cudagraph_fx.wait_event.default(new_event_default);  new_event_default = wait_event_default = None
        # stream 1
        relu_default = torch.ops.aten.relu.default(mul_tensor);  mul_tensor = None
        # stream 1
        add_tensor = torch.ops.aten.add.Tensor(relu_default, 1.0);  relu_default = None
        # stream 1
        new_event_default_1 = torch.ops.torch_cudagraph_fx.new_event.default()
        # stream 1
        record_event_default_1 = torch.ops.torch_cudagraph_fx.record_event.default(new_event_default_1);  record_event_default_1 = None
        # stream 0
        wait_event_default_1 = torch.ops.torch_cudagraph_fx.wait_event.default(new_event_default_1);  new_event_default_1 = wait_event_default_1 = None
        # stream 0
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor, 3.0);  add_tensor = None
        return (mul_tensor_1,)""",
        )

        (expected,) = fn(x)
        (got,) = fx_trace.replay(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @requires_gpus(1)
    def test_opaque_event_input_becomes_placeholder(self) -> None:
        def fn(event: CudaGraphEvent, x: torch.Tensor) -> tuple[torch.Tensor]:
            record_event(event)
            return (x * 2.0,)

        x = torch.randn(64, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(new_event(), x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        # stream 0
        record_event_default = torch.ops.torch_cudagraph_fx.record_event.default(arg_0);  arg_0 = record_event_default = None
        # stream 0
        mul_tensor = torch.ops.aten.mul.Tensor(arg_1, 2.0);  arg_1 = None
        return (mul_tensor,)""",
        )

        (expected,) = fn(new_event(), x)
        (got,) = fx_trace.replay(new_event(), x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


@requires_gpus(1)
def test_trace_cuda_event_constructors_patches_and_restores() -> None:
    original_event = torch.cuda.Event
    original_streams_event = torch.cuda.streams.Event
    with trace_cuda_event_constructors():
        assert isinstance(torch.cuda.Event(), CudaGraphEvent)
        assert isinstance(torch.cuda.streams.Event(), CudaGraphEvent)
    assert torch.cuda.Event is original_event
    assert torch.cuda.streams.Event is original_streams_event
