# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import expecttest
import pytest
import torch
from torch_cudagraph_fx import cgfx_compile, fused_op, SchedulePlan
from torch_cudagraph_fx._metadata import node_metadata
from utils import requires_gpus


class Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 16, bias=False)
        self.scale = torch.nn.Parameter(torch.ones(16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x)) * self.scale + x


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(), Layer()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def configure_schedule(plan: SchedulePlan) -> None:
    ig = plan.indexed_graph
    for i in ig.fwd.layers:
        for name, phase in (("fwd", ig.fwd), ("bwd", ig.bwd)):
            plan.fuse(f"{name}_layer", [phase.layers[i]])


class TestCudaGraphFxCompile(expecttest.TestCase):
    @requires_gpus(1)
    def test_compiled_step_matches_eager_on_every_call(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            model = Model().cuda()
            params = list(model.parameters())
            traced_calls = 0

            def step(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, ...]:
                nonlocal traced_calls
                traced_calls += 1
                loss = (model(x) * scale).square().sum()
                return (loss, *torch.autograd.grad(loss, params))

            compiled = cgfx_compile(
                step, schedule=configure_schedule, kernel_cache_dir=cache_dir
            )
            for _ in range(3):
                x = torch.randn(8, 16, device="cuda")
                got = compiled(x, 0.5)
                expected = step(x, 0.5)
                for g, e in zip(got, expected, strict=True):
                    torch.testing.assert_close(g, e, rtol=0, atol=0)
            # One trace for three calls, plus the three eager reference calls.
            assert traced_calls == 4

            kinds = {
                node_metadata(n)["kernel_cache"]["kind"]
                for n in compiled.graph_module.graph.nodes
                if n.target is fused_op
            }
            assert kinds == {"reference"} and any(cache_dir.glob("pat_*/*.py"))
            with pytest.raises(ValueError, match="differs from the traced value"):
                compiled(x, 0.25)

    @requires_gpus(1)
    def test_lowered_graph_replays_in_a_caller_capture(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            model = Model().cuda()
            params = list(model.parameters())
            streams = []

            def step(x: torch.Tensor) -> torch.Tensor:
                streams.append(torch.cuda.current_stream())
                loss = model(x).square().sum()
                loss.backward()
                return loss

            static_x = torch.randn(8, 16, device="cuda")
            pool, stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                step(static_x)  # warm up eagerly, as a caller does before capture
            model.zero_grad(set_to_none=True)
            compiled = cgfx_compile(
                step,
                schedule=configure_schedule,
                kernel_cache_dir=tmp_dir,
                pool=pool,
                stream=stream,
            )
            lowered = compiled.lower(static_x)
            assert streams == [stream, stream]
            # The gradients tracing assigned hold no memory until replay rebinds them.
            assert all(p.grad is not None and p.grad.numel() == 0 for p in params)
            graph = torch.cuda.CUDAGraph()
            with (
                torch.no_grad(),
                torch.cuda.graph(
                    graph, pool=pool, stream=stream, capture_error_mode="thread_local"
                ),
            ):
                static_loss = lowered.replay(static_x)
            for _ in range(2):
                x = torch.randn(8, 16, device="cuda")
                static_x.copy_(x)
                graph.replay()
                loss = model(x).square().sum()
                expected = torch.autograd.grad(loss, params)
                torch.testing.assert_close(static_loss, loss, rtol=0, atol=0)
                for p, e in zip(params, expected, strict=True):
                    torch.testing.assert_close(p.grad, e, rtol=0, atol=0)
            kinds = {
                node_metadata(n)["kernel_cache"]["kind"]
                for n in lowered.graph_module.graph.nodes
                if n.target is fused_op
            }
            assert kinds == {"reference"}
            with pytest.raises(RuntimeError, match="has not been called yet"):
                compiled.graph_module

    @requires_gpus(1)
    def test_schedule_runs_after_trace_once_per_compilation(self) -> None:
        events = []

        def configure(plan: SchedulePlan) -> None:
            events.append("schedule")
            plan.fuse("sin", [plan.indexed_graph.fwd.sin])

        @cgfx_compile(schedule=configure)
        def step(x: torch.Tensor) -> torch.Tensor:
            events.append("trace")
            return x.sin().cos()

        assert events == []
        for _ in range(3):
            x = torch.randn(4, device="cuda")
            torch.testing.assert_close(step(x), x.sin().cos(), rtol=0, atol=0)
            assert events == ["trace", "schedule"]
        assert any(n.target is fused_op for n in step.graph_module.graph.nodes)
        new_step = cgfx_compile(step.__wrapped__, schedule=configure)
        assert events == ["trace", "schedule"]
        x = torch.randn(4, device="cuda")
        torch.testing.assert_close(new_step(x), x.sin().cos(), rtol=0, atol=0)
        assert events == ["trace", "schedule", "trace", "schedule"]
