# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Iterator

import expecttest
import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch_cudagraph_fx import make_cudagraph_fx
from torch_cudagraph_fx._metadata import node_fx_id
from utils import requires_gpus

WORLD_SIZE = 2


@pytest.fixture
def _fake_pg() -> Iterator[None]:
    torch.cuda.set_device(0)
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(
        backend="fake",
        rank=0,
        world_size=WORLD_SIZE,
        store=FakeStore(),
    )
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gpus(1)
def test_all_reduce_traces_collective(_fake_pg: None) -> None:
    def step(x: torch.Tensor) -> tuple[torch.Tensor]:
        work = dist.all_reduce(x, async_op=True)
        work.wait()
        return (x,)

    x = torch.full((8,), 1.0, device="cuda")
    fx_trace = make_cudagraph_fx(step)(x)
    (got,) = fx_trace.replay(x.clone())

    expecttest.TestCase().assertExpectedInline(
        fx_trace.print_readable(print_output=False),
        """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        _constant_0 = self._constant_0
        _constant_1 = self._constant_1
        # stream 0
        allreduce__default = torch.ops.c10d.allreduce_.default([arg_0], _constant_0, _constant_1, None);  arg_0 = _constant_0 = _constant_1 = None
        getitem = allreduce__default[0];  allreduce__default = None
        getitem_1 = getitem[0];  getitem = None
        return (getitem_1,)""",
        skip=1,
    )
    assert got.is_cuda

    nodes = list(fx_trace.graph.nodes)[:-1]
    assert [node_fx_id(node) for node in nodes] == list(range(len(nodes)))


@requires_gpus(1)
def test_reduce_scatter_traces_collective(_fake_pg: None) -> None:
    def step(x: torch.Tensor) -> tuple[torch.Tensor]:
        out = torch.empty(8, device="cuda")
        work = dist.reduce_scatter_tensor(out, x, async_op=True)
        work.wait()
        return (out,)

    x = torch.arange(16.0, device="cuda")
    fx_trace = make_cudagraph_fx(step)(x)
    (got,) = fx_trace.replay(x.clone())

    expecttest.TestCase().assertExpectedInline(
        fx_trace.print_readable(print_output=False),
        """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        empty_memory_format = torch.ops.aten.empty.memory_format([8], device = device(type='cuda'), pin_memory = False)
        _constant_0 = self._constant_0
        _constant_1 = self._constant_1
        # stream 0
        _reduce_scatter_base__default = torch.ops.c10d._reduce_scatter_base_.default(empty_memory_format, arg_0, _constant_0, _constant_1);  empty_memory_format = arg_0 = _constant_0 = _constant_1 = None
        getitem = _reduce_scatter_base__default[0];  _reduce_scatter_base__default = None
        return (getitem,)""",
        skip=1,
    )
    assert got.is_cuda


@requires_gpus(1)
def test_all_gather_traces_collective(_fake_pg: None) -> None:
    def step(x: torch.Tensor) -> tuple[torch.Tensor]:
        out = torch.empty(16, device="cuda")
        work = dist.all_gather_into_tensor(out, x, async_op=True)
        work.wait()
        return (out,)

    x = torch.arange(8.0, device="cuda")
    fx_trace = make_cudagraph_fx(step)(x)
    (got,) = fx_trace.replay(x.clone())

    expecttest.TestCase().assertExpectedInline(
        fx_trace.print_readable(print_output=False),
        """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        empty_memory_format = torch.ops.aten.empty.memory_format([16], device = device(type='cuda'), pin_memory = False)
        _constant_0 = self._constant_0
        # stream 0
        _allgather_base__default = torch.ops.c10d._allgather_base_.default(empty_memory_format, arg_0, _constant_0);  empty_memory_format = arg_0 = _constant_0 = None
        getitem = _allgather_base__default[0];  _allgather_base__default = None
        return (getitem,)""",
        skip=1,
    )
    assert got.is_cuda
