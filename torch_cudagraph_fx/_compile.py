# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tracing records operators without running them, so replay executes each
call's GPU work exactly once. Captured graphs bind tensor addresses, so
new tensor arguments are copied into the captured inputs."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import Any, Generic, ParamSpec, TypeVar

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch_cudagraph_fx._kernel_cache import substitute_with_kernels
from torch_cudagraph_fx._schedule import SchedulePlan
from torch_cudagraph_fx._tracer import (
    _check_uncaptured_kernels,
    annotated_capture,
    CudaGraphFxTrace,
    make_cudagraph_fx,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")
_LITERALS = (type(None), bool, int, float, complex, str, bytes, Enum, torch.dtype)


@dataclass
class _Compiled:
    out_spec: pytree.TreeSpec
    graph_module: fx.GraphModule
    cuda_graph: torch.cuda.CUDAGraph
    in_spec: pytree.TreeSpec
    flat_inputs: list[Any]
    flat_outputs: list[Any]


class CompiledFunction(Generic[_P, _R]):
    def __init__(
        self,
        fn: Callable[_P, _R],
        *,
        schedule: Callable[[SchedulePlan], None] | None,
        kernel_cache_dir: str | PathLike[str] | None,
        pool: tuple[int, int] | None,
        stream: torch.cuda.Stream | None,
    ) -> None:
        functools.update_wrapper(self, fn)
        self._name: str = getattr(fn, "__name__", type(fn).__name__)
        self._fn = fn
        self._schedule = schedule
        self._kernel_cache_dir = kernel_cache_dir
        self._pool = pool
        self._stream = stream
        self._compiled: _Compiled | None = None

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        flat, in_spec = pytree.tree_flatten((args, kwargs))
        if self._compiled is None:
            self._compiled = self._compile(flat, in_spec, args, kwargs)
        else:
            self._copy_in(flat, in_spec)
        compiled = self._compiled
        compiled.cuda_graph.replay()
        return pytree.tree_unflatten(compiled.flat_outputs, compiled.out_spec)

    @property
    def graph_module(self) -> fx.GraphModule:
        return self._require().graph_module

    @property
    def cuda_graph(self) -> torch.cuda.CUDAGraph:
        return self._require().cuda_graph

    def _require(self) -> _Compiled:
        if self._compiled is None:
            raise RuntimeError(f"{self._name} has not been called yet")
        return self._compiled

    def lower(self, *args: _P.args, **kwargs: _P.kwargs) -> CudaGraphFxTrace:
        """Trace, schedule and substitute kernels as the first call does, without
        running or capturing anything.

        The result's ``replay`` runs the graph on the current stream, so a caller
        that owns CUDA graph capture can capture it. Replay it with autograd
        disabled: the trace already records the backward operators.
        """
        trace = make_cudagraph_fx(
            self._fn, allow_missing_kernels=True, pool=self._pool, stream=self._stream
        )(*args, **kwargs)
        plan = SchedulePlan(trace.indexed_graph)
        if self._schedule is not None:
            self._schedule(plan)
        graph_module = plan.apply().graph_module
        if self._kernel_cache_dir is not None:
            substitute_with_kernels(graph_module, self._kernel_cache_dir)
        return CudaGraphFxTrace(
            graph_module, trace.in_spec, trace.out_spec, escaped=trace.escaped
        )

    def _compile(
        self,
        flat: list[Any],
        in_spec: pytree.TreeSpec,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _Compiled:
        lowered = self.lower(*args, **kwargs)
        cuda_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with (
            torch.no_grad(),
            torch.autograd.grad_mode.set_multithreading_enabled(False),
            annotated_capture(cuda_graph, pool=self._pool, stream=self._stream),
        ):
            flat_outputs = pytree.tree_leaves(lowered.replay(*args, **kwargs))
        _check_uncaptured_kernels(cuda_graph, allow_missing_kernels=True)
        return _Compiled(
            lowered.out_spec,
            lowered.graph_module,
            cuda_graph,
            in_spec,
            flat,
            flat_outputs,
        )

    def _copy_in(self, flat: list[Any], in_spec: pytree.TreeSpec) -> None:
        compiled = self._require()
        if in_spec != compiled.in_spec:
            raise ValueError(f"{self._name}: argument structure changed since tracing")
        copies = []
        for i, (captured, value) in enumerate(zip(compiled.flat_inputs, flat)):
            if value is captured:
                continue
            if isinstance(captured, torch.Tensor):
                if not isinstance(value, torch.Tensor) or (
                    value.shape,
                    value.stride(),
                    value.dtype,
                    value.device,
                ) != (
                    captured.shape,
                    captured.stride(),
                    captured.dtype,
                    captured.device,
                ):
                    raise ValueError(
                        f"{self._name}: tensor argument {i} changed shape, stride, "
                        "dtype or device since tracing"
                    )
                copies.append((captured, value))
            elif not (
                isinstance(captured, _LITERALS)
                and type(value) is type(captured)
                and value == captured
            ):
                raise ValueError(
                    f"{self._name}: argument {i} differs from the traced value "
                    f"{captured!r}; non-tensor arguments must stay the same"
                )
        # A rejected call must leave every captured input unchanged.
        for captured, value in copies:
            captured.copy_(value)


def cgfx_compile(
    fn: Callable[_P, _R] | None = None,
    *,
    schedule: Callable[[SchedulePlan], None] | None = None,
    kernel_cache_dir: str | PathLike[str] | None = None,
    pool: tuple[int, int] | None = None,
    stream: torch.cuda.Stream | None = None,
) -> CompiledFunction[_P, _R] | Callable[[Callable[_P, _R]], CompiledFunction[_P, _R]]:
    """Compile on the first call and replay subsequent calls.

    ``schedule(plan)`` adds fusion requests to a compiler-owned SchedulePlan and
    returns None. It runs once after tracing and is not called during replay.
    ``pool`` and ``stream`` set the memory pool and stream of both the tracing
    capture and the CUDA graph capture, as in ``torch.cuda.graph``.

    A schedule is written for one model:

    - Name each operator by its path in ``plan.indexed_graph``: the phase
      (``fwd`` or ``bwd``), the module path, the operator name and ``[k]``, the
      k-th call of that operator in that module during the phase. Compute k from
      the loop variables, e.g. ``mul[2 * m + 1]`` for microbatch ``m``, instead
      of matching paths with patterns, so that reading the schedule shows what
      each region holds.
    - List only the operators that launch kernels; ``SchedulePlan.fuse`` adds
      the views between them.

    ``for op in plan.indexed_graph: print(op.path)`` lists the paths; a path
    omits ``[0]`` when the module calls the operator once. When the model
    changes, a path that no longer exists raises a ``KeyError`` naming the
    closest ones. For a stack of blocks that compute
    ``h = x + attn(attn_norm(x))`` and ``h + ffn(ffn_norm(h))``, with an RMSNorm
    written as ``x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w``::

        def configure_schedule(plan: SchedulePlan) -> None:
            ig = plan.indexed_graph
            for i in ig.fwd.layers:
                F, B = ig.fwd.layers[i], ig.bwd.layers[i]
                plan.fuse(
                    "attn_residual_ffn_norm",
                    [
                        F.add[0],
                        F.ffn_norm.pow[0],
                        F.ffn_norm.mean[0],
                        F.ffn_norm.add[0],
                        F.ffn_norm.rsqrt[0],
                        F.ffn_norm.mul[0],
                        F.ffn_norm.mul[1],
                    ],
                )
                norm = B.ffn_norm
                plan.fuse(
                    "ffn_norm_bwd",
                    [
                        *norm.mul[0:8],
                        *norm.sum[0:2],
                        *norm.add[0:2],
                        *norm.pow[0:2],
                        norm.div[0],
                    ],
                )

        step = cgfx_compile(
            train_step, schedule=configure_schedule, kernel_cache_dir="kernels"
        )
    """

    def wrap(f: Callable[_P, _R]) -> CompiledFunction[_P, _R]:
        return CompiledFunction(
            f,
            schedule=schedule,
            kernel_cache_dir=kernel_cache_dir,
            pool=pool,
            stream=stream,
        )

    return wrap if fn is None else wrap(fn)
