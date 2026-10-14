# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The validation criterion from DESIGN.md.

A trace is valid if, run eagerly, it issues *the same GPU ops, in the same order,
on the same streams* as its source function -- per the PyTorch profiler. We
profile each run, extract the ``(op_name, stream)`` sequence, and compare.

NB: profile the eager run, not the CUDA-graph *replay* -- replay collapses every
op onto one stream and loses the per-op stream structure the criterion is about.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.profiler import profile, ProfilerActivity

KernelStreamSeq = list[tuple[str, int]]


def _extract(prof: profile) -> KernelStreamSeq:
    # Read kineto events in memory (no chrome-trace export to disk). A GPU-side
    # event's ``device_resource_id`` is its CUDA stream; normalize raw stream ids
    # to small first-seen ids so two runs compare on relative stream structure
    # rather than the physical stream ids, which are not stable run-to-run.
    events = [
        e
        for e in prof.profiler.kineto_results.events()
        if str(e.device_type()) == "DeviceType.CUDA"
    ]
    events.sort(key=lambda e: e.start_ns())

    normalized: dict[int, int] = {}
    seq: KernelStreamSeq = []
    for e in events:
        stream = e.device_resource_id()
        if stream not in normalized:
            normalized[stream] = len(normalized)
        seq.append((e.name(), normalized[stream]))
    return seq


def kernel_stream_sequence(fn: Callable[[], Any]) -> KernelStreamSeq:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return _extract(prof)


def assert_trace_matches(
    reference: Callable[[], Any],
    candidate: Callable[[], Any],
) -> KernelStreamSeq:
    ref_seq = kernel_stream_sequence(reference)
    cand_seq = kernel_stream_sequence(candidate)
    if ref_seq != cand_seq:
        raise AssertionError(
            "trace does not satisfy the validation criterion\n"
            f"  reference ({len(ref_seq)} kernels):\n"
            + "\n".join(f"    {i}: {k} @stream {s}" for i, (k, s) in enumerate(ref_seq))
            + f"\n  candidate ({len(cand_seq)} kernels):\n"
            + "\n".join(
                f"    {i}: {k} @stream {s}" for i, (k, s) in enumerate(cand_seq)
            )
        )
    return ref_seq
