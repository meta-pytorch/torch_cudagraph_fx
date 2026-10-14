# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shim over the CUDA-graph kernel-annotation facility.

Prefers torch's built-in ``torch.cuda._graph_annotations`` when present and
otherwise uses a self-contained polyfill
(:mod:`torch_cudagraph_fx._cudagraph_annotations_polyfill`). Either backend needs the
``cudaGraphNodeGetToolsId`` driver API (cuda.bindings >= 13.1 / cuda-compat);
without it :func:`annotations_available` returns False and the tracer skips the
capture-completeness check.
"""

from __future__ import annotations

from collections.abc import Mapping

try:
    from torch.cuda._graph_annotations import (
        get_kernel_annotations,
        is_available as annotations_available,
        mark_kernels,
    )

    _USE_TORCH_GRAPH_ANNOTATIONS = True
except ImportError:
    from torch_cudagraph_fx._cudagraph_annotations_polyfill import (
        annotations_available,
        get_kernel_annotations,
        mark_kernels,
    )

    _USE_TORCH_GRAPH_ANNOTATIONS = False


try:
    from torch.cuda.graph_annotations import (
        get_kernel_py_stacks as maybe_get_kernel_py_stacks,
    )
except ImportError:

    def maybe_get_kernel_py_stacks() -> Mapping[int, str] | None:
        return None


__all__ = [
    "annotations_available",
    "get_kernel_annotations",
    "maybe_get_kernel_py_stacks",
    "mark_kernels",
]
