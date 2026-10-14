# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest
import torch

_T = TypeVar("_T")


def requires_gpus(count: int) -> Callable[[_T], _T]:
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_marker = getattr(pytest.mark, f"gpus_needed_{count}")
    skip_marker = pytest.mark.skipif(
        available < count,
        reason=f"needs {count} GPU(s), found {available}",
    )

    def decorator(obj: _T) -> _T:
        return skip_marker(gpu_marker(obj))

    return decorator
