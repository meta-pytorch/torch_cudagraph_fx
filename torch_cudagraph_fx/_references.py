# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import Any

_reference_types: tuple[type[Any], ...] = ()


def trace_reference(cls: type[Any]) -> None:
    """Declare a weak-referenceable class as an identity-preserved trace leaf.

    Set CGFX_TRACING=1 before registration. References remain Python objects;
    dispatchless custom operators represent them with an Any schema slot.
    """
    global _reference_types
    if os.environ.get("CGFX_TRACING") == "1" and cls not in _reference_types:
        _reference_types += (cls,)


def is_trace_reference(value: Any) -> bool:
    return isinstance(value, _reference_types)
