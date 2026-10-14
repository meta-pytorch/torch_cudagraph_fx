# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cudagraph dispatch-mode integration for custom_op: emit the FX graph node.

The implementation runs with the cudagraph mode disabled so the tracer records
only our custom-op node, not the operations inside its implementation. Those
operations' kernels are annotated as belonging to the custom-op node for the
CUDA-graph completeness check.
"""

# pyre-strict

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import _pop_mode_temporarily
from torch_cudagraph_fx._cudagraph_annotations import mark_kernels
from torch_cudagraph_fx._metadata import (
    FX_NODE_ANNOTATION_KEY,
    FX_NODE_ID_KEY,
    node_fx_id_from_metadata,
    NodeMetadata,
)
from torch_cudagraph_fx._tracer import _CudagraphFxDispatchMode
from torch_cudagraph_fx.custom_ops.schema import (
    build_schema_body,
    flat_arg_names,
    mutated_flat_indices,
    schema_annotations,
)

if TYPE_CHECKING:
    from torch_cudagraph_fx.custom_ops.custom_ops import CustomOp


def tensor_metadata(tensor: torch.Tensor) -> tuple[Any, ...]:
    strides = tuple(tensor.stride()) if tensor.layout is torch.strided else None
    offset = tensor.storage_offset() if tensor.layout is torch.strided else None
    return (
        tuple(tensor.shape),
        strides,
        offset,
        tensor.dtype,
        tensor.device,
        tensor.layout,
    )


def _run_and_build_overload(
    custom_op: CustomOp,
    mode: _CudagraphFxDispatchMode,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any, list[Any], list[Any], NodeMetadata]:
    """Run the op and get/create the overload for this calling convention.

    Runs the real function with the cudagraph dispatch mode disabled, infers the
    leaf schema and aliasing from the observed values, and returns
    ``(out, op, lifted_flat_in, flat_out, node_meta)`` for the caller to wire
    into the graph.
    """
    # Canonicalize positional arguments, keywords, and defaults so equivalent
    # calls share one overload identity.
    bound = custom_op._sig.bind(*args, **kwargs)
    bound.apply_defaults()
    args, kwargs = bound.args, bound.kwargs

    flat_in, in_spec = pytree.tree_flatten((args, kwargs))
    lifted_flat_in = [mode.lift(value) for value in flat_in]
    arg_names = flat_arg_names(custom_op._sig, args, kwargs)
    mutated = mutated_flat_indices(custom_op._sig, custom_op._mutates, args, kwargs)
    metadata_before = [
        tensor_metadata(value) if isinstance(value, torch.Tensor) else None
        for value in flat_in
    ]

    node_meta = mode.metadata.before_call(args, kwargs)
    # Absorb operations inside the implementation into this custom-op boundary.
    # Kernel annotations preserve the boundary in the captured CUDA graph even
    # though the inner operations do not become individual FX nodes.
    with _pop_mode_temporarily() as popped_mode:
        assert popped_mode is mode
        with mark_kernels(
            {
                FX_NODE_ANNOTATION_KEY: True,
                FX_NODE_ID_KEY: node_fx_id_from_metadata(node_meta),
            }
        ):
            out = custom_op._fn(*args, **kwargs)

        for i, value in enumerate(flat_in):
            if (
                isinstance(value, torch.Tensor)
                and tensor_metadata(value) != metadata_before[i]
            ):
                raise RuntimeError("metadata-changing input mutations are unsupported")

        # out_spec is None for a void op (see build_schema_body /
        # _get_or_create_overload).
        flat_out, out_spec = ([], None) if out is None else pytree.tree_flatten(out)
        in_types, out_types = schema_annotations(flat_in, flat_out, mutated)
        schema_body = build_schema_body(arg_names, in_types, out_types)
        op = custom_op._get_or_create_overload(schema_body, in_spec, out_spec)
    return out, op, lifted_flat_in, flat_out, node_meta


def trace_custom_op(
    custom_op: CustomOp,
    mode: _CudagraphFxDispatchMode,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    out, op, lifted_flat_in, flat_out, node_meta = _run_and_build_overload(
        custom_op, mode, args, kwargs
    )
    # Flat leaf args match the leaf-only schema; the pytree structure is metadata
    # on op._in_spec, reconstructed for display and replay. _record_call_function
    # binds flat output leaves and records the call-site stream and stack.
    mode._record_call_function(
        op,
        tuple(lifted_flat_in),
        {},
        flat_out,
        node_meta,
    )
    return out
