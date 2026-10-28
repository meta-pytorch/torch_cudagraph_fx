# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import inspect
import os
from collections.abc import Callable, Iterable
from types import MethodType
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
)
from torch_cudagraph_fx._tracer import _CudagraphFxDispatchMode
from torch_cudagraph_fx.custom_ops.arg_path import parse_arg_path
from torch_cudagraph_fx.custom_ops.dispatchless_op_overload import (
    DispatchlessOpOverload,
    DispatchlessOpOverloadPacket,
)
from torch_cudagraph_fx.custom_ops.schema import encode_overload_name
from torch_cudagraph_fx.custom_ops.tracing import trace_custom_op

try:
    from torch._custom_class_base import CustomClassBase as _ReferenceBase
    from torch._library.opaque_object import (
        register_custom_class as _register_reference,
    )

    _REFERENCE_KIND = "symbolic"
except ImportError:
    from torch._library.opaque_object import register_opaque_type as _register_reference
    from torch._opaque_base import OpaqueBase as _ReferenceBase

    _REFERENCE_KIND = "reference"


class OpaqueRef(_ReferenceBase):  # noqa: B903
    """Reference wrapper for stateful objects used by custom operators."""

    def __init__(self, value: Any) -> None:
        self.value = value


_register_reference(OpaqueRef, typ=_REFERENCE_KIND)


def register_dataclass_pytree(
    cls: type[Any],
    *,
    opaque_ref_fields: Iterable[str] = (),
) -> None:
    """Register a dataclass pytree, wrapping selected fields by reference."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")

    field_names = tuple(field.name for field in dataclasses.fields(cls))
    opaque_ref_field_set = frozenset(opaque_ref_fields)
    bad_fields = opaque_ref_field_set - set(field_names)
    if bad_fields:
        raise ValueError(
            f"opaque_ref_fields names not in dataclass: {sorted(bad_fields)}"
        )

    def flatten(obj: object) -> tuple[list[Any], list[str]]:
        children = []
        for name in field_names:
            value = getattr(obj, name)
            if name in opaque_ref_field_set:
                value = OpaqueRef(value)
            children.append(value)
        return children, list(field_names)

    def flatten_with_keys(
        obj: object,
    ) -> tuple[list[tuple[pytree.KeyEntry, Any]], list[str]]:
        children, context = flatten(obj)
        return [
            (pytree.GetAttrKey(name), child)
            for name, child in zip(field_names, children, strict=True)
        ], context

    def unflatten(children: list[Any], context: list[str]) -> Any:
        result = cls.__new__(cls)
        for name, child in zip(context, children, strict=True):
            if name in opaque_ref_field_set and isinstance(child, OpaqueRef):
                child = child.value
            object.__setattr__(result, name, child)
        return result

    pytree.register_pytree_node(
        cls,
        flatten,
        unflatten,
        serialized_type_name=f"{cls.__module__}.{cls.__qualname__}",
        flatten_with_keys_fn=flatten_with_keys,
    )


def trace_dataclass(
    cls: type[Any],
    *,
    opaque_ref_fields: Iterable[str] = (),
) -> None:
    """Declare a dataclass as structured trace data.

    Fields in opaque_ref_fields are preserved by reference. Set CGFX_TRACING=1
    before calling; otherwise no registration is performed.
    """
    if os.environ.get("CGFX_TRACING") == "1":
        register_dataclass_pytree(cls, opaque_ref_fields=opaque_ref_fields)


class CustomOp:
    def __init__(
        self,
        qualname: str,
        fn: Callable[..., Any],
        mutates_args: Iterable[str] | str = (),
    ) -> None:
        parts = qualname.split("::")
        if len(parts) != 2 or not all(parts):
            raise ValueError("custom_op qualnames must have the form 'namespace::name'")
        self._ns, self._name = parts
        if "." in self._name:
            raise ValueError("custom_op qualnames cannot include overload names")
        self._fn = fn
        self._sig = inspect.signature(fn)
        if mutates_args == "unknown":
            self._mutates = "unknown"
        else:
            paths = (mutates_args,) if isinstance(mutates_args, str) else mutates_args
            self._mutates = tuple(parse_arg_path(path) for path in paths)
            bad = {path.root for path in self._mutates} - self._sig.parameters.keys()
            if bad:
                raise ValueError(f"mutates_args names not in signature: {sorted(bad)}")
        self._cache: dict[tuple[Any, ...], DispatchlessOpOverload] = {}
        self._packet = DispatchlessOpOverloadPacket(self)
        setattr(getattr(torch.ops, self._ns), self._name, self._packet)

    @property
    def qualname(self) -> str:
        return f"{self._ns}::{self._name}"

    def _get_or_create_overload(
        self,
        # The "(args) -> returns" portion of the schema, e.g.
        # "(Tensor(a0) t) -> (Tensor(a0), Tensor)".
        schema_body: str,
        in_spec: pytree.TreeSpec,
        # `None` out_spec means the op returns None (see tracing.py); we should
        # probably support None as a pytree type but that needs more maneuvering.
        out_spec: pytree.TreeSpec | None,
    ) -> DispatchlessOpOverload:
        key = (schema_body, in_spec, out_spec)
        op = self._cache.get(key)
        if op is not None:
            return op

        overload_name = encode_overload_name(
            {
                "v": 1,
                "schema": schema_body,
                "in_spec": pytree.treespec_dumps(in_spec),
                "out_spec": (
                    None if out_spec is None else pytree.treespec_dumps(out_spec)
                ),
            }
        )
        # Avoid the dispatcher's 64-arg limit via our own kind of OpOverload.
        schema = torch._C.parse_schema(
            f"{self._ns}::{self._name}.{overload_name}{schema_body}"
        )
        op = DispatchlessOpOverload(
            self._packet, overload_name, schema, in_spec, out_spec
        )
        self._cache[key] = op
        return op

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # This is the eager-hot path. Keep it as light as possible.
        mode = _get_current_dispatch_mode()
        if mode is None:
            return self._fn(*args, **kwargs)

        if isinstance(mode, _CudagraphFxDispatchMode):
            tracer_modes = [
                candidate
                for candidate in _get_current_dispatch_mode_stack()
                if isinstance(candidate, _CudagraphFxDispatchMode)
            ]
            if len(tracer_modes) != 1:
                raise RuntimeError(
                    "nested torch_cudagraph_fx dispatch modes are unsupported"
                )
            return trace_custom_op(self, mode, args, kwargs)

        if any(
            isinstance(candidate, _CudagraphFxDispatchMode)
            for candidate in _get_current_dispatch_mode_stack()
        ):
            raise RuntimeError(
                "a foreign TorchDispatchMode is shadowing the active "
                "torch_cudagraph_fx tracer"
            )
        return self._fn(*args, **kwargs)

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> Any:
        return self if instance is None else MethodType(self, instance)


def custom_op(
    qualname: str,
    *,
    mutates_args: Iterable[str] | str = (),
) -> Callable[[Callable[..., Any]], CustomOp]:
    """Decorate a Python function as a custom operator.

    Each input of the custom operator may be:
    - one of the supported leaf types (Tensor, SymInt/int, bool, float,
      str, dtype, device, ProcessGroup, an opaque object registered by
      ``torch._library.register_opaque_type``, or a trace_reference class)
    - nested containers (list/tuple/dict/dataclasses) of the previous item

    Each output of the custom operator may be:
    - a symbolic leaf type (Tensor, SymInt, a registered opaque object, or a
      trace_reference class)
    - nested containers of the previous item

    Please specify which Tensors are being mutated via `mutates_args`
    (these include paths to the tensors, e.g. "state['buf']" or "buffers[0]").
    """

    return lambda fn: CustomOp(qualname, fn, mutates_args)


def trace_leaf(
    qualname: str,
    *,
    mutates_args: Iterable[str] | str = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Record calls as opaque graph nodes rather than tracing their bodies.

    Currently supported by CGFX. Set CGFX_TRACING=1 before decorating; otherwise
    the original function is returned unchanged. Argument and mutation semantics
    follow custom_op.
    """
    if os.environ.get("CGFX_TRACING") != "1":
        return lambda fn: fn

    return custom_op(qualname, mutates_args=mutates_args)
