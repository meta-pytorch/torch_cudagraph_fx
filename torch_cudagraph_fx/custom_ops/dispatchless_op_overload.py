# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The dispatch-less graph-object layer for custom_op.

``DispatchlessOpOverload`` and its packet are ``torch._ops`` subclasses that are
constructed directly in Python and never enter the C++ dispatcher, so they exist
only to carry a ``FunctionSchema`` and serve as FX graph markers.

Convention: these subclass ``torch._ops`` types. Mark overriding attribute
assignments with a ``# @override`` comment, and keep these comments when editing.
"""

# pyre-strict

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Any, TYPE_CHECKING

import torch
import torch.fx.graph
import torch.utils._pytree as pytree
from torch_cudagraph_fx.custom_ops.schema import (
    _OVERLOAD_PREFIX,
    decode_overload_name,
)

if TYPE_CHECKING:
    from torch_cudagraph_fx.custom_ops.custom_ops import CustomOp


class DispatchlessOpOverload(torch._ops.OpOverload):
    """A dispatch-less ``OpOverload`` used purely as an FX graph marker.

    Constructed directly in Python (no ``Library.define``), so it sidesteps the
    dispatcher's 64-argument limit and carries a ``FunctionSchema`` only for
    alias/mutation metadata.
    """

    def __init__(
        self,
        packet: DispatchlessOpOverloadPacket,
        overload_name: str,
        schema: torch._C.FunctionSchema,
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec | None,
    ) -> None:
        # The overload is a graph-node marker. If called directly during replay,
        # reconstruct the original calling convention and run the Python function.
        def dispatch_flat(*flat_args: Any, **kwargs: Any) -> Any:
            if kwargs:
                raise RuntimeError("dispatchless custom-op overload expects flat args")
            reconstructed = pytree.tree_unflatten(list(flat_args), in_spec)
            if not (
                isinstance(reconstructed, tuple)
                and len(reconstructed) == 2
                and isinstance(reconstructed[0], tuple)
                and isinstance(reconstructed[1], dict)
            ):
                raise RuntimeError("invalid custom-op input pytree specification")
            original_args, original_kwargs = reconstructed
            out = packet._custom_op._fn(*original_args, **original_kwargs)
            if out_spec is None:
                if out is not None:
                    raise RuntimeError(
                        "custom-op output pytree changed between tracing and replay"
                    )
                return []
            flat_out, replay_out_spec = pytree.tree_flatten(out)
            if replay_out_spec != out_spec:
                raise RuntimeError(
                    "custom-op output pytree changed between tracing and replay"
                )
            return flat_out

        super().__init__(packet, dispatch_flat, dispatch_flat, schema, tags=[])
        self._custom_op = packet._custom_op
        self._custom_op_overload_name = overload_name
        self._in_spec = in_spec  # node args are flat; this restructures printouts
        self._out_spec = out_spec
        self.__name__ = self._custom_op._name  # @override
        self.__module__ = f"torch._ops.{self._custom_op._ns}"  # @override

    def name(self) -> str:
        return f"{self._namespace}::{self._opname}"

    def __repr__(self) -> str:
        return f"<DispatchlessOpOverload(op='{self._namespace}.{self._opname}')>"

    def __str__(self) -> str:
        return f"{self._namespace}.{self._opname}"

    def _pretty_print(self) -> str:
        return f"torch.ops.{self._namespace}.{self._opname}[*]"


class DispatchlessOpOverloadPacket(torch._ops.OpOverloadPacket):
    def __init__(self, custom_op: CustomOp) -> None:
        # We skip super().__init__() (its signature builds from a registered op),
        # so we set the OpOverloadPacket fields it would ourselves.
        self._custom_op = custom_op
        self._qualified_op_name = f"{custom_op._ns}::{custom_op._name}"  # @override
        self.__name__ = custom_op._name  # @override
        self._op = None  # @override: no registered dispatcher op
        self._has_torchbind_op_overload = False  # @override
        self.__module__ = f"torch.ops.{custom_op._ns}"  # @override

    @property
    def _schemas(self) -> dict[str, torch._C.FunctionSchema]:
        return {
            op._custom_op_overload_name: op._schema
            for op in self._custom_op._cache.values()
        }

    @property
    def _overload_names(self) -> list[str]:
        # On demand: overloads are generated per calling convention (conceptually
        # unbounded), so this is the list materialized (in use) so far.
        return [op._custom_op_overload_name for op in self._custom_op._cache.values()]

    def __getattr__(self, key: str) -> Any:
        # An overload name is a reversible encoding of the schema + pytree specs, so
        # packet.<name> reconstructs the overload even for a calling convention that
        # has not been traced yet. _get_or_create_overload deduplicates existing ones.
        if not key.startswith(_OVERLOAD_PREFIX):
            raise AttributeError(key)
        try:
            metadata = decode_overload_name(key)
            in_spec = pytree.treespec_loads(metadata["in_spec"])
            out_spec = (
                None
                if metadata["out_spec"] is None
                else pytree.treespec_loads(metadata["out_spec"])
            )
        except Exception as error:
            raise AttributeError(key) from error
        return self._custom_op._get_or_create_overload(
            metadata["schema"], in_spec, out_spec
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._custom_op(*args, **kwargs)

    def __hash__(self) -> int:
        return hash(self._qualified_op_name)

    def __iter__(self):
        # Base iterates self._dir (accessed-so-far overloads); we don't track that,
        # so iterate the materialized overloads instead.
        return iter(self._overload_names)

    def overloads(self) -> list[str]:
        return self._overload_names


def _register_display_types(value: Any) -> None:
    # A pytree dataclass reprs as its constructor (Box(t=...)) with a bare,
    # unimported name; register the type as an FX custom builtin so that spelling
    # resolves in generated-code globals, making structured gm.code replayable.
    if isinstance(value, (tuple, list)):
        for item in value:
            _register_display_types(item)
    elif isinstance(value, dict):
        for item in value.values():
            _register_display_types(item)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        existing = torch.fx.graph._custom_builtins.get(cls.__name__)
        if existing is not None and existing.obj is not cls:
            raise RuntimeError(
                f"FX custom builtin name {cls.__name__!r} is already registered "
                "for a different class"
            )
        if existing is None:
            torch.fx.graph._register_custom_builtin(
                cls.__name__, f"from {cls.__module__} import {cls.__name__}", cls
            )
        for field in dataclasses.fields(value):
            _register_display_types(getattr(value, field.name))


@contextlib.contextmanager
def _structured_display(graph: torch.fx.Graph) -> Iterator[None]:
    # TODO: Show structured custom-op outputs in printed graphs.
    # Node args are flat (for BC); temporarily restore the structured pytree form
    # (from in_spec) for printouts. No-op for flat calls, so compile graphs are
    # untouched.
    saved: list[tuple[torch.fx.Node, Any, Any]] = []
    for node in graph.nodes:
        if not isinstance(node.target, DispatchlessOpOverload):
            continue
        new_args, new_kwargs = pytree.tree_unflatten(
            list(node.args), node.target._in_spec
        )
        if new_args != node.args or new_kwargs != node.kwargs:
            _register_display_types(new_args)
            _register_display_types(new_kwargs)
            saved.append((node, node.args, node.kwargs))
            node.args, node.kwargs = new_args, new_kwargs
    try:
        yield
    finally:
        for node, args, kwargs in saved:
            node.args, node.kwargs = args, kwargs


def _install_global_hooks() -> None:
    # str(gm.graph) renders a call target via the Node._pretty_print_target
    # staticmethod, which doesn't dispatch to the target -- delegate to ours.
    orig_pretty_print_target = torch.fx.Node._pretty_print_target

    def pretty_print_target(target: Any) -> str:
        if isinstance(target, DispatchlessOpOverload):
            return target._pretty_print()
        return orig_pretty_print_target(target)

    torch.fx.Node._pretty_print_target = staticmethod(pretty_print_target)

    # Printouts restore the structured (pytree) arg form; node args stay flat.
    orig_python_code = torch.fx.Graph.python_code
    orig_graph_str = torch.fx.Graph.__str__

    def python_code(self: torch.fx.Graph, *args: Any, **kwargs: Any) -> Any:
        with _structured_display(self):
            return orig_python_code(self, *args, **kwargs)

    def graph_str(self: torch.fx.Graph) -> str:
        with _structured_display(self):
            return orig_graph_str(self)

    torch.fx.Graph.python_code = python_code
    torch.fx.Graph.__str__ = graph_str


_install_global_hooks()
