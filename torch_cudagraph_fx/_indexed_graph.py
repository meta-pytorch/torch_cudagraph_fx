# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import difflib
import operator
import re
from collections.abc import Iterable, Iterator
from copy import deepcopy
from typing import Any, overload

import torch
import torch.fx as fx
from torch_cudagraph_fx._interpreter import fused_op
from torch_cudagraph_fx._metadata import (
    CUDAGRAPH_FX_METADATA_KEY,
    node_autograd_meta,
    node_metadata,
)

PHASES = ("fwd", "bwd")


def op_name(node: fx.Node) -> str:
    target = node.target
    if isinstance(target, torch.library.OpOverload):
        return target.overloadpacket.__name__
    return getattr(target, "__name__", str(target))


def _segment(key: str) -> str:
    if key.isdigit():
        return f"[{key}]"
    if key.isidentifier():
        return f".{key}"
    return f'["{key}"]'


class Op:
    __slots__ = ("node", "group", "position")

    def __init__(self, node: fx.Node, group: OpGroup, position: int) -> None:
        self.node = node
        self.group = group
        self.position = position

    @property
    def path(self) -> str:
        if self.group._next_position == 1:
            return self.group.path
        return f"{self.group.path}[{self.position}]"

    @property
    def scope(self) -> Scope:
        return self.group.scope

    def __repr__(self) -> str:
        return f"Op({self.path})"


class OpGroup:
    def __init__(self, scope: Scope, name: str) -> None:
        self.scope = scope
        self.name = name
        self._ops: dict[int, Op] = {}
        self._next_position = 0

    @property
    def path(self) -> str:
        return f"{self.scope._path}{_segment(self.name)}"

    @overload
    def __getitem__(self, i: int) -> Op: ...
    @overload
    def __getitem__(self, i: slice) -> list[Op]: ...
    def __getitem__(self, i: int | slice) -> Op | list[Op]:
        if isinstance(i, slice):
            return list(self)[i]
        return self._ops[i]

    def __iter__(self) -> Iterator[Op]:
        return iter(self._ops.values())

    def __len__(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:
        return f"OpGroup({self.path}, {len(self)} ops)"


class Scope:
    """A module's submodules and operator groups, as attributes or items.
    Iterating yields the submodule keys."""

    def __init__(self, namespace: str, key: str, parent: Scope | None) -> None:
        self._namespace = namespace
        self._key = key
        self._parent = parent
        self._path = (
            f"{parent._path}{_segment(key)}" if parent is not None else namespace
        )
        self._modules: dict[str, Scope] = {}
        self._groups: dict[str, OpGroup] = {}
        self._ops: dict[fx.Node, Op] = {}
        self._all_ops: dict[fx.Node, Op] = {}

    def __getitem__(self, key: int | str) -> Scope | OpGroup:
        key = str(key)
        if key in self._modules and key in self._groups:
            raise KeyError(f"{self._path}: {key!r} names a submodule and an operator")
        if key in self._groups:
            return self._groups[key]
        if key in self._modules:
            return self._modules[key]
        closest = difflib.get_close_matches(key, [*self._modules, *self._groups], n=3)
        raise KeyError(
            f"{self._path} has no submodule or operator {key!r}; closest: {closest}"
        )

    def __getattr__(self, name: str) -> Scope | OpGroup:
        # Copying and unpickling look attributes up before __init__ has run.
        if "_groups" not in vars(self):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __contains__(self, key: object) -> bool:
        return str(key) in self._modules or str(key) in self._groups

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)

    def __repr__(self) -> str:
        return (
            f"Scope({self._path}, {len(self._modules)} submodules, "
            f"{len(self._ops)} own ops)"
        )

    def _child(self, key: str) -> Scope:
        if key not in self._modules:
            self._modules[key] = Scope(self._namespace, key, self)
        return self._modules[key]


class IndexedGraph:
    def __init__(self, graph_module: fx.GraphModule) -> None:
        self.graph_module = graph_module
        self._roots = {name: Scope(name, name, None) for name in (*PHASES, "fuse")}
        self._by_node: dict[fx.Node, Op] = {}
        self._by_path: dict[str, Op] = {}
        self._index_module(graph_module)
        self._by_path = {op.path: op for op in self}

    def _index_module(self, gm: fx.GraphModule, scope: Scope | None = None) -> None:
        for node in gm.graph.nodes:
            if node.op == "output":
                continue
            if node.target is fused_op:
                region = self._region(node)
                self._index_module(gm.get_submodule(str(node.args[1].target)), region)
                continue
            module = node_metadata(node).get("module_fqn")
            if not isinstance(module, str):
                module = ""
            if node.op != "call_function" or node.target is operator.getitem:
                continue
            owner = scope
            if owner is None:
                phase = (
                    "bwd"
                    if node_autograd_meta(node).get("phase") == "backward"
                    else "fwd"
                )
                owner = self._roots[phase]
                for key in module.split(".") if module else ():
                    owner = owner._child(key)
            self._add(node, owner)

    def _add(self, node: fx.Node, scope: Scope) -> None:
        name = op_name(node)
        group = scope._groups.get(name)
        if group is None:
            group = scope._groups[name] = OpGroup(scope, name)
        op = Op(node, group, group._next_position)
        group._ops[op.position] = op
        group._next_position += 1
        scope._ops[node] = op
        while scope is not None:
            scope._all_ops[node] = op
            scope = scope._parent
        self._by_node[node] = op

    def _region(self, call: fx.Node) -> Scope:
        namespace = node_metadata(call).get("index_namespace", "fuse")
        if namespace not in self._roots:
            self._roots[namespace] = Scope(namespace, namespace, None)
        root = self._roots[namespace]
        name = str(call.args[0])
        if name in root:
            raise ValueError(f"index region {namespace}.{name} already exists")
        return root._child(name)

    def replace_region(
        self, removed: Iterable[fx.Node], call: fx.Node, copied: Iterable[fx.Node]
    ) -> None:
        """Update only the replaced operators and their ancestor scopes.

        Surviving handles and occurrence keys remain stable; removed handles
        become invalid. New operators appear in transformation order.
        """
        scope = self._region(call)
        for node in removed:
            op = self._by_node.pop(node, None)
            if op is None:
                continue
            del self._by_path[op.path]
            del op.group._ops[op.position]
            if not op.group:
                del op.scope._groups[op.group.name]
            del op.scope._ops[node]
            ancestor = op.scope
            while ancestor is not None:
                del ancestor._all_ops[node]
                ancestor = ancestor._parent
        for node in copied:
            if node.op == "call_function" and node.target is not operator.getitem:
                self._add(node, scope)
        self._by_path.update((op.path, op) for op in scope._all_ops.values())

    def copy(self) -> IndexedGraph:
        """Copy graphs and index state while sharing parameters and buffers."""
        memo: dict[int, Any] = {}

        def clone(source: fx.GraphModule) -> fx.GraphModule:
            if id(source) in memo:
                return memo[id(source)]
            graph = deepcopy(source.graph)
            attrs = {}
            for old, new in zip(source.graph.nodes, graph.nodes):
                memo[id(old)] = new
                if CUDAGRAPH_FX_METADATA_KEY in old.meta:
                    new.meta[CUDAGRAPH_FX_METADATA_KEY] = deepcopy(node_metadata(old))
                if old.op in ("get_attr", "call_module"):
                    value = operator.attrgetter(str(old.target))(source)
                    attrs[str(old.target)] = (
                        clone(value) if isinstance(value, fx.GraphModule) else value
                    )
            result = fx.GraphModule(attrs, graph)
            memo[id(source)] = result
            return result

        clone(self.graph_module)
        return deepcopy(self, memo)

    @property
    def fwd(self) -> Scope:
        return self._roots["fwd"]

    @property
    def bwd(self) -> Scope:
        return self._roots["bwd"]

    @property
    def fuse(self) -> Scope:
        return self._roots["fuse"]

    def ops(self) -> list[Op]:
        return list(self)

    def op_of(self, node: fx.Node) -> Op | None:
        return self._by_node.get(node)

    def __iter__(self) -> Iterator[Op]:
        return iter(self._by_node.values())

    def __len__(self) -> int:
        return len(self._by_node)

    def __getitem__(self, path: str) -> Scope | OpGroup | Op:
        if path in self._by_path:
            return self._by_path[path]
        tokens = re.findall(r'[^.\[\]"]+|\[\d+\]|\["[^"]*"\]', path)
        if not tokens or tokens[0] not in self._roots:
            raise KeyError(f"paths start with one of {tuple(self._roots)}: {path!r}")
        node: Scope | OpGroup | Op = self._roots[tokens[0]]
        for token in tokens[1:]:
            key = token.strip('[]"')
            if isinstance(node, OpGroup) and token.startswith("[") and key.isdigit():
                node = node[int(key)]
            elif isinstance(node, Scope):
                node = node[key]
            else:
                raise KeyError(f"cannot index {node!r} with {token!r}")
        return node

    def select(self, pattern: str) -> list[Op]:
        regex_parts = []
        in_subscript = False
        for char in pattern:
            if char == "[":
                in_subscript = True
            elif char == "]":
                in_subscript = False
            if char in ("*", "?"):
                wildcard = r"[^\[\]]" if in_subscript else "."
                regex_parts.append(wildcard + ("*" if char == "*" else ""))
            else:
                regex_parts.append(re.escape(char))
        regex = re.compile("".join(regex_parts))
        return [op for path, op in self._by_path.items() if regex.fullmatch(path)]

    def resolve(self, selection: object) -> list[Op]:
        if isinstance(selection, str):
            if any(c in selection for c in "*?"):
                return self.select(selection)
            return self.resolve(self[selection])
        if isinstance(selection, Op):
            ops = [selection]
        elif isinstance(selection, OpGroup):
            ops = list(selection)
        elif isinstance(selection, Scope):
            ops = list(selection._all_ops.values())
        else:
            raise TypeError(
                f"select operators with a Scope, OpGroup, Op or path, not {selection!r}"
            )
        if any(self._by_node.get(op.node) is not op for op in ops):
            raise ValueError(
                "selection contains removed operators or belongs to another index"
            )
        return ops
