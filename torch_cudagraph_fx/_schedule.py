# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import operator
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, TypeAlias

import torch
import torch.fx as fx
from torch.utils._pytree import tree_leaves
from torch_cudagraph_fx._indexed_graph import IndexedGraph, Op, OpGroup, Scope
from torch_cudagraph_fx._interpreter import fused_op
from torch_cudagraph_fx._metadata import (
    CUDAGRAPH_FX_METADATA_KEY,
    FX_NODE_ID_KEY,
    node_metadata,
    node_stream,
    node_tensor_meta,
)

logger: logging.Logger = logging.getLogger(__name__)

Selection: TypeAlias = Scope | OpGroup | Op
Storage: TypeAlias = int | str | None

# Random operators read and advance the generator state, which no argument
# carries; it is one more storage, so they keep their order among themselves.
RNG_STATE = "rng_state"
_COMMUNICATION = {"c10d", "_c10d_functional", "_c10d_functional_autograd"}
_FUNCTIONAL_HIGHER_ORDER_OPS = {"flex_attention", "flex_attention_backward"}


@runtime_checkable
class PythonOpOverload(Protocol):
    _custom_op: Any
    _custom_op_overload_name: str


def is_barrier(node: fx.Node) -> bool:
    """An operator whose effect is not visible in its inputs, outputs and schema.

    Events order all earlier or later work on their stream. Eager collectives
    write their tensors asynchronously, without schema write annotations, and
    complete at a ``Work.wait`` that is not in the graph. Higher-order operators
    have no schema, so only known functional ones are ordered by their edges.
    """
    target = node.target
    if node.op == "call_module":
        return True
    if node.op != "call_function" or target in (operator.getitem, fused_op):
        return False
    if isinstance(target, torch._ops.HigherOrderOperator):
        return target.name() not in _FUNCTIONAL_HIGHER_ORDER_OPS
    if not isinstance(target, torch.library.OpOverload):
        return True
    return target.namespace == "c10d" or target in {
        torch.ops.torch_cudagraph_fx.new_event.default,
        torch.ops.torch_cudagraph_fx.record_event.default,
        torch.ops.torch_cudagraph_fx.wait_event.default,
    }


def is_fusable(node: fx.Node) -> bool:
    """Any operator with a schema, except collectives and barriers."""
    target = node.target
    return (
        isinstance(target, torch.library.OpOverload)
        and target.namespace not in _COMMUNICATION
        and not is_barrier(node)
    )


def _is_view(node: fx.Node) -> bool:
    """An operator whose results alias its inputs, which it does not write."""
    target = node.target
    return isinstance(target, torch.library.OpOverload) and target.is_view


def _views_inside(nodes: set[fx.Node]) -> list[fx.Node]:
    """The views between the first and last of ``nodes`` that read their results,
    and the views after the first that feed them."""
    region = set(nodes)
    first, last = min(region), max(region)
    views: list[fx.Node] = []
    for node in _span(first, last):
        if node in region or not (_is_view(node) or node.target is operator.getitem):
            continue
        if any(i in region for i in node.all_input_nodes):
            region.add(node)
            if node.target is not operator.getitem:
                views.append(node)
    pending = list(region)
    while pending:
        for i in pending.pop().all_input_nodes:
            if i not in region and i > first and _is_view(i):
                region.add(i)
                views.append(i)
                pending.append(i)
    return views


@dataclass(frozen=True)
class OutlineRegion:
    name: str
    paths: tuple[str, ...]

    def apply(self, indexed_graph: IndexedGraph) -> None:  # noqa: C901
        selected = {
            op.node for path in self.paths for op in indexed_graph.resolve(path)
        }
        if not selected:
            raise ValueError(f"region {self.name} is empty")
        if unfusable := [n for n in selected if not is_fusable(n)]:
            raise ValueError(f"region {self.name} cannot fuse {unfusable[:3]}")
        if len({n.graph for n in selected}) != 1:
            raise ValueError(f"region {self.name} selects nodes from different graphs")
        pending = list(selected)
        while pending:
            node = pending.pop()
            for user in node.users:
                if user.target is operator.getitem and user not in selected:
                    selected.add(user)
                    pending.append(user)
        nodes = [n for n in _span(min(selected), max(selected)) if n in selected]
        gm = nodes[0].graph.owning_module
        if self.name in indexed_graph.fuse or hasattr(gm, self.name):
            raise ValueError(f"region {self.name!r} is already a graph attribute")
        moved = _check_region(nodes, self.name, indexed_graph)
        anchor = nodes[-1]
        for node in moved:
            anchor.append(node)
            anchor = node
        if moved:
            logger.info(
                "region %s: moved %d dependent nodes after its call: %s",
                self.name,
                len(moved),
                ", ".join(n.name for n in moved),
            )
        members = set(nodes)
        inputs, outputs = _boundary(nodes)
        users = {u for n in outputs for u in n.users if u not in members}
        constants, copied = _outline(gm, nodes, self.name)
        with gm.graph.inserting_after(nodes[-1]):
            region = gm.graph.get_attr(self.name)
        with gm.graph.inserting_after(region):
            call = gm.graph.call_function(
                fused_op, (self.name, region, *inputs, *constants)
            )
        region.meta[CUDAGRAPH_FX_METADATA_KEY] = {}
        call.meta[CUDAGRAPH_FX_METADATA_KEY] = {
            "execution": {"stream": node_stream(nodes[0])},
            "fusion_region": self.name,
            "storage_reads": list(set().union(*(_reads(n) for n in nodes))),
            "storage_writes": list(set().union(*(_writes(n) for n in nodes))),
        }
        replacements = {}
        previous = call
        for i, node in enumerate(outputs):
            with gm.graph.inserting_after(previous):
                previous = gm.graph.call_function(operator.getitem, (call, i))
            previous.meta = deepcopy(node.meta)
            node_metadata(previous).pop(FX_NODE_ID_KEY, None)
            replacements[node] = previous
        for user in users:
            user.args, user.kwargs = fx.node.map_arg(
                (user.args, user.kwargs), lambda n: replacements.get(n, n)
            )
        for node in reversed(nodes):
            gm.graph.erase_node(node)

        indexed_graph.replace_region(nodes, call, copied)


class SchedulePlan:
    def __init__(self, indexed_graph: IndexedGraph) -> None:
        self.indexed_graph = indexed_graph
        self._regions: list[OutlineRegion] = []
        self._next_region = len(indexed_graph.fuse)

    def fuse(self, name: str, targets: Sequence[Selection]) -> None:
        """Fuse the selected operators into one region. The views between them
        that read their results or feed them join the region, so a request names
        only the operators that compute."""
        if isinstance(targets, (str, Scope, OpGroup, Op)):
            raise TypeError("targets must be a sequence of selections")
        if not targets:
            raise ValueError(f"region {name} selects nothing")
        ops = []
        for target in targets:
            if not isinstance(target, (Scope, OpGroup, Op)):
                raise TypeError("fusion targets must be Op, OpGroup or Scope handles")
            ops.extend(self.indexed_graph.resolve(target))
        nodes = {op.node for op in ops}
        if len({n.graph for n in nodes}) == 1:
            views = map(self.indexed_graph.op_of, _views_inside(nodes))
            ops.extend(op for op in views if op is not None)
        paths = tuple(op.path for op in ops)
        self._regions.append(OutlineRegion(f"{name}_{self._next_region}", paths))
        self._next_region += 1

    def apply(self) -> IndexedGraph:
        for module in self.indexed_graph.graph_module.modules():
            if isinstance(module, fx.GraphModule):
                module.graph.lint()
        indexed_graph = self.indexed_graph.copy()
        for region in self._regions:
            region.apply(indexed_graph)
        for module in indexed_graph.graph_module.modules():
            if isinstance(module, fx.GraphModule):
                module.graph.lint()
                module.recompile()
        return indexed_graph


def _span(first: fx.Node, last: fx.Node) -> Iterator[fx.Node]:
    node = first
    for _ in range(len(first.graph.nodes)):
        yield node
        if node is last:
            return
        node = node.next
    raise ValueError("region end is not reachable in graph order")


def _boundary(nodes: list[fx.Node]) -> tuple[list[fx.Node], list[fx.Node]]:
    members = set(nodes)
    inputs = list(
        dict.fromkeys(i for n in nodes for i in n.all_input_nodes if i not in members)
    )
    outputs = [n for n in nodes if any(u not in members for u in n.users)]
    return inputs, outputs


def _is_random(node: fx.Node) -> bool:
    target = node.target
    return (
        isinstance(target, torch.library.OpOverload)
        and torch.Tag.nondeterministic_seeded in target.tags
    )


def _writes(node: fx.Node) -> set[Storage]:
    """Torch exposes argument alias information only through OpOverload._schema."""
    if node.target is fused_op:
        return set(node_metadata(node)["storage_writes"])
    if not isinstance(node.target, torch.library.OpOverload):
        return set()
    writes: set[Storage] = {RNG_STATE} if _is_random(node) else set()
    for i, arg in enumerate(node.target._schema.arguments):
        if arg.alias_info is None or not arg.alias_info.is_write:
            continue
        value = node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
        for leaf in tree_leaves(value):
            if leaf is None:
                continue
            storage = (
                node_tensor_meta(leaf).get("storage_id")
                if isinstance(leaf, fx.Node)
                else None
            )
            writes.add(storage if isinstance(storage, int) else None)
    return writes


def _storages(nodes: Sequence[fx.Node]) -> set[int]:
    return {
        sid
        for n in nodes
        if isinstance(sid := node_tensor_meta(n).get("storage_id"), int)
    }


def _reads(node: fx.Node) -> set[Storage]:
    if node.target is fused_op:
        return set(node_metadata(node)["storage_reads"])
    reads: set[Storage] = set(_storages(node.all_input_nodes))
    if _is_random(node):
        reads.add(RNG_STATE)
    return reads


def _path(node: fx.Node, indexed_graph: IndexedGraph) -> str:
    while node.target is operator.getitem:
        node = node.args[0]
    if node.target is fused_op:
        return f"fuse.{node.args[0]}"
    op = indexed_graph.op_of(node)
    return node.name if op is None else op.path


def _depends(node: fx.Node, other: fx.Node) -> bool:
    """Whether ``node`` must run after ``other``: it reads a value of ``other``, or
    the two access one storage and one of them writes it."""
    writes = _writes(other)
    return (
        other in node.all_input_nodes
        or bool(_reads(node) & writes)
        or bool(_writes(node) & (_reads(other) | writes | _storages([other])))
    )


def _missing_error(
    name: str,
    member: fx.Node,
    nodes: list[fx.Node],
    moved: list[fx.Node],
    indexed_graph: IndexedGraph,
) -> ValueError:
    """The region operator ``member`` depends on moved nodes, which depend on the
    region; name them and the path from a region operator through them."""
    path = [next(m for m in moved if _depends(member, m)), member]
    while path[0] not in nodes:
        earlier = [n for n in (*nodes, *moved) if n < path[0]]
        previous = next((n for n in earlier if _depends(path[0], n)), None)
        if previous is None:
            break
        path.insert(0, previous)
    missing = [n for n in path if n not in nodes]
    names = list(dict.fromkeys(_path(n, indexed_graph) for n in missing))
    chain = " -> ".join(dict.fromkeys(_path(n, indexed_graph) for n in path))
    verb, it = ("is", "it") if len(names) == 1 else ("are", "them")
    problem = f"region {name}: {', '.join(names)} {verb} missing from the region"
    if all(is_fusable(n) for n in missing if n.target is not operator.getitem):
        return ValueError(f"{problem}; add {it} to form a valid region ({chain})")
    return ValueError(f"{problem} but cannot be fused; split the region ({chain})")


def _check_region(
    nodes: list[fx.Node], name: str, indexed_graph: IndexedGraph
) -> list[fx.Node]:
    """Return the nodes between the region's operators that depend on it, in
    graph order; they move after its call, which runs at its last operator.

    A node depends on the region through a data edge, or through a storage the
    region accessed before it, with one of the two accesses writing. Every other
    node keeps its place before the call. A region operator that depends on a
    moved node would have to run both before and after it.
    """
    members = set(nodes)
    _, outputs = _boundary(nodes)
    writes = set().union(*(_writes(n) for n in nodes))
    if not outputs and not writes:
        raise ValueError(f"region {name} has no outputs or writes")
    streams = {node_stream(n) for n in nodes if n.target is not operator.getitem}
    if len(streams) != 1:
        raise ValueError(f"region {name} spans multiple CUDA streams")
    if None in writes:
        raise ValueError(f"region {name} writes storage of unknown identity")
    read: set[Storage] = set()
    written: set[Storage] = set()
    moved: dict[fx.Node, None] = {}
    moved_read: set[Storage] = set()
    moved_written: set[Storage] = set()
    for node in _span(nodes[0], nodes[-1]):
        if node not in members and is_barrier(node):
            raise ValueError(f"region {name} crosses the barrier {node}")
        node_reads, node_writes = _reads(node), _writes(node)
        if node in members:
            if (
                any(i in moved for i in node.all_input_nodes)
                or node_reads & moved_written
                or node_writes & (moved_read | moved_written)
            ):
                raise _missing_error(name, node, nodes, list(moved), indexed_graph)
            read |= node_reads | _storages([node])
            written |= node_writes
            continue
        if None in node_writes:
            raise ValueError(
                f"region {name}: {node} writes storage of unknown identity"
            )
        if (
            any(i in members or i in moved for i in node.all_input_nodes)
            or node_reads & written
            or node_writes & (read | written)
        ):
            moved[node] = None
            moved_read |= node_reads
            moved_written |= node_writes
            read |= node_reads
            written |= node_writes
    return list(moved)


_CONSTANT_TYPES = (
    bool,
    int,
    float,
    str,
    torch.dtype,
    torch.device,
    torch.layout,
    torch.memory_format,
)


def constant_type(node: fx.Node) -> str | None:
    if node.op != "placeholder":
        return None
    constant = node.meta.get(CUDAGRAPH_FX_METADATA_KEY, {}).get("constant")
    return constant if isinstance(constant, str) else None


def _outline(
    gm: fx.GraphModule, nodes: list[fx.Node], name: str
) -> tuple[list[Any], list[fx.Node]]:
    """Lift literal arguments into inputs so regions differing only in constants
    share generated code and each call supplies its own values."""
    inputs, outputs = _boundary(nodes)
    sub = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    for i, node in enumerate(inputs):
        env[node] = sub.placeholder(f"arg_{i}")
        env[node].meta = deepcopy(node.meta)
    constants: list[Any] = []
    first_op: fx.Node | None = None

    def lift(value: Any) -> Any:
        # Exact types only: subclasses such as enums keep their identity.
        if type(value) not in _CONSTANT_TYPES:
            return value
        with sub.inserting_before(first_op):
            placeholder = sub.placeholder(
                f"const_{len(constants)}", type_expr=type(value)
            )
        placeholder.meta[CUDAGRAPH_FX_METADATA_KEY] = {"constant": type(value).__name__}
        constants.append(value)
        return placeholder

    for i, node in enumerate(nodes):
        copied = sub.node_copy(node, env.__getitem__)
        copied.name = f"op_{i}"
        if first_op is None:
            first_op = copied
        if node.target is not operator.getitem:
            copied.args, copied.kwargs = fx.node.map_aggregate(
                (copied.args, copied.kwargs), lift
            )
        env[node] = copied
    sub.output(tuple(env[node] for node in outputs))
    gm.add_submodule(name, fx.GraphModule(gm, sub))
    return constants, [env[node] for node in nodes]
