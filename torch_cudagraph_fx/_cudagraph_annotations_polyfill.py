# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
# ``cuda.bindings`` (NVIDIA cuda-python) ships no type stubs; this file is
# low-level driver glue over it, so it opts out of pyre-strict. The typed,
# torch-only surface lives in ``_cudagraph_annotations.py``.

"""Standalone fallback for ``torch.cuda._graph_annotations`` on torch builds that
predate it.

The capture-side annotation logic here -- snapshot the stream's capture frontier
on ``mark_kernels`` entry, walk the dependent edges the scope adds, and stamp
those nodes' ``toolsId``s -- mirrors the upstream implementation. This copy is
trimmed to exactly what the tracer needs:

- Public surface only: ``mark_kernels``, the annotation registry
  (:func:`get_kernel_annotations` / :func:`clear_kernel_annotations`),
  :func:`capture_with_annotations_enabled`, :func:`walk_graph_nodes`,
  :func:`annotations_available`.
- Capture-side ``toolsId`` only. The throwaway graph is never instantiated, so
  (unlike upstream, which reads an instantiated exec graph via
  ``CUDAGraph.get_graph_data``) there is no exec-graph remap: the toolsIds
  recorded by :func:`mark_kernels` and those walked by :func:`walk_graph_nodes`
  share one key space.
- Dependencies limited to ``torch`` and ``cuda.bindings``.

Requires the ``cudaGraphNodeGetToolsId`` driver API (cuda.bindings >= 13.1 /
cuda-compat); without it :func:`annotations_available` is ``False`` and
:func:`mark_kernels` is a no-op.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

import torch

try:
    from cuda.bindings import (
        driver as _drv,  # pyre-ignore[21]
        runtime as _rt,  # pyre-ignore[21]
    )

    _HAS_CUDA_BINDINGS = True
except ImportError:  # pragma: no cover - exercised only without cuda.bindings
    _drv = None
    _rt = None
    _HAS_CUDA_BINDINGS = False


def _check_rt(result: Any) -> Any:
    """Strip and check the leading cudaError_t from a cuda.bindings runtime call."""
    err, *vals = result
    if err != _rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA runtime error: {err!r}")
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else tuple(vals)


def _check_drv(result: Any) -> Any:
    """Strip and check the leading CUresult from a cuda.bindings driver call."""
    err, *vals = result
    if err != _drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err!r}")
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else tuple(vals)


_tools_id_available: bool | None = None


def _probe_tools_id() -> bool:
    """Probe whether ``cudaGraphNodeGetToolsId`` is usable by the driver.

    A null-node call returning ``cudaErrorCallRequiresNewerDriver`` means the
    driver lacks it (missing cuda-compat); any other error (e.g.
    ``cudaErrorInvalidValue``) means the API is present.
    """
    if not hasattr(_rt, "cudaGraphNodeGetToolsId"):
        return False
    err, *_ = _rt.cudaGraphNodeGetToolsId(0)
    return err != _rt.cudaError_t.cudaErrorCallRequiresNewerDriver


def _is_tools_id_available() -> bool:
    global _tools_id_available
    if not _HAS_CUDA_BINDINGS:
        return False
    if _tools_id_available is None:
        _tools_id_available = _probe_tools_id()
    return _tools_id_available


def annotations_available() -> bool:
    return torch.cuda.is_available() and _is_tools_id_available()


# Whether annotation recording is active. Scoped to a capture by
# :func:`capture_with_annotations_enabled`; when False, ``mark_kernels`` is a no-op.
_annotations_enabled: bool = False

# capture-side toolsId -> list of annotation dicts.
_kernel_annotations: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
# (annotation, toolsIds) discovered per ``mark_kernels`` scope, pending resolve.
_pending_scopes: list[tuple[dict[str, Any], list[int]]] = []

# Driver node-type enums we annotate. Lazily built to avoid touching
# cuda.bindings at import time.
_ANNOTATABLE_TYPES: set[Any] | None = None


def _get_annotatable_types() -> set[Any]:
    global _ANNOTATABLE_TYPES
    if _ANNOTATABLE_TYPES is None:
        t = _drv.CUgraphNodeType
        _ANNOTATABLE_TYPES = {
            t.CU_GRAPH_NODE_TYPE_KERNEL,
            t.CU_GRAPH_NODE_TYPE_MEMCPY,
            t.CU_GRAPH_NODE_TYPE_MEMSET,
            t.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
        }
    return _ANNOTATABLE_TYPES


def _get_capture_state(stream: Any) -> tuple[Any, list[Any]] | None:
    """Return ``(graph, frontier)`` for an active capture on ``stream``, else None."""
    status, _id, graph, deps, _edge, num_deps = _check_rt(
        _rt.cudaStreamGetCaptureInfo(stream)
    )
    if status != _rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return None
    return graph, list(deps[:num_deps])


def _get_root_nodes(graph: Any) -> list[Any]:
    _, num_roots = _check_rt(_rt.cudaGraphGetRootNodes(graph))
    if num_roots == 0:
        return []
    roots, num_roots = _check_rt(
        _rt.cudaGraphGetRootNodes(graph, pNumRootNodes=num_roots)
    )
    return list(roots[:num_roots])


def _get_dependent_nodes(node: Any) -> list[Any]:
    _, _, num_dependents = _check_rt(_rt.cudaGraphNodeGetDependentNodes(node))
    if num_dependents == 0:
        return []
    dependents, _edge, num_dependents = _check_rt(
        _rt.cudaGraphNodeGetDependentNodes(node, pNumDependentNodes=num_dependents)
    )
    return list(dependents[:num_dependents])


def _collect_descendants(
    start_nodes: list[Any],
    *,
    existing_direct_dependents: dict[int, set[int]] | None = None,
    include_start_nodes: bool = False,
) -> dict[int, Any]:
    """Walk dependent edges from ``start_nodes``, skipping pre-existing edges."""
    existing_direct_dependents = existing_direct_dependents or {}
    seen = {int(node) for node in start_nodes}
    descendants: dict[int, Any] = {}
    stack = list(start_nodes)

    if include_start_nodes:
        for node in start_nodes:
            descendants[int(node)] = node

    while stack:
        node = stack.pop()
        old_dependents = existing_direct_dependents.get(int(node), set())
        for dependent in _get_dependent_nodes(node):
            dependent_key = int(dependent)
            if dependent_key in old_dependents or dependent_key in seen:
                continue
            seen.add(dependent_key)
            descendants[dependent_key] = dependent
            stack.append(dependent)

    return descendants


@contextmanager
def mark_kernels(annotation: str | dict[str, Any]) -> Iterator[None]:
    """Annotate every kernel/memcpy/memset node the capture adds within scope.

    A no-op outside an active capture, with annotations disabled, or when
    :func:`annotations_available` is ``False``. On entry, snapshots the current
    stream's capture frontier and its existing direct dependents; on exit, walks
    only the dependent edges added since entry (falling back to newly created
    roots when the scope is the first captured work) and records the annotatable
    nodes' ``toolsId``s. Mirrors ``torch.cuda._graph_annotations.mark_kernels``.
    """
    if not _annotations_enabled or not _is_tools_id_available():
        yield
        return

    if isinstance(annotation, str):
        annotation = {"name": annotation}
    elif not isinstance(annotation, dict):
        raise TypeError("mark_kernels annotation must be a string or dict")

    stream = _rt.cudaStream_t(init_value=torch.cuda.current_stream().cuda_stream)
    capture_state = _get_capture_state(stream)
    if capture_state is None:
        yield
        return
    graph, frontier = capture_state

    entry_root_keys: set[int] | None = None
    entry_direct_dependents = {
        int(node): {int(dep) for dep in _get_dependent_nodes(node)} for node in frontier
    }
    if not frontier:
        entry_root_keys = {int(node) for node in _get_root_nodes(graph)}

    yield

    if frontier:
        scope_nodes = _collect_descendants(
            frontier,
            existing_direct_dependents=entry_direct_dependents,
        )
    else:
        new_roots = [
            node
            for node in _get_root_nodes(graph)
            if int(node) not in (entry_root_keys or set())
        ]
        scope_nodes = _collect_descendants(new_roots, include_start_nodes=True)

    if not scope_nodes:
        return

    annotatable = _get_annotatable_types()
    tools_ids: list[int] = []
    for node in scope_nodes.values():
        if _check_drv(_drv.cuGraphNodeGetType(node)) not in annotatable:
            continue
        tools_ids.append(_check_rt(_rt.cudaGraphNodeGetToolsId(node)))

    if tools_ids:
        _pending_scopes.append((annotation, tools_ids))


def resolve_pending_annotations() -> None:
    """Fold pending scope toolsIds into the annotation registry (capture-side)."""
    if not _pending_scopes:
        return
    per_tools_id: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation, tools_ids in _pending_scopes:
        for tools_id in tools_ids:
            per_tools_id[tools_id].append(annotation)

    for tools_id, ann_list in per_tools_id.items():
        if len(ann_list) == 1:
            _kernel_annotations[tools_id].append(ann_list[0])
            continue
        merged: dict[str, Any] = {}
        for annotation in ann_list:
            for key, value in annotation.items():
                merged.setdefault(key, value)
        _kernel_annotations[tools_id].append(merged)

    _pending_scopes.clear()


def get_kernel_annotations() -> dict[int, list[dict[str, Any]]]:
    """Return the live capture-side annotation registry."""
    return _kernel_annotations


def clear_kernel_annotations() -> None:
    _kernel_annotations.clear()
    _pending_scopes.clear()


@contextmanager
def capture_with_annotations_enabled(
    cuda_graph: torch.cuda.CUDAGraph, **kwargs: Any
) -> Iterator[None]:
    """Capture into ``cuda_graph`` with annotation recording enabled.

    Enables recording, runs the capture, then folds pending scopes into the
    registry on exit — the standalone equivalent of
    ``torch.cuda.graph(..., enable_annotations=True)``.
    """
    global _annotations_enabled
    _annotations_enabled = True
    try:
        with torch.cuda.graph(cuda_graph, **kwargs):
            yield
        resolve_pending_annotations()
    finally:
        _annotations_enabled = False


# Runtime node-type enum -> the string names get_graph_data uses. Lazily built.
_NODE_TYPE_NAMES: dict[Any, str] | None = None


def _node_type_names() -> dict[Any, str]:
    global _NODE_TYPE_NAMES
    if _NODE_TYPE_NAMES is None:
        nt = _rt.cudaGraphNodeType
        _NODE_TYPE_NAMES = {
            nt.cudaGraphNodeTypeKernel: "kernel",
            nt.cudaGraphNodeTypeMemcpy: "memcpy",
            nt.cudaGraphNodeTypeMemset: "memset",
            nt.cudaGraphNodeTypeHost: "host",
            nt.cudaGraphNodeTypeGraph: "child_graph",
            nt.cudaGraphNodeTypeEmpty: "empty",
            nt.cudaGraphNodeTypeWaitEvent: "wait_event",
            nt.cudaGraphNodeTypeEventRecord: "event_record",
            nt.cudaGraphNodeTypeMemAlloc: "mem_alloc",
            nt.cudaGraphNodeTypeMemFree: "mem_free",
        }
    return _NODE_TYPE_NAMES


def walk_graph_nodes(
    cuda_graph: torch.cuda.CUDAGraph,
) -> Iterator[tuple[str, int, str | None]]:
    """Yield ``(node_type, tools_id, kernel_name)`` for each node in the template.

    Capture-side equivalent of the kernel/memcpy fields of
    ``CUDAGraph.get_graph_data``: walks ``raw_cuda_graph()`` (no instantiate, no
    exec-graph remap), so the ``toolsId``s match those recorded by
    :func:`mark_kernels`.
    """
    raw = cuda_graph.raw_cuda_graph()
    _, num = _check_rt(_rt.cudaGraphGetNodes(raw, numNodes=0))
    if num == 0:
        return
    nodes, num = _check_rt(_rt.cudaGraphGetNodes(raw, numNodes=num))
    names = _node_type_names()
    for i in range(num):
        node = nodes[i]
        ntype = _check_rt(_rt.cudaGraphNodeGetType(node))
        tools_id = _check_rt(_rt.cudaGraphNodeGetToolsId(node))
        kernel_name = None
        if ntype == _rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            cu_node = _drv.CUgraphNode(init_value=int(node))
            err, params = _drv.cuGraphKernelNodeGetParams(cu_node)
            if err == _drv.CUresult.CUDA_SUCCESS and int(params.func):
                cu_func = _drv.CUfunction(init_value=int(params.func))
                err, name = _drv.cuFuncGetName(cu_func)
                if err == _drv.CUresult.CUDA_SUCCESS:
                    kernel_name = name.decode() if isinstance(name, bytes) else name
        yield names.get(ntype, str(ntype)), tools_id, kernel_name
