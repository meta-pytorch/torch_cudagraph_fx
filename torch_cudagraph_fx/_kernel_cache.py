# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shapes and constants can vary without changing operator structure, so
pattern directories group specializations by structure. SPEC includes
the interface and numerics to avoid reusing incompatible kernels."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import linecache
import math
import operator
import os
import re
import sys
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.fx as fx
from torch._library.custom_ops import OPDEFS
from torch.utils._pytree import tree_map
from torch_cudagraph_fx._interpreter import fused_op, loaded_kernels
from torch_cudagraph_fx._metadata import node_metadata, node_tensor_meta
from torch_cudagraph_fx._schedule import constant_type, is_fusable, PythonOpOverload


def _constant_source(value: Any) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return f'float("{value}")'
    if isinstance(value, torch.device):
        return f'torch.device("{value}")'
    if isinstance(value, (torch.dtype, torch.layout, torch.memory_format)):
        return str(value)
    return repr(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def pattern_key(gm: fx.GraphModule, dependencies: tuple = ()) -> str:
    names: dict[fx.Node, str] = {}
    viewed: dict[fx.Node, fx.Node] = {}
    structure = []
    num_inputs = 0

    def argument(value: Any) -> Any:
        nonlocal num_inputs
        if not isinstance(value, fx.Node):
            return None if value is None else type(value).__name__
        if constant_type(value):
            return constant_type(value)
        value = viewed.get(value, value)
        if value not in names:
            names[value] = f"in{num_inputs}"
            num_inputs += 1
        return names[value]

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        source = node.args[0] if node.args else None
        if (
            node.op == "call_function"
            and isinstance(source, fx.Node)
            and (
                (
                    isinstance(node.target, torch.library.OpOverload)
                    and node.target.is_view
                )
                or (node.target is operator.getitem and source in viewed)
            )
        ):
            viewed[node] = viewed.get(source, source)
            continue
        args = tree_map(argument, (node.args, node.kwargs))
        names[node] = f"v{len(structure)}"
        structure.append((node.op, str(node.target), args))
    return "pat_" + _digest((structure, dependencies))


def _normalize(gm: fx.GraphModule, region: str) -> fx.GraphModule:
    graph = fx.Graph()
    env = {}
    constants = 0
    for i, node in enumerate(gm.graph.nodes):
        if node.op not in ("placeholder", "call_function", "output"):
            raise ValueError(f"cached region {region} must pass all state as arguments")
        copied = graph.node_copy(node, env.__getitem__)
        if constant_type(node):
            copied.name = f"c{constants}"
            constants += 1
        else:
            copied.name = f"in{i}" if node.op == "placeholder" else f"v{i}"
        if node.op == "placeholder":
            copied.target = copied.name
        env[node] = copied
    return fx.GraphModule({}, graph)


def _python_definition(target: torch.library.OpOverload) -> Callable | None:
    if isinstance(target, PythonOpOverload):
        return target._custom_op._fn
    definition = OPDEFS.get(target._name)
    return None if definition is None else definition._init_fn


def _dependencies(gm: fx.GraphModule, region: str) -> tuple[tuple[str, str], ...]:
    modules = {}
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is operator.getitem:
            continue
        target = node.target
        if not is_fusable(node):
            raise ValueError(f"cached region {region} does not support {target}")
        fn = _python_definition(target)
        if fn is None:
            if target.namespace in ("aten", "prims"):
                continue
            raise ValueError(
                f"{target} in region {region} has no Python definition to import; "
                "define it with torch.library.custom_op, triton_op or "
                "torch_cudagraph_fx.custom_op"
            )
        module = inspect.getmodule(fn)
        if module is None or module.__name__ == "__main__":
            raise ValueError(
                f"{target} in region {region} needs an importable defining module"
            )
        modules[module.__name__] = hashlib.sha256(
            inspect.getsource(module).encode()
        ).hexdigest()
    return tuple(sorted(modules.items()))


def _interface(gm: fx.GraphModule, region: str, args: Sequence[Any]) -> dict[str, Any]:
    storages = {}
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    examples = dict(zip(placeholders, args, strict=True))

    def describe(value: Any) -> Any:
        if not isinstance(value, fx.Node):
            if value is None or isinstance(value, (bool, int, float, str)):
                return {"value": value}
            raise ValueError(
                f"cached region {region} has an unsupported value: {type(value).__name__}"
            )
        if constant_type(value):
            example = _constant_source(examples[value])
            return {"constant": constant_type(value), "example": example}
        tensor = node_tensor_meta(value)
        if not tensor:
            raise ValueError(
                f"cached region {region} needs tensor metadata for {value}"
            )
        return {
            "shape": tensor["shape"],
            "stride": tensor["stride"],
            "dtype": tensor["dtype"],
            "device": torch.device(str(tensor["device"])).type,
            "alias_group": storages.setdefault(tensor["storage_id"], len(storages)),
        }

    output = next(n for n in gm.graph.nodes if n.op == "output")
    return {
        "signature": f"def run({', '.join(n.name for n in placeholders)})",
        "inputs": tree_map(describe, placeholders),
        "outputs": tree_map(describe, output.args[0]),
    }


def _specialization_key(spec: dict[str, Any]) -> str:
    # Constants are call arguments, so their example values select no kernel.
    inputs = [{k: v for k, v in i.items() if k != "example"} for i in spec["inputs"]]
    return _digest({**spec, "inputs": inputs})


def _source(value: Any, region: str) -> str:
    if isinstance(value, fx.Node):
        return value.name
    if isinstance(value, list):
        return f"[{', '.join(_source(v, region) for v in value)}]"
    if isinstance(value, tuple):
        items = ", ".join(_source(v, region) for v in value)
        return f"({items},)" if len(value) == 1 else f"({items})"
    if value is None or type(value) in (bool, int, float, str):
        return _constant_source(value)
    raise ValueError(f"cached region {region} has an unsupported argument: {value!r}")


def _reference_source(gm: fx.GraphModule, dependencies: tuple, region: str) -> str:
    """Call each operator overload directly: FX's generated code would call trace
    leaves through their overload packets. Values are released after their last
    use, as in FX's generated code."""
    released: dict[fx.Node, list[str]] = {}
    seen: set[fx.Node] = set()
    for node in reversed(gm.graph.nodes):
        for used in node.all_input_nodes:
            if used not in seen:
                seen.add(used)
                released.setdefault(node, []).append(used.name)
    params = ", ".join(n.name for n in gm.graph.nodes if n.op == "placeholder")
    lines = [f"def reference({params}):"]
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "output":
            lines.append(f"    return {_source(node.args[0], region)}")
            continue
        if node.target is operator.getitem:
            line = f"{node.name} = {node.args[0].name}[{node.args[1]!r}]"
        else:
            target = f"torch.ops.{node.target}"
            if isinstance(node.target, PythonOpOverload):
                target += f".{node.target._custom_op_overload_name}"
            args = [_source(a, region) for a in node.args]
            args += [f"{k}={_source(v, region)}" for k, v in node.kwargs.items()]
            line = f"{node.name} = {target}({', '.join(args)})"
        if node in released:
            line += f";  {' = '.join(released[node])} = None"
        lines.append(f"    {line}")
    imports = ["import torch", *(f"import {module}" for module, _ in dependencies)]
    return "\n".join(imports) + "\n\n" + "\n".join(lines) + "\n"


def _environment(gm: fx.GraphModule) -> dict[str, Any]:
    devices = {
        str(node_tensor_meta(n).get("device"))
        for n in gm.graph.nodes
        if n.op != "output" and node_tensor_meta(n)
    }
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capabilities": sorted(
            {
                torch.cuda.get_device_capability(torch.device(device))
                for device in devices
                if device.startswith("cuda")
            }
        ),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }


def _create_if_missing(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A file opened normally, unlike a tempfile, follows the umask, so other users
    # can read a shared cache.
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(source)
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        # `import a.b` rebinds `a` to the same package, so it counts as `a.b`.
        return {alias.asname or alias.name for alias in node.names}
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del))
    }


def _verify_reference(path: Path, source: str, reference: str) -> None:
    expected = ast.parse(reference).body
    preserved_names = set().union(*(_bound_names(node) for node in expected))
    actual = [
        node for node in ast.parse(source).body if _bound_names(node) & preserved_names
    ]
    if [ast.dump(node) for node in actual] != [ast.dump(node) for node in expected]:
        raise ValueError(f"cached kernel reference was modified: {path}")


def _spec_json(spec: dict[str, Any]) -> str:
    text = json.dumps(spec, indent=2)
    return re.sub(
        r'"(?:\\.|[^"\\])*"|\[[\d\s,.-]*\]',
        lambda m: m.group()
        if m.group().startswith('"')
        else json.dumps(json.loads(m.group())),
        text,
    )


def _lookup(
    gm: fx.GraphModule, root: Path, region: str, args: Sequence[Any]
) -> tuple[Any, dict[str, str]]:
    normalized = _normalize(gm, region)
    dependencies = _dependencies(normalized, region)
    reference = _reference_source(normalized, dependencies, region)
    pattern = pattern_key(normalized, dependencies)
    spec = {
        **_interface(normalized, region, args),
        "format_version": 1,
        "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        "environment": _environment(normalized),
    }
    specialization = _specialization_key(spec)
    path = root / pattern / f"{specialization}.py"
    names = ", ".join(n.name for n in normalized.graph.nodes if n.op == "placeholder")
    source = (
        '"""Edit impl and select run = impl; preserve SPEC and reference."""\n'
        "import json\n"
        f'SPEC = json.loads(r"""\n{_spec_json(spec)}\n""")\n\n'
        f"{reference}\n"
        f"def impl({names}):\n    raise NotImplementedError\n\n"
        "run = reference\n"
    )
    _create_if_missing(path, source)
    loaded_source = path.read_text()
    _verify_reference(path, loaded_source, reference)
    module = _load_source(path, loaded_source)
    if _specialization_key(module.SPEC) != specialization:
        raise ValueError(
            f"cached kernel SPEC does not match its specialization: {path}"
        )
    expected = next(
        node for node in ast.parse(reference).body if isinstance(node, ast.FunctionDef)
    )
    actual = ast.parse(inspect.getsource(module.reference)).body[0]
    if ast.dump(expected) != ast.dump(actual):
        raise ValueError(f"cached kernel reference was modified: {path}")
    if not callable(module.run):
        raise TypeError(f"cached kernel run must be callable: {path}")
    is_reference = module.run is module.reference
    return module.run, {
        "path": str(path),
        "pattern_key": pattern,
        "specialization_key": specialization,
        "kind": "reference" if is_reference else "implementation",
    }


def _load_source(path: Path, source: str) -> ModuleType:
    name = "cudagraph_fx_kernel_" + _digest((str(path), source))
    if name in sys.modules:
        return sys.modules[name]
    filename = f"{path}::{name}"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module
    try:
        # dont_inherit: this module's `from __future__ import annotations` must not
        # turn the kernel's annotations into strings, which code generators that
        # evaluate annotations (such as CuTe DSL's compile) cannot resolve.
        exec(compile(source, filename, "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def substitute_with_kernels(
    gm: fx.GraphModule, cache_dir: str | os.PathLike[str]
) -> fx.GraphModule:
    if "://" in str(cache_dir):
        raise ValueError("kernel_cache_dir must be a local filesystem path")
    root = Path(cache_dir).expanduser().resolve()
    for node in gm.graph.nodes:
        if node.target is not fused_op:
            continue
        name, region = node.args[:2]
        submodule = gm.get_submodule(str(region.target))
        if not isinstance(submodule, fx.GraphModule):
            raise TypeError("kernel substitution requires outlined FX GraphModules")
        run, metadata = _lookup(submodule, root, name, node.args[2:])
        node_metadata(node)["kernel_cache"] = metadata
        loaded_kernels[metadata["path"]] = run
        node.kwargs = {**node.kwargs, "cache_path": metadata["path"]}
    gm.graph.lint()
    gm.recompile()
    return gm
