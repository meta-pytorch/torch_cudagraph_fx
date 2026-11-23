# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import expecttest
import torch
import torch.fx as fx
import torch_cudagraph_fx._kernel_cache as kernel_cache
from torch.utils._pytree import tree_structure
from torch_cudagraph_fx import (
    custom_op,
    fused_op,
    make_cudagraph_fx,
    SchedulePlan,
    substitute_with_kernels,
)
from torch_cudagraph_fx._interpreter import CudaGraphFxInterpreter
from torch_cudagraph_fx._metadata import CUDAGRAPH_FX_METADATA_KEY, node_metadata
from utils import requires_gpus


@torch.library.custom_op("cgfx_kernel_cache_test::library_neg", mutates_args=())
def _library_neg(x: torch.Tensor) -> torch.Tensor:
    return -x


# An operator defined without custom_op, whose definition the cache cannot find.
_LIBRARY = torch.library.Library("cgfx_kernel_cache_test", "FRAGMENT")
_LIBRARY.define("defined_neg(Tensor x) -> Tensor")
_LIBRARY.impl("defined_neg", torch.neg, "CompositeExplicitAutograd")


class Layer(torch.nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 16, bias=False)
        self.scale = torch.nn.Parameter(torch.ones(16))
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x)) * self.scale * self.factor + x


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(0.5), Layer(0.25)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# The kernel evaluates the annotations of a function it defines, as code
# generators such as CuTe DSL's compile do, on a name imported in the kernel.
_IMPL = """\
def impl({params}):
    from collections import OrderedDict as LocalName

    def annotated(value: LocalName) -> None:
        pass

    inspect.signature(annotated, eval_str=True)
    return reference({params})


run = impl
"""


def _scheduled() -> tuple[fx.GraphModule, torch.Tensor, tuple[torch.Tensor, ...]]:
    model = Model().cuda()
    params = list(model.parameters())

    def step(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        loss = model(x).square().sum()
        return (loss, *torch.autograd.grad(loss, params))

    x = torch.randn(8, 16, device="cuda")
    trace = make_cudagraph_fx(step)(x)
    ig = trace.indexed_graph
    plan = SchedulePlan(ig)
    for i in ig.fwd.layers:
        layer = ig.fwd.layers[i]
        plan.fuse("tail", [layer.tanh, layer.detach, layer.mul, layer.add])
    return plan.apply().graph_module, x, trace.replay(x)


def _kinds(gm: fx.GraphModule) -> dict[str, str]:
    return {
        node_metadata(n)["fusion_region"]: node_metadata(n)["kernel_cache"]["kind"]
        for n in gm.graph.nodes
        if n.op != "output" and "kernel_cache" in node_metadata(n)
    }


def _cpu_region(
    *,
    custom: bool = False,
    target: torch.library.OpOverload = torch.ops.aten.neg.default,
    args: tuple = (),
) -> fx.GraphModule:
    if custom:

        @custom_op("cgfx_kernel_cache_test::neg")
        def neg(x: torch.Tensor) -> torch.Tensor:
            return -x

        tensor = torch.ones(2)
        target = neg._get_or_create_overload(
            "(Tensor x) -> Tensor",
            tree_structure(((tensor,), {})),
            tree_structure(tensor),
        )
    graph = fx.Graph()
    value = graph.placeholder("value")
    result = graph.call_function(target, (value, *args))
    if custom:
        result = graph.call_function(operator.getitem, (result, 0))
    graph.output(result)
    for i, node in enumerate(graph.nodes):
        node.meta[CUDAGRAPH_FX_METADATA_KEY] = {
            "tensor": {
                "storage_id": i,
                "shape": [2],
                "stride": [1],
                "dtype": "float32",
                "device": "cpu",
            },
        }
    region = fx.GraphModule({}, graph)
    graph = fx.Graph()
    value = graph.placeholder("value")
    value.meta[CUDAGRAPH_FX_METADATA_KEY] = {}
    attr = graph.get_attr("region")
    call = graph.call_function(fused_op, ("region", attr, value))
    attr.meta[CUDAGRAPH_FX_METADATA_KEY] = {}
    call.meta[CUDAGRAPH_FX_METADATA_KEY] = {"fusion_region": "region"}
    graph.output(call)
    return fx.GraphModule({"region": region}, graph)


def _cache_file(gm: fx.GraphModule) -> Path:
    call = next(n for n in gm.graph.nodes if n.target is fused_op)
    return Path(node_metadata(call)["kernel_cache"]["path"])


class TestCudaGraphFxKernelCache(expecttest.TestCase):
    def test_failed_stub_write_removes_temporary_file(self) -> None:
        def write_text(path: Path, text: str) -> None:
            path.touch()
            raise OSError("write failed")

        with TemporaryDirectory() as tmp_dir:
            with (
                patch.object(kernel_cache.Path, "write_text", write_text),
                self.assertRaisesRegex(OSError, "write failed"),
            ):
                substitute_with_kernels(_cpu_region(), tmp_dir)
            assert not list(Path(tmp_dir).glob("pat_*/*"))

    def test_failed_stub_link_removes_temporary_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with (
                patch.object(
                    kernel_cache.os, "link", side_effect=OSError("link failed")
                ),
                self.assertRaisesRegex(OSError, "link failed"),
            ):
                substitute_with_kernels(_cpu_region(), tmp_dir)
            assert not list(Path(tmp_dir).glob("pat_*/*"))

    def test_stub_creation_preserves_existing_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            gm = _cpu_region()
            substitute_with_kernels(gm, tmp_dir)
            path = _cache_file(gm)
            umask = os.umask(0)
            os.umask(umask)
            assert path.stat().st_mode & 0o777 == 0o666 & ~umask
            source = path.read_text() + "\n# Kernel implementation notes.\n"
            path.write_text(source)
            substitute_with_kernels(_cpu_region(), tmp_dir)
            assert path.read_text() == source
            assert list(Path(tmp_dir).glob("pat_*/*")) == [path]

    def test_lookup_rejects_modified_reference_import(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            gm = _cpu_region()
            substitute_with_kernels(gm, tmp_dir)
            path = _cache_file(gm)
            path.write_text(
                path.read_text().replace("import torch\n", "import torch as other\n")
            )
            with self.assertRaisesRegex(ValueError, "reference was modified"):
                substitute_with_kernels(_cpu_region(), tmp_dir)

    def test_lookup_accepts_implementation_with_extra_import(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            gm = _cpu_region()
            substitute_with_kernels(gm, tmp_dir)
            path = _cache_file(gm)
            source = path.read_text().replace(
                "import torch\n", "import inspect\nimport torch\n"
            )
            params = re.search(r"^def impl\((.*)\):$", source, re.M).group(1)
            path.write_text(
                source[: source.index("def impl(")] + _IMPL.format(params=params)
            )
            gm = _cpu_region()
            substitute_with_kernels(gm, tmp_dir)
            assert _kinds(gm) == {"region": "implementation"}
            value = torch.tensor([1.0, -2.0])
            torch.testing.assert_close(gm(value), -value, rtol=0, atol=0)

    def test_reference_rejects_changed_or_rebound_preamble(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            gm = _cpu_region(custom=True)
            substitute_with_kernels(gm, tmp_dir)
            path = _cache_file(gm)
            source = path.read_text()
            call = re.search(r"torch\.ops\.cgfx_kernel_cache_test\.neg\.\w+", source)
            assert call is not None
            changes = (
                source.replace(call.group(0), "torch.ops.aten.relu.default"),
                source.replace("import torch\n", "import torch as other\n"),
                source.replace("import torch\n", ""),
                source + "\nimport operator as torch\n",
                source + "\ndel torch\n",
                source + "\nreference = None\n",
                re.sub(r"(?m)^    return .*$", "    return None", source, count=1),
            )
            for changed in changes:
                with self.subTest(source=changed):
                    path.write_text(changed)
                    with self.assertRaisesRegex(ValueError, "reference was modified"):
                        substitute_with_kernels(_cpu_region(custom=True), tmp_dir)

    def test_reference_allows_kernel_edits_and_formatting(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            gm = _cpu_region(custom=True)
            substitute_with_kernels(gm, tmp_dir)
            path = _cache_file(gm)
            source = path.read_text().replace(" = torch.ops.", "=torch.ops.")
            source = "import inspect\nimport torch.nn.functional\n" + source
            params = re.search(r"^def impl\((.*)\):$", source, re.M).group(1)
            path.write_text(
                source[: source.index("def impl(")] + _IMPL.format(params=params)
            )
            gm = _cpu_region(custom=True)
            substitute_with_kernels(gm, tmp_dir)
            assert _kinds(gm) == {"region": "implementation"}
            value = torch.tensor([1.0, -2.0])
            torch.testing.assert_close(gm(value), -value, rtol=0, atol=0)

    def test_regions_hold_the_operators_fuse_accepts(self) -> None:
        library_neg = torch.ops.cgfx_kernel_cache_test.library_neg.default
        value = torch.tensor([1.0, -2.0])
        with TemporaryDirectory() as tmp_dir:
            gm = _cpu_region(target=library_neg)
            substitute_with_kernels(gm, tmp_dir)
            assert _kinds(gm) == {"region": "reference"}
            torch.testing.assert_close(gm(value), -value, rtol=0, atol=0)
            path = _cache_file(gm)
            source = path.read_text()
            assert f"\nimport {__name__}\n" in source
            assert "torch.ops.cgfx_kernel_cache_test.library_neg.default(" in source
            params = re.search(r"^def impl\((.*)\):$", source, re.M).group(1)
            path.write_text(
                "import inspect\n"
                + source[: source.index("def impl(")]
                + _IMPL.format(params=params)
            )
            gm = _cpu_region(target=library_neg)
            substitute_with_kernels(gm, tmp_dir)
            assert _kinds(gm) == {"region": "implementation"}
            torch.testing.assert_close(gm(value), -value, rtol=0, atol=0)

            defined_neg = torch.ops.cgfx_kernel_cache_test.defined_neg.default
            with self.assertRaisesRegex(ValueError, "has no Python definition"):
                substitute_with_kernels(_cpu_region(target=defined_neg), tmp_dir)

            all_reduce = torch.ops._c10d_functional.all_reduce.default
            with self.assertRaisesRegex(
                ValueError, "^cached region region does not support"
            ):
                substitute_with_kernels(
                    _cpu_region(target=all_reduce, args=("sum", "group")), tmp_dir
                )

    @requires_gpus(1)
    def test_stubs_keep_regions_and_selected_kernels_replace_them(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            gm, x, expected = _scheduled()
            substitute_with_kernels(gm, cache_dir)
            assert _kinds(gm) == {"tail_0": "reference", "tail_1": "reference"}
            files = sorted(cache_dir.glob("pat_*/*.py"))
            # The regions differ only in a constant, which is a call argument.
            assert len(files) == 1
            for file in files:
                source = file.read_text()
                assert source.endswith("run = reference\n")
                assert '"constant": "float",\n      "example": "0.5"' in source
                assert '"device": "cuda",' in source
                params = re.search(r"^def impl\((.*)\):$", source, re.M).group(1)
                stub = source[source.index("def impl(") :]
                file.write_text(
                    source.replace(stub, _IMPL.format(params=params)).replace(
                        "import torch\n", "import inspect\nimport torch\n", 1
                    )
                )

            gm, x, expected = _scheduled()
            substitute_with_kernels(gm, cache_dir)
            assert _kinds(gm) == {
                "tail_0": "implementation",
                "tail_1": "implementation",
            }
            calls = [n for n in gm.graph.nodes if n.target is fused_op]
            assert {n.kwargs["cache_path"] for n in calls} == {
                str(file.resolve()) for file in files
            }
            with torch.no_grad():
                got = CudaGraphFxInterpreter(gm).run(x)
            for g, e in zip(got, expected, strict=True):
                torch.testing.assert_close(g, e, rtol=0, atol=0)
