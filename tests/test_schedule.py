# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
import re
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import expecttest
import pytest
import torch
import torch.fx as fx
from torch._higher_order_ops.flex_attention import flex_attention
from torch._higher_order_ops.invoke_subgraph import invoke_subgraph
from torch_cudagraph_fx import (
    custom_op,
    fused_op,
    IndexedGraph,
    make_cudagraph_fx,
    SchedulePlan,
)
from torch_cudagraph_fx._export_json import export_graph
from torch_cudagraph_fx._interpreter import CudaGraphFxInterpreter
from torch_cudagraph_fx._metadata import (
    CUDAGRAPH_FX_METADATA_KEY,
    FX_NODE_ANNOTATION_KEY,
    node_autograd_meta,
    node_fx_id,
    node_metadata,
)
from torch_cudagraph_fx._schedule import is_barrier, is_fusable
from utils import requires_gpus


@torch.library.custom_op("cgfx_schedule_library_test::scale", mutates_args=())
def _scale_impl(x: torch.Tensor, factor: float) -> torch.Tensor:
    return x * factor


_scale = torch.ops.cgfx_schedule_library_test.scale.default


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


def _trace() -> tuple:
    model = Model().cuda()
    params = list(model.parameters())

    def step(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        loss = model(x).square().sum()
        return (loss, *torch.autograd.grad(loss, params))

    x = torch.randn(8, 16, device="cuda")
    return make_cudagraph_fx(step)(x), x


def _regions(gm: fx.GraphModule) -> dict[str, fx.Node]:
    return {n.args[0]: n for n in gm.graph.nodes if n.target is fused_op}


def _index(graph: fx.Graph) -> IndexedGraph:
    for i, node in enumerate(graph.nodes):
        node.meta[CUDAGRAPH_FX_METADATA_KEY] = {
            "fx_node_id": i,
            "execution": {"stream": 0},
            "autograd": {"phase": "forward"},
            "tensor": {"storage_id": i},
        }
    return IndexedGraph(fx.GraphModule({}, graph))


class TestCudaGraphFxSchedule(expecttest.TestCase):
    @requires_gpus(1)
    def test_fuse_layers_and_replay_identically(self) -> None:
        trace, x = _trace()
        ig = trace.indexed_graph
        plan = SchedulePlan(ig)
        for i in ig.fwd.layers:
            for name, phase in (("fwd", ig.fwd), ("bwd", ig.bwd)):
                plan.fuse(f"{name}_layer", [phase.layers[i]])
        scheduled = plan.apply()
        gm = scheduled.graph_module

        regions = _regions(gm)
        self.assertExpectedInline(
            str(sorted(regions)),
            "['bwd_layer_1', 'bwd_layer_3', 'fwd_layer_0', 'fwd_layer_2']",
        )
        fused = {node_fx_id(op.node) for op in scheduled.resolve(scheduled.fuse)}
        for scope in (ig.fwd.layers, ig.bwd.layers):
            for op in ig.resolve(scope):
                assert node_metadata(op.node)["fx_node_id"] in fused, op

        expected = trace.replay(x)
        with torch.no_grad():
            got = CudaGraphFxInterpreter(gm).run(x)
        assert len(got) == len(expected)
        for g, e in zip(got, expected):
            torch.testing.assert_close(g, e, rtol=0, atol=0)

    @requires_gpus(1)
    def test_fused_regions_take_constants_as_call_arguments(self) -> None:
        trace, _ = _trace()
        ig = trace.indexed_graph
        plan = SchedulePlan(ig)
        for i in ig.fwd.layers:
            layer = ig.fwd.layers[i]
            plan.fuse(
                "tail",
                [layer.tanh, layer.detach, layer.mul, layer.add],
            )
        regions = _regions(plan.apply().graph_module)
        first, second = (regions[f"tail_{i}"] for i in ("0", "1"))
        gm = first.graph.owning_module
        assert gm.get_submodule("tail_0").code == gm.get_submodule("tail_1").code
        assert (first.args[-1], second.args[-1]) == (0.5, 0.25)
        assert first.users and all(u.target is operator.getitem for u in first.users)

    @requires_gpus(1)
    def test_region_must_be_outlinable(self) -> None:
        trace, _ = _trace()
        ig = trace.indexed_graph
        plan = SchedulePlan(ig)
        layer = ig.fwd.layers[0]
        plan.fuse("layer_0", [layer.proj.t, layer.tanh])
        # proj.t feeds mm, whose result tanh consumes.
        with pytest.raises(
            ValueError,
            match=re.escape(
                "fwd.layers[0].proj.mm is missing from the region; add it to form a "
                "valid region (fwd.layers[0].proj.t -> fwd.layers[0].proj.mm -> "
                "fwd.layers[0].tanh)"
            ),
        ):
            plan.apply().graph_module

    @requires_gpus(1)
    def test_fused_custom_operator_keeps_its_traced_return(self) -> None:
        @custom_op("cgfx_schedule_test::double")
        def double(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
            return (double(x) + 1,)

        x = torch.randn(4, 8, device="cuda")
        trace = make_cudagraph_fx(fn)(x)
        ig = trace.indexed_graph
        plan = SchedulePlan(ig)
        plan.fuse("region", [ig.fwd])
        plan = SchedulePlan(plan.apply())
        plan.fuse("inner", [plan.indexed_graph.fuse.region_0.double])
        gm = plan.apply().graph_module
        # The traced operator returns a flat list that the graph indexes; the torch
        # op registered under its name returns the tensor itself.
        with torch.no_grad():
            (got,) = CudaGraphFxInterpreter(gm).run(x)
        (expected,) = trace.replay(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @requires_gpus(1)
    def test_replay_annotations_do_not_require_trace_ids(self) -> None:
        x = torch.randn(4, device="cuda")
        trace = make_cudagraph_fx(lambda x: x.sin().cos())(x)
        expected = trace.replay(x)
        ig = trace.indexed_graph
        for node in ig.graph_module.graph.nodes:
            if node.op != "output":
                node_metadata(node).pop("fx_node_id", None)
        plan = SchedulePlan(ig)
        plan.fuse("sin", [ig.fwd.sin])
        scheduled = plan.apply()
        with patch("torch_cudagraph_fx._interpreter.mark_kernels") as annotate:
            got = CudaGraphFxInterpreter(scheduled.graph_module).run(x)
        torch.testing.assert_close(got, (expected,), rtol=0, atol=0)
        assert annotate.call_count > 0
        assert all(
            call.args[0] == {FX_NODE_ANNOTATION_KEY: True}
            for call in annotate.call_args_list
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            with self.assertRaisesRegex(ValueError, "original trace IDs"):
                export_graph(scheduled.graph_module.graph, path)
            assert not path.exists()

    def test_fuse_allows_mixed_phases_but_requires_one_stream(self) -> None:
        graph = fx.Graph()
        value = graph.placeholder("value")
        nodes = []
        for target in (torch.ops.aten.sin.default, torch.ops.aten.cos.default) * 3:
            value = graph.call_function(target, (value,))
            nodes.append(value)
        graph.output(value)
        ig = _index(graph)
        for node, (phase, stream) in zip(
            nodes,
            [("forward", 0)] * 2 + [("forward", 1)] * 2 + [("backward", 1)] * 2,
        ):
            node_autograd_meta(node)["phase"] = phase
            node_metadata(node)["execution"]["stream"] = stream
        ig = IndexedGraph(ig.graph_module)
        plan = SchedulePlan(ig)
        plan.fuse("invalid", [ig.op_of(nodes[i]) for i in (1, 2)])
        with self.assertRaisesRegex(ValueError, "multiple CUDA streams"):
            plan.apply().graph_module
        plan = SchedulePlan(ig)
        plan.fuse("mixed", [ig.op_of(n) for n in reversed(nodes[2:])])
        plan.fuse("fwd_main", [ig.op_of(n) for n in reversed(nodes[:2])])
        scheduled = plan.apply().graph_module
        self.assertExpectedInline(
            str(list(dict(scheduled.named_children()))),
            "['mixed_0', 'fwd_main_1']",
        )
        self.assertExpectedInline(
            str(list(_regions(scheduled))), "['fwd_main_1', 'mixed_0']"
        )
        assert node_autograd_meta(_regions(scheduled)["mixed_0"]) == {}
        self.assertExpectedInline(
            str(
                [
                    node_autograd_meta(n)["phase"]
                    for n in scheduled.get_submodule("mixed_0").graph.nodes
                    if n.op == "call_function"
                ]
            ),
            "['forward', 'forward', 'backward', 'backward']",
        )
        x = torch.randn(4)
        torch.testing.assert_close(scheduled(x), ig.graph_module(x), rtol=0, atol=0)

    @requires_gpus(1)
    def test_fuse_forward_and_backward_together(self) -> None:
        trace, x = _trace()
        ig = trace.indexed_graph
        plan = SchedulePlan(ig)
        plan.fuse("step", [ig.fwd, ig.bwd])
        scheduled = plan.apply().graph_module
        expected = trace.replay(x)
        with torch.no_grad():
            got = CudaGraphFxInterpreter(scheduled).run(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_fuse_keeps_selected_scalar_outputs_and_projections(self) -> None:
        graph = fx.Graph()
        value = graph.placeholder("value")
        chunks = graph.call_function(torch.ops.aten.split.Tensor, (value, 2))
        first = graph.call_function(operator.getitem, (chunks, 0))
        element = graph.call_function(operator.getitem, (first, 0))
        sine = graph.call_function(torch.ops.aten.sin.default, (element,))
        size = graph.call_function(torch.ops.aten.sym_size.int, (value, 0))
        cosine = graph.call_function(torch.ops.aten.cos.default, (sine,))
        second = graph.call_function(operator.getitem, (chunks, 1))
        graph.output((cosine, second, size))
        ig = _index(graph)
        for node in (chunks, size):
            del node_metadata(node)["tensor"]
        for node in (first, element, second):
            node_metadata(node)["tensor"] = {"storage_id": 0}
        plan = SchedulePlan(ig)
        plan.fuse("region", [ig.fwd])
        scheduled = plan.apply().graph_module
        self.assertExpectedInline(
            str(
                [
                    str(n.target)
                    for n in scheduled.region_0.graph.nodes
                    if n.op == "call_function"
                ]
            ),
            "['aten.split.Tensor', '<built-in function getitem>', '<built-in function getitem>', 'aten.sin.default', 'aten.sym_size.int', 'aten.cos.default', '<built-in function getitem>']",
        )
        x = torch.randn(4)
        torch.testing.assert_close(scheduled(x), ig.graph_module(x), rtol=0, atol=0)

    def test_storage_accesses_order_nodes_around_a_region(self) -> None:
        sin, zero_, rand_like = (
            torch.ops.aten.sin.default,
            torch.ops.aten.zero_.default,
            torch.ops.aten.rand_like.default,
        )
        # The middle node keeps its place before the call, moves after it, or
        # would have to do both: it writes what the region read before it.
        for first_target, middle_target, outcome in (
            (sin, zero_, "missing from the region"),
            (zero_, sin, "moved"),
            (sin, rand_like, "kept"),
            (rand_like, rand_like, "moved"),
        ):
            for outlined in (False, True):
                with self.subTest(
                    first=first_target, middle=middle_target, outlined=outlined
                ):
                    graph = fx.Graph()
                    value = graph.placeholder("value")
                    first = graph.call_function(first_target, (value,))
                    middle = graph.call_function(middle_target, (value,))
                    last = graph.call_function(torch.ops.aten.cos.default, (value,))
                    graph.output((first, middle, last))
                    ig = _index(graph)
                    plan = SchedulePlan(ig)
                    if outlined:
                        plan.fuse("middle", [ig.op_of(middle)])
                    plan.fuse("region", [ig.op_of(first), ig.op_of(last)])
                    if outcome not in ("moved", "kept"):
                        with self.assertRaisesRegex(ValueError, outcome):
                            plan.apply()
                        continue
                    scheduled = plan.apply().graph_module
                    names = [
                        n.args[0] if n.target is fused_op else n.name
                        for n in scheduled.graph.nodes
                        if n.op != "get_attr"
                    ]
                    middle_name = "middle_0" if outlined else middle.name
                    region_name = f"region_{int(outlined)}"
                    assert (names.index(middle_name) > names.index(region_name)) == (
                        outcome == "moved"
                    )
                    x = torch.randn(4)
                    torch.manual_seed(0)
                    expected = ig.graph_module(x.clone())
                    torch.manual_seed(0)
                    got = scheduled(x.clone())
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_fuse_moves_consumers_after_the_call_and_keeps_producers(self) -> None:
        graph = fx.Graph()
        value, buffer, source = (
            graph.placeholder(name) for name in ("value", "buffer", "source")
        )
        first = graph.call_function(torch.ops.aten.sin.default, (value,))
        consumer = graph.call_function(torch.ops.aten.cos.default, (first,))
        graph.call_function(torch.ops.aten.copy_.default, (buffer, source))
        scaled = graph.call_function(torch.ops.aten.exp.default, (buffer,))
        last = graph.call_function(torch.ops.aten.mul.Tensor, (first, scaled))
        graph.output((consumer, last))
        ig = _index(graph)
        plan = SchedulePlan(ig)
        plan.fuse("region", [ig.op_of(first), ig.op_of(last)])
        scheduled = plan.apply().graph_module
        # The copy writes the buffer before the region reads it, and exp feeds
        # the region: both stay. cos uses the region's output: it moves.
        self.assertExpectedInline(
            str([n.name for n in scheduled.graph.nodes if n.op == "call_function"]),
            "['copy__default', 'exp_default', 'fused_op', 'getitem', 'getitem_1', 'cos_default']",
        )
        value, buffer, source = torch.randn(4), torch.randn(4), torch.randn(4)
        torch.testing.assert_close(
            scheduled(value, buffer.clone(), source),
            ig.graph_module(value, buffer.clone(), source),
            rtol=0,
            atol=0,
        )

    def test_fuse_adds_the_views_between_selected_operators(self) -> None:
        aten = torch.ops.aten
        graph = fx.Graph()
        value, weight = graph.placeholder("value"), graph.placeholder("weight")
        graph.call_function(aten.view.default, (weight, [2, 2]), name="before")
        first = graph.call_function(aten.sin.default, (value,), name="first")
        chunks = graph.call_function(aten.split.Tensor, (first, 2), name="chunks")
        half = graph.call_function(operator.getitem, (chunks, 0), name="half")
        row = graph.call_function(aten.unsqueeze.default, (half, 0), name="row")
        graph.call_function(aten.view.default, (value, [2, 2]), name="unrelated")
        square = graph.call_function(aten.view.default, (weight, [2, 2]), name="square")
        column = graph.call_function(aten.select.int, (square, 1, 0), name="column")
        last = graph.call_function(aten.mul.Tensor, (row, column), name="last")
        graph.call_function(aten.view.default, (last, [2]), name="after")
        outside = ("before", "unrelated", "after")
        graph.output(tuple(n for n in graph.nodes if n.name in outside))
        ig = _index(graph)
        plan = SchedulePlan(ig)
        plan.fuse("region", [ig.op_of(first), ig.op_of(last)])
        scheduled = plan.apply().graph_module
        # Views read the region's results (chunks, row) or feed it (square,
        # column); those before, after or beside it stay outside.
        self.assertExpectedInline(
            str([n.name for n in scheduled.graph.nodes if n.op == "call_function"]),
            "['before', 'unrelated', 'fused_op', 'getitem', 'after']",
        )
        region = scheduled.region_0.graph.nodes
        self.assertExpectedInline(
            str([str(n.target) for n in region if n.op == "call_function"]),
            "['aten.sin.default', 'aten.split.Tensor', '<built-in function getitem>', 'aten.unsqueeze.default', 'aten.view.default', 'aten.select.int', 'aten.mul.Tensor']",
        )
        value, weight = torch.randn(4), torch.randn(4)
        torch.testing.assert_close(
            scheduled(value, weight), ig.graph_module(value, weight), rtol=0, atol=0
        )

    def test_fuse_library_custom_operators(self) -> None:
        graph = fx.Graph()
        value = graph.placeholder("value")
        scaled = graph.call_function(_scale, (value, 2.0))
        graph.output(graph.call_function(torch.ops.aten.sin.default, (scaled,)))
        ig = _index(graph)
        plan = SchedulePlan(ig)
        plan.fuse("region", [ig.fwd])
        scheduled = plan.apply().graph_module
        x = torch.randn(4)
        torch.testing.assert_close(scheduled(x), ig.graph_module(x), rtol=0, atol=0)

    def test_barriers_are_operators_with_effects_outside_their_schema(self) -> None:
        targets = {
            "aten": torch.ops.aten.sin.default,
            "random": torch.ops.aten.rand_like.default,
            "custom_op": _scale,
            "functional_collective": torch.ops._c10d_functional.all_reduce.default,
            "eager_collective": torch.ops.c10d.allreduce_.default,
            "event": torch.ops.torch_cudagraph_fx.record_event.default,
            "flex_attention": flex_attention,
            "invoke_subgraph": invoke_subgraph,
        }
        graph = fx.Graph()
        value = graph.placeholder("value")
        nodes = {name: graph.call_function(t, (value,)) for name, t in targets.items()}
        self.assertExpectedInline(
            "\n".join(
                f"{name}: fusable={is_fusable(n)} barrier={is_barrier(n)}"
                for name, n in nodes.items()
            ),
            """\
aten: fusable=True barrier=False
random: fusable=True barrier=False
custom_op: fusable=True barrier=False
functional_collective: fusable=False barrier=False
eager_collective: fusable=False barrier=True
event: fusable=False barrier=True
flex_attention: fusable=False barrier=False
invoke_subgraph: fusable=False barrier=True""",
        )

    def test_regions_cross_independent_nodes_but_not_barriers(self) -> None:
        # Cross-stream ordering goes through events, so an independent node on
        # another stream, here an earlier region, keeps its place.
        for stream in (0, 1):
            with self.subTest(stream=stream):
                graph = fx.Graph()
                value = graph.placeholder("value")
                first = graph.call_function(torch.ops.aten.sin.default, (value,))
                middle = graph.call_function(torch.ops.aten.cos.default, (value,))
                last = graph.call_function(torch.ops.aten.sin.default, (first,))
                graph.output((middle, last))
                ig = _index(graph)
                node_metadata(middle)["execution"]["stream"] = stream
                plan = SchedulePlan(ig)
                plan.fuse("middle", [ig.op_of(middle)])
                plan.fuse("region", [ig.op_of(first), ig.op_of(last)])
                scheduled = plan.apply().graph_module
                value = torch.randn(4)
                torch.testing.assert_close(
                    scheduled(value), ig.graph_module(value), rtol=0, atol=0
                )
        graph = fx.Graph()
        value = graph.placeholder("value")
        event = graph.call_function(torch.ops.torch_cudagraph_fx.new_event.default)
        first = graph.call_function(torch.ops.aten.sin.default, (value,))
        graph.call_function(torch.ops.torch_cudagraph_fx.record_event.default, (event,))
        last = graph.call_function(torch.ops.aten.sin.default, (first,))
        graph.output(last)
        ig = _index(graph)
        plan = SchedulePlan(ig)
        plan.fuse("region", [ig.op_of(first), ig.op_of(last)])
        with self.assertRaisesRegex(ValueError, "crosses the barrier record_event"):
            plan.apply()

    def test_fuse_rejects_unsupported_selected_ops(self) -> None:
        graph = fx.Graph()
        value = graph.placeholder("value")
        sine = graph.call_function(torch.ops.aten.sin.default, (value,))
        reduced = graph.call_function(
            torch.ops._c10d_functional.all_reduce.default, (sine, "sum", "group")
        )
        graph.output(reduced)
        ig = _index(graph)
        plan = SchedulePlan(ig)
        plan.fuse("region", [ig.fwd])
        with self.assertRaisesRegex(ValueError, "cannot fuse"):
            plan.apply()

    def test_fuse_deduplicates_names_and_rejects_overlapping_selections(self) -> None:
        graph = fx.Graph()
        value = graph.placeholder("value")
        names = ("norm", "norm", "class", "layers[0].norm", "norm")
        nodes = []
        for _ in names:
            value = graph.call_function(torch.ops.aten.sin.default, (value,))
            nodes.append(value)
        graph.output(value)
        ig = _index(graph)
        plan = SchedulePlan(ig)
        for name, node in zip(names, nodes):
            plan.fuse(name, [ig.op_of(node)])
        scheduled = plan.apply().graph_module
        self.assertExpectedInline(
            str(list(_regions(scheduled))),
            "['norm_0', 'norm_1', 'class_2', 'layers[0].norm_3', 'norm_4']",
        )
        x = torch.randn(4)
        torch.testing.assert_close(scheduled(x), ig.graph_module(x), rtol=0, atol=0)
        for target in ("fwd.sin", "fwd.sin[*]"):
            with self.assertRaisesRegex(TypeError, "handles"):
                plan.fuse("norm", [target])
        plan.fuse("norm", [ig.fwd])
        with self.assertRaises(KeyError):
            plan.apply()

    def test_fuse_requires_producers_before_consumers(self) -> None:
        graph = fx.Graph()
        value = graph.placeholder("value")
        sine = graph.call_function(torch.ops.aten.sin.default, (value,))
        cosine = graph.call_function(torch.ops.aten.cos.default, (sine,))
        graph.output(cosine)
        ig = _index(graph)
        sine.prepend(cosine)
        plan = SchedulePlan(ig)
        plan.fuse("region", [ig.fwd])
        with self.assertRaisesRegex(RuntimeError, "used before it has been defined"):
            plan.apply()

    def test_regions_rewrite_a_shared_consumer(self) -> None:
        graph = fx.Graph()
        value = graph.placeholder("value")
        nodes = [
            graph.call_function(torch.ops.aten.neg.default, (value,)) for _ in range(64)
        ]
        joined = graph.call_function(torch.ops.aten.cat.default, (nodes,))
        tail = graph.call_function(torch.ops.aten.cos.default, (joined,))
        graph.output(tail)
        ig = _index(graph)
        original = str(graph)
        plan = SchedulePlan(ig)
        plan.fuse("wide", [ig.op_of(n) for n in nodes])
        plan.fuse("tail", [ig.op_of(tail)])
        scheduled = plan.apply().graph_module
        value = torch.randn(4)
        torch.testing.assert_close(
            scheduled(value), ig.graph_module(value), rtol=0, atol=0
        )
        assert str(graph) == original

    def test_trace_ids_are_optional_provenance_without_changing_trace(
        self,
    ) -> None:
        graph = fx.Graph()
        value = graph.placeholder("value")
        first = graph.call_function(torch.ops.aten.sin.default, (value,))
        second = graph.call_function(torch.ops.aten.cos.default, (first,))
        backward = graph.call_function(torch.ops.aten.neg.default, (second,))
        graph.output((first, backward))
        ig = _index(graph)
        node_autograd_meta(backward).update(
            phase="backward", forward_node_ids=[node_fx_id(first), node_fx_id(second)]
        )
        original = deepcopy([node_metadata(n) for n in graph.nodes])
        plan = SchedulePlan(ig)
        plan.fuse("region", [ig.op_of(first), ig.op_of(second)])
        scheduled = plan.apply().graph_module
        nodes = [n for n in scheduled.graph.nodes if n.op != "output"]
        call = _regions(scheduled)["region_0"]
        assert "fx_node_id" not in node_metadata(call)
        assert all("fx_node_id" not in node_metadata(n) for n in call.users)
        remaining = next(n for n in nodes if n.target is torch.ops.aten.neg.default)
        assert node_autograd_meta(remaining)["forward_node_ids"] == [1, 2]
        assert node_fx_id(remaining) == node_fx_id(backward)
        assert [node_metadata(n) for n in graph.nodes] == original
        x = torch.randn(4)
        torch.testing.assert_close(scheduled(x), ig.graph_module(x), rtol=0, atol=0)

        for node in graph.nodes:
            node_metadata(node).pop("fx_node_id", None)
        scheduled = plan.apply().graph_module
        torch.testing.assert_close(scheduled(x), ig.graph_module(x), rtol=0, atol=0)

    def test_later_fusions_resolve_updated_index_and_preserve_source(self) -> None:
        graph = fx.Graph()
        value = graph.placeholder("value")
        first = graph.call_function(torch.ops.aten.sin.default, (value,))
        second = graph.call_function(torch.ops.aten.sin.default, (first,))
        last = graph.call_function(torch.ops.aten.cos.default, (second,))
        graph.output(last)
        ig = _index(graph)
        original = str(graph)
        plan = SchedulePlan(ig)
        plan.fuse("pair", [ig.fwd.sin])
        with self.assertRaises(AttributeError):
            plan.fuse("inner", [ig.fuse.pair_0.sin[0]])
        outlined = plan.apply()
        plan = SchedulePlan(outlined)
        plan.fuse("inner", [outlined.fuse.pair_0.sin[0]])
        plan.fuse("tail", [outlined.fwd.cos])
        scheduled = plan.apply()
        self.assertExpectedInline(
            "\n".join(op.path for op in scheduled),
            """\
fuse.pair_0.sin[1]
fuse.inner_1.sin
fuse.tail_2.cos""",
        )
        assert "inner_1" in _regions(scheduled.graph_module.pair_0)
        second_plan = SchedulePlan(scheduled)
        second_plan.fuse("inner", [scheduled.fuse.pair_0.sin[1]])
        again = second_plan.apply()
        assert "inner_3" in _regions(again.graph_module.pair_0)
        assert scheduled.fuse.pair_0.sin[1].path == "fuse.pair_0.sin[1]"
        assert str(graph) == original
        x = torch.randn(4)
        for result in (scheduled, again):
            torch.testing.assert_close(
                result.graph_module(x), ig.graph_module(x), rtol=0, atol=0
            )

    def test_dependent_regions_preserve_intervening_mutation(self) -> None:
        graph = fx.Graph()
        value, other = graph.placeholder("value"), graph.placeholder("other")
        first = graph.call_function(torch.ops.aten.sin.default, (other,))
        consumer = graph.call_function(torch.ops.aten.cos.default, (other,))
        graph.call_function(torch.ops.aten.add_.Tensor, (value, 1))
        late = graph.call_function(torch.ops.aten.sin.default, (value,))
        last = graph.call_function(torch.ops.aten.add.Tensor, (first, late))
        result = graph.call_function(torch.ops.aten.add.Tensor, (consumer, first))
        graph.output((last, result, value))
        ig = _index(graph)
        plan = SchedulePlan(ig)
        plan.fuse("consumer", [ig.op_of(consumer), ig.op_of(result)])
        plan.fuse("producer", [ig.op_of(first), ig.op_of(last)])
        scheduled = plan.apply().graph_module
        self.assertExpectedInline(
            str(list(dict(scheduled.named_children()))), "['consumer_0', 'producer_1']"
        )
        self.assertExpectedInline(
            str(list(_regions(scheduled))), "['producer_1', 'consumer_0']"
        )
        value, other = torch.randn(4), torch.randn(4)
        torch.testing.assert_close(
            scheduled(value.clone(), other),
            ig.graph_module(value.clone(), other),
            rtol=0,
            atol=0,
        )

    def test_regions_cannot_form_a_dependency_cycle(self) -> None:
        graph = fx.Graph()
        value = graph.placeholder("value")
        first = graph.call_function(torch.ops.aten.sin.default, (value,))
        second = graph.call_function(torch.ops.aten.cos.default, (value,))
        third = graph.call_function(torch.ops.aten.sin.default, (second,))
        fourth = graph.call_function(torch.ops.aten.cos.default, (first,))
        graph.output((third, fourth))
        ig = _index(graph)
        plan = SchedulePlan(ig)
        plan.fuse("first", [ig.op_of(first), ig.op_of(third)])
        plan.fuse("second", [ig.op_of(second), ig.op_of(fourth)])
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "fuse.first_0 is missing from the region but cannot be fused; split "
                "the region (fwd.cos[0] -> fuse.first_0 -> fwd.cos[1])"
            ),
        ):
            plan.apply().graph_module

    def test_region_writes_exclude_accesses_outside_its_interval(self) -> None:
        graph = fx.Graph()
        value, other = graph.placeholder("value"), graph.placeholder("other")
        graph.call_function(torch.ops.aten.zero_.default, (value,))
        first = graph.call_function(torch.ops.aten.add_.Tensor, (value, 1))
        middle = graph.call_function(torch.ops.aten.cos.default, (other,))
        last = graph.call_function(torch.ops.aten.sin.default, (value,))
        graph.call_function(torch.ops.aten.add_.Tensor, (value, 1))
        graph.output((last, middle, value))
        ig = _index(graph)
        plan = SchedulePlan(ig)
        plan.fuse("region", [ig.op_of(first), ig.op_of(last)])
        scheduled = plan.apply().graph_module
        value, other = torch.randn(4), torch.randn(4)
        torch.testing.assert_close(
            scheduled(value.clone(), other),
            ig.graph_module(value.clone(), other),
            rtol=0,
            atol=0,
        )
