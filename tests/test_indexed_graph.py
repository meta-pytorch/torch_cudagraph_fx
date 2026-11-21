# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import patch

import expecttest
import pytest
import torch
from torch_cudagraph_fx import (
    fused_op,
    IndexedGraph,
    make_cudagraph_fx,
    Op,
    OpGroup,
    Scope,
)
from torch_cudagraph_fx._metadata import CUDAGRAPH_FX_METADATA_KEY, node_metadata
from utils import requires_gpus


class Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 16, bias=False)
        self.scale = torch.nn.Parameter(torch.ones(16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x)) * self.scale * 0.5


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(), Layer()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TrainingStep(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loss = self.model(x).square().sum()
        loss.backward()
        return loss


class TestCudaGraphFxIndexedGraph(expecttest.TestCase):
    @requires_gpus(1)
    def test_index_addresses_every_operator_by_phase_module_and_name(self) -> None:
        step = TrainingStep().cuda()
        x = torch.randn(8, 16, device="cuda")
        ig = make_cudagraph_fx(step)(x).indexed_graph

        paths = [op.path for op in ig]
        assert len(set(paths)) == len(paths)
        assert all(ig[path] is op for path, op in zip(paths, ig))

        layers = ig.fwd.model.layers
        assert isinstance(layers, Scope) and list(layers) == ["0", "1"]
        layer = layers[0]
        assert ig["fwd.model.layers[0]"] is layer
        self.assertExpectedInline(
            "\n".join(op.path for op in ig.resolve(layer.proj)),
            """\
fwd.model.layers[0].proj.t
fwd.model.layers[0].proj.mm""",
        )
        muls = layer.mul
        assert isinstance(muls, OpGroup) and len(muls) == 2
        self.assertExpectedInline(
            "\n".join(op.path for op in muls),
            """\
fwd.model.layers[0].mul[0]
fwd.model.layers[0].mul[1]""",
        )
        assert isinstance(muls[1], Op) and ig["fwd.model.layers[0].mul[1]"] is muls[1]
        assert ig.select("fwd.model.layers[0].*") == ig.resolve(layer)
        assert ig.resolve("fwd.model.layers[0]") == ig.resolve(layer)

        # Backward ran inside TrainingStep.forward, so its ops first recorded the
        # root module; each is renamed after the layer its forward ops ran in.
        for i in ("0", "1"):
            bwd_layer = ig.bwd.model.layers[i]
            names = {op.group.name for op in ig.resolve(bwd_layer)}
            assert "tanh_backward" in names and "mm" in names
        assert "tanh_backward" not in ig.bwd
        owned = sum(len(ig.resolve(scope)) for scope in (ig.fwd, ig.bwd))
        assert owned == len(ig)

        with pytest.raises(KeyError, match="closest"):
            layer["prj"]

        # An operator without a module sits directly under its phase.
        node = ig.bwd.model.layers[1].tanh_backward[0].node
        del node_metadata(node)["module_fqn"]
        assert IndexedGraph(ig.graph_module)["bwd.tanh_backward"].node is node

    def test_select_bracket_wildcards_stay_within_one_index(self) -> None:
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        value.meta[CUDAGRAPH_FX_METADATA_KEY] = {}
        outputs = []
        for module in (
            "layers.0.norm",
            "layers.1.norm",
            "layers.10.norm",
            "layers.0.blocks.1.norm",
            "layers.0.norm.inner",
        ):
            node = graph.call_function(torch.ops.aten.add.Tensor, (value, 1))
            node.meta[CUDAGRAPH_FX_METADATA_KEY] = {"module_fqn": module}
            outputs.append(node)
        graph.output(tuple(outputs))
        ig = IndexedGraph(torch.fx.GraphModule({}, graph))

        selected = ig.select("fwd.layers[*].norm.*")
        self.assertExpectedInline(
            "\n".join(op.path for op in selected),
            """\
fwd.layers[0].norm.add
fwd.layers[1].norm.add
fwd.layers[10].norm.add
fwd.layers[0].norm.inner.add""",
        )
        assert ig.resolve("fwd.layers[*].norm.*") == selected
        assert ig.select("fwd.layers[?].norm.*") == [
            op for op in selected if "layers[10]" not in op.path
        ]
        assert ig.select("fwd.layers[1*].norm.*") == selected[1:3]
        assert ig.select("fwd.layers[*].blocks[*].norm.*") == [
            ig["fwd.layers[0].blocks[1].norm.add"]
        ]
        assert ig.select("fwd.layers[0].*") == ig.resolve(ig.fwd.layers[0])

    def test_replace_region_updates_only_affected_entries(self) -> None:
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        first = graph.call_function(torch.ops.aten.add.Tensor, (value, 1))
        second = graph.call_function(torch.ops.aten.add.Tensor, (value, 2))
        graph.output((first, second))
        for node in graph.nodes:
            node.meta[CUDAGRAPH_FX_METADATA_KEY] = {}
        ig = IndexedGraph(torch.fx.GraphModule({}, graph))
        removed, survivor = ig.fwd.add
        group, scope = ig.fwd.add, ig.fwd
        sub = torch.fx.Graph()
        arg = sub.placeholder("value")
        arg.meta[CUDAGRAPH_FX_METADATA_KEY] = {}
        copied = sub.node_copy(first, lambda _: arg)
        sub.output(copied)
        ig.graph_module.add_submodule("region_0", torch.fx.GraphModule({}, sub))
        with graph.inserting_before(first):
            region = graph.get_attr("region_0")
        with graph.inserting_before(first):
            call = graph.call_function(fused_op, ("region_0", region, value))
        region.meta[CUDAGRAPH_FX_METADATA_KEY] = call.meta[
            CUDAGRAPH_FX_METADATA_KEY
        ] = {}
        first.replace_all_uses_with(call)
        graph.erase_node(first)
        with patch.object(type(graph.nodes), "__iter__", side_effect=AssertionError):
            ig.replace_region([first], call, [copied])
        assert ig.fwd is scope and ig.fwd.add is group
        assert ig["fwd.add[1]"] is survivor and group[1] is survivor
        assert list(group) == [survivor]
        assert ig.op_of(first) is None
        assert ig["fuse.region_0.add"].node is copied
        self.assertExpectedInline(
            "\n".join(op.path for op in ig),
            """\
fwd.add[1]
fuse.region_0.add""",
        )
        self.assertExpectedInline(
            "\n".join(op.path for op in IndexedGraph(ig.graph_module)),
            """\
fuse.region_0.add
fwd.add""",
        )
        with self.assertRaises(KeyError):
            ig["fwd.add[0]"]
        with self.assertRaisesRegex(ValueError, "removed operators"):
            ig.resolve(removed)
        clone = ig.copy()
        assert [op.path for op in clone] == [op.path for op in ig]
        assert clone["fwd.add[1]"].node is not survivor.node
        assert clone["fuse.region_0.add"].node is not copied
        assert (
            clone["fuse.region_0.add"].node.graph is clone.graph_module.region_0.graph
        )
        with self.assertRaisesRegex(ValueError, "another index"):
            clone.resolve(survivor)
        node_metadata(clone["fuse.region_0.add"].node)["module_fqn"] = "changed"
        assert node_metadata(copied) == {}
        clone.graph_module.graph.lint()
        value = torch.randn(4)
        torch.testing.assert_close(clone.graph_module(value), (value + 1, value + 2))

    def test_attributes_reach_any_submodule_or_operator_name(self) -> None:
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        cast = graph.call_function(torch.ops.aten._to_copy.default, (value,))
        shift = graph.call_function(torch.ops.aten.__lshift__.Scalar, (cast, 16))
        graph.output(shift)
        for node in graph.nodes:
            node.meta[CUDAGRAPH_FX_METADATA_KEY] = {}
        node_metadata(shift)["module_fqn"] = "path.keys"
        ig = IndexedGraph(torch.fx.GraphModule({}, graph))
        assert ig.fwd._to_copy is ig.fwd["_to_copy"]
        assert ig.fwd.path.keys.__lshift__[0] is ig["fwd.path.keys.__lshift__"]
        assert list(ig.fwd) == ["path"] and len(ig.fwd.path.keys) == 0
        assert not hasattr(ig.fwd, "_repr_html_")
        assert ig.copy().fwd._to_copy[0].node.target is cast.target

    def test_copy_shares_parameters_but_not_graphs_or_metadata(self) -> None:
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        weight = graph.get_attr("layer.weight")
        result = graph.call_function(torch.ops.aten.mul.Tensor, (value, weight))
        graph.output(result)
        for node in graph.nodes:
            node.meta[CUDAGRAPH_FX_METADATA_KEY] = {"execution": {"stream": 0}}
        parameter = torch.nn.Parameter(torch.ones(4))
        ig = IndexedGraph(torch.fx.GraphModule({"layer.weight": parameter}, graph))
        clone = ig.copy()
        assert clone.graph_module.layer is not ig.graph_module.layer
        assert clone.graph_module.layer.weight is parameter
        node_metadata(clone["fwd.mul"].node)["execution"]["stream"] = 1
        assert node_metadata(result)["execution"]["stream"] == 0
