# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import expecttest
import torch
from torch_cudagraph_fx import custom_op, make_cudagraph_fx
from torch_cudagraph_fx._metadata import node_autograd_meta, node_fx_id
from utils import requires_gpus


class TestCudaGraphFxExportJson(expecttest.TestCase):
    @requires_gpus(1)
    def test_export_tensor_metadata(self) -> None:
        x = torch.ones(4, device="cuda")
        weight = torch.nn.Parameter(torch.ones_like(x))
        calls = []

        def step(x: torch.Tensor) -> torch.Tensor:
            calls.append(True)
            x.add_(1)
            return torch.add(x, weight).to(torch.float64)

        fx_trace = make_cudagraph_fx(step)(x)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            fx_trace.export_json(path)
            exported = path.read_text()
        data = json.loads(exported)
        assert calls == [True]
        torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)
        assert exported.endswith("\n")
        assert set(data) == {"format_version", "operators", "tensors", "outputs"}
        assert data["format_version"] == 1
        operators, tensors = data["operators"], data["tensors"]
        assert set(operators) == {
            str(node_fx_id(node))
            for node in fx_trace.graph.nodes
            if node.op != "output"
        }
        assert operators["0"] == {
            "schema": "placeholder(str name) -> Tensor",
            "arguments": {"name": "arg_0"},
            "returns": {"output": {"fx_node_id": 0}},
        }
        self.assertExpectedInline(
            json.dumps(tensors["0"], indent=2),
            """\
{
  "fx_producer_id": 0,
  "storage_id": 0,
  "tensor_type": "input",
  "tensor_meta": {
    "shape": [
      4
    ],
    "stride": [
      1
    ],
    "dtype": "float32",
    "device": "cuda:0",
    "layout": "strided",
    "element_size": 4,
    "nbytes": 16,
    "requires_grad": false,
    "is_leaf": true
  }
}""",
        )
        weight_id = next(
            key for key, op in operators.items() if op["schema"].startswith("get_attr(")
        )
        assert tensors[weight_id]["tensor_type"] == "weight"
        assert tensors[weight_id]["tensor_meta"]["requires_grad"] is True
        add = next(
            op
            for op in operators.values()
            if op["schema"].startswith("aten::add.Tensor")
        )
        assert add["arguments"]["other"] == {"fx_node_id": int(weight_id)}
        assert add["arguments"]["alpha"] == 1
        cast_id, cast = next(
            (key, op)
            for key, op in operators.items()
            if op["schema"].startswith("aten::_to_copy")
        )
        assert cast["arguments"]["self"] == add["returns"]["output"]
        assert cast["arguments"]["dtype"] == "float64"
        assert cast["returns"] == {"output": {"fx_node_id": int(cast_id)}}
        assert tensors[cast_id]["tensor_meta"]["dtype"] == "float64"
        assert data["outputs"] == [{"fx_node_id": int(cast_id)}]

        expected_input = x.clone()
        expected = step(expected_input)
        got = fx_trace.replay(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        torch.testing.assert_close(x, expected_input, rtol=0, atol=0)

    @requires_gpus(1)
    def test_shared_storage_preserves_parameter_and_buffer_types(self) -> None:
        for create_parameter_during_trace in (False, True):
            with self.subTest(
                create_parameter_during_trace=create_parameter_during_trace
            ):
                buffer = torch.ones(8, device="cuda")
                weight = (
                    None
                    if create_parameter_during_trace
                    else torch.nn.Parameter(buffer.view(2, 4))
                )

                def step(
                    buffer: torch.Tensor = buffer,
                    weight: torch.nn.Parameter | None = weight,
                    create_parameter_during_trace: bool = create_parameter_during_trace,
                ) -> tuple[torch.Tensor, ...]:
                    buffer_view = buffer.view(2, 4)
                    parameter = (
                        torch.nn.Parameter(buffer_view)
                        if create_parameter_during_trace
                        else weight
                    )
                    assert isinstance(parameter, torch.nn.Parameter)
                    weight_view = parameter.t()
                    buffer_slice = buffer[1:5]
                    computed = (parameter + 1).flatten()
                    with torch.no_grad():
                        written_buffer = torch.add(parameter, 1, out=buffer_view)
                    return (
                        buffer_view,
                        parameter,
                        weight_view,
                        buffer_slice,
                        computed,
                        written_buffer,
                    )

                trace = make_cudagraph_fx(step)()
                with TemporaryDirectory() as tmp_dir:
                    path = Path(tmp_dir) / "trace.json"
                    trace.export_json(path)
                    data = json.loads(path.read_text())
                tensors = [
                    data["tensors"][str(ref["fx_node_id"])] for ref in data["outputs"]
                ]
                assert [tensor["tensor_type"] for tensor in tensors] == [
                    "constant",
                    "weight",
                    "weight",
                    "constant",
                    "forward_tensor",
                    "constant",
                ]
                storage_id = tensors[0]["storage_id"]
                assert all(tensors[i]["storage_id"] == storage_id for i in (1, 2, 3, 5))
                assert tensors[4]["storage_id"] != storage_id

    @requires_gpus(1)
    def test_storage_ids_distinguish_empty_allocations(self) -> None:
        first = torch.empty(0, device="cuda")
        second = torch.empty(0, device="cuda")
        assert first.data_ptr() == second.data_ptr() == 0

        trace = make_cudagraph_fx(lambda a, b: (a, b, a.view(-1)))(first, second)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            trace.export_json(path)
            data = json.loads(path.read_text())
        tensors = [data["tensors"][str(ref["fx_node_id"])] for ref in data["outputs"]]
        assert tensors[0]["storage_id"] == tensors[2]["storage_id"]
        assert tensors[0]["storage_id"] != tensors[1]["storage_id"]

    @requires_gpus(1)
    def test_export_multiple_returns_and_tensor_lists(self) -> None:
        def fn(x: torch.Tensor) -> tuple[object, ...]:
            return torch.max(x, dim=1), torch.split(x, [2, 2])

        x = torch.ones((4, 3), device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            fx_trace.export_json(path)
            data = json.loads(path.read_text())
        operators, tensors = data["operators"], data["tensors"]
        max_id, max_op = next(
            (key, op)
            for key, op in operators.items()
            if op["schema"].startswith("aten::max.dim")
        )
        assert set(max_op["returns"]) == {"output_0", "output_1"}
        for index, dtype in enumerate(("float32", "int64")):
            tensor_id = str(max_op["returns"][f"output_{index}"]["fx_node_id"])
            tensor = tensors[tensor_id]
            assert tensor["fx_producer_id"] == int(max_id)
            assert tensor["tensor_meta"]["dtype"] == dtype
            assert operators[tensor_id]["arguments"] == {
                "input": {"fx_node_id": int(max_id)},
                "index": index,
            }
        split_id, split_op = next(
            (key, op)
            for key, op in operators.items()
            if op["schema"].startswith("aten::split_with_sizes")
        )
        assert split_op["arguments"]["split_sizes"] == [2, 2]
        assert split_op["arguments"]["dim"] == 0
        assert set(split_op["returns"]) == {"output"}
        assert len(split_op["returns"]["output"]) == 2
        assert len({ref["fx_node_id"] for ref in split_op["returns"]["output"]}) == 2
        for reference in split_op["returns"]["output"]:
            assert tensors[str(reference["fx_node_id"])]["fx_producer_id"] == int(
                split_id
            )
            assert (
                tensors[str(reference["fx_node_id"])]["storage_id"]
                == tensors["0"]["storage_id"]
            )
        assert data["outputs"] == [
            max_op["returns"]["output_0"],
            max_op["returns"]["output_1"],
            *split_op["returns"]["output"],
        ]

        expected = fn(x)
        got = fx_trace.replay(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @requires_gpus(1)
    def test_export_custom_op_tensor_returns(self) -> None:
        @custom_op("cudagraph_fx_test::export_single")
        def single(x: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
            return x * scale

        @custom_op("cudagraph_fx_test::export_mixed")
        def mixed(x: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor]:
            return x + 1, 7, x + 2

        def fn(x: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor]:
            return mixed(single(x))

        x = torch.ones(4, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            fx_trace.export_json(path)
            data = json.loads(path.read_text())
        single_id, single_op = next(
            (key, op)
            for key, op in data["operators"].items()
            if op["schema"].startswith("cudagraph_fx_test::export_single(")
        )
        assert single_op["schema"] == (
            "cudagraph_fx_test::export_single(Tensor x, float scale) -> Tensor"
        )
        assert single_op["arguments"] == {"x": {"fx_node_id": 0}, "scale": 2.0}
        assert set(single_op["returns"]) == {"output"}
        single_output = single_op["returns"]["output"]
        assert data["tensors"][str(single_output["fx_node_id"])][
            "fx_producer_id"
        ] == int(single_id)
        mixed_id, mixed_op = next(
            (key, op)
            for key, op in data["operators"].items()
            if op["schema"].startswith("cudagraph_fx_test::export_mixed(")
        )
        assert mixed_op["arguments"] == {"x": single_output}
        assert set(mixed_op["returns"]) == {"output_0", "output_2"}
        assert mixed_op["returns"]["output_0"] != mixed_op["returns"]["output_2"]
        for reference in mixed_op["returns"].values():
            assert data["tensors"][str(reference["fx_node_id"])][
                "fx_producer_id"
            ] == int(mixed_id)
        assert data["outputs"] == [
            mixed_op["returns"]["output_0"],
            7,
            mixed_op["returns"]["output_2"],
        ]

        expected = fn(x)
        got = fx_trace.replay(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @requires_gpus(1)
    def test_export_both_autograd_sequences(self) -> None:
        @custom_op("cudagraph_fx_test::export_composite")
        def composite(x: torch.Tensor) -> torch.Tensor:
            return x.sin() + x.cos()

        def step(x: torch.Tensor) -> tuple[torch.Tensor]:
            return torch.autograd.grad(composite(x).sum(), (x,))

        x = torch.ones(4, device="cuda", requires_grad=True)
        fx_trace = make_cudagraph_fx(step)(x)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            fx_trace.export_json(path)
            data = json.loads(path.read_text())
        for node in fx_trace.graph.nodes:
            if node.op != "call_function":
                continue
            autograd = node_autograd_meta(node)
            op = data["operators"][str(node_fx_id(node))]
            assert op["invoked_in"] == autograd["phase"]
            assert op["creating_seq_nrs"] == autograd["creating_seq_nrs"]
            assert op["executing_seq_nr"] == autograd["executing_seq_nr"]
            assert op["depends_on_gradient"] == autograd["depends_on_gradient"]
            assert op["forward_operators"] == autograd["forward_node_ids"]
        composite_op = next(
            op
            for op in data["operators"].values()
            if op["schema"].startswith("cudagraph_fx_test::export_composite(")
        )
        assert len(composite_op["creating_seq_nrs"]) >= 2
        backward = [
            op
            for op in data["operators"].values()
            if op.get("invoked_in") == "backward"
        ]
        assert any(op["depends_on_gradient"] is True for op in backward)
        assert any(op["forward_operators"] for op in backward)

        expected = step(x)
        got = fx_trace.replay(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @requires_gpus(1)
    def test_export_nonfinite_arguments_are_valid_json(self) -> None:
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.clamp(x, min=-float("inf"))

        x = torch.ones(4, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            fx_trace.export_json(str(path))
            data = json.loads(path.read_text())
        clamp = next(
            op
            for op in data["operators"].values()
            if op["schema"].startswith("aten::clamp")
        )
        assert clamp["arguments"]["min"] == "-inf"
        json.dumps(data, allow_nan=False)

        expected = fn(x)
        got = fx_trace.replay(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @requires_gpus(1)
    def test_export_nested_tensor_returns(self) -> None:
        @torch.library.custom_op("cudagraph_fx_test::export_nested", mutates_args=())
        def nested(x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
            return [x + 1, x + 2], x + 3

        def fn(x: torch.Tensor) -> tuple[object, ...]:
            return nested(x), None, 7

        x = torch.ones(4, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            fx_trace.export_json(path)
            data = json.loads(path.read_text())
        op_id, op = next(
            (key, op)
            for key, op in data["operators"].items()
            if op["schema"].startswith("cudagraph_fx_test::export_nested(")
        )
        tensor_list = op["returns"]["output_0"]
        assert len(tensor_list) == 2
        references = [*tensor_list, op["returns"]["output_1"]]
        assert len({reference["fx_node_id"] for reference in references}) == 3
        assert data["outputs"] == [*references, None, 7]
        for reference in references:
            assert data["tensors"][str(reference["fx_node_id"])][
                "fx_producer_id"
            ] == int(op_id)

        expected = fn(x)
        got = fx_trace.replay(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
