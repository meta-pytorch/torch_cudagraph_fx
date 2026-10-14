# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import expecttest
import pytest
import torch
import torch_cudagraph_fx._interpreter as interpreter
import torch_cudagraph_fx._tracer as tracer
import triton
import triton.language as tl
from torch._opaque_base import OpaqueBase
from torch.utils._pytree import tree_flatten
from torch_cudagraph_fx import (
    CudaGraphFxTrace,
    custom_op,
    make_cudagraph_fx,
    new_event,
    record_event,
    wait_event,
)
from torch_cudagraph_fx._cudagraph_annotations import (
    annotations_available,
    get_kernel_annotations,
    mark_kernels,
    maybe_get_kernel_py_stacks,
)
from torch_cudagraph_fx._metadata import (
    CUDAGRAPH_FX_METADATA_KEY,
    FX_NODE_ANNOTATION_KEY,
    FX_NODE_ID_KEY,
    MetadataManager,
    node_autograd_meta,
    node_fx_id,
    node_metadata,
    node_stack_trace,
    node_stream,
)
from torch_cudagraph_fx._tracer import annotated_capture
from torch_cudagraph_fx._validation import assert_trace_matches
from utils import requires_gpus


@triton.jit
def _triton_add_one_inplace(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr) -> None:
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(x_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + 1.0, mask=mask)


class TestCudaGraphFxGraph(expecttest.TestCase):
    @requires_gpus(1)
    def test_nodes_have_metadata(self) -> None:
        def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.max(x.sin(), dim=0)

        annotations: list[dict[str, Any]] = []
        original_mark_kernels = tracer.mark_kernels

        @contextmanager
        def recording_mark_kernels(annotation: dict[str, Any]):
            annotations.append(annotation.copy())
            with original_mark_kernels(annotation):
                yield

        x = torch.randn(8, 4, device="cuda")
        with patch.object(tracer, "mark_kernels", recording_mark_kernels):
            fx_trace = make_cudagraph_fx(fn)(x)

        nodes = list(fx_trace.graph.nodes)
        output = nodes[-1]
        assert output.op == "output"
        assert CUDAGRAPH_FX_METADATA_KEY not in output.meta
        metadata_nodes = nodes[:-1]
        metadata = [node.meta[CUDAGRAPH_FX_METADATA_KEY] for node in metadata_nodes]
        assert len({id(item) for item in metadata}) == len(metadata_nodes)
        assert [item[FX_NODE_ID_KEY] for item in metadata] == list(
            range(len(metadata_nodes))
        )

        for returned in output.args[0]:
            assert node_stream(returned) is None
            assert node_stack_trace(returned) is None

        assert annotations == [
            {
                FX_NODE_ANNOTATION_KEY: True,
                FX_NODE_ID_KEY: node.meta[CUDAGRAPH_FX_METADATA_KEY][FX_NODE_ID_KEY],
            }
            for node in metadata_nodes
            if isinstance(node.target, torch._ops.OpOverload)
        ]

        annotations.clear()
        with patch.object(interpreter, "mark_kernels", recording_mark_kernels):
            fx_trace.replay(x)
        assert annotations == [
            {
                FX_NODE_ANNOTATION_KEY: True,
                FX_NODE_ID_KEY: node.meta[CUDAGRAPH_FX_METADATA_KEY][FX_NODE_ID_KEY],
            }
            for node in metadata_nodes
            if node.op == "call_function"
        ]

        max_node = next(
            node for node in metadata_nodes if node.target == torch.ops.aten.max.dim
        )
        max_meta = max_node.meta[CUDAGRAPH_FX_METADATA_KEY]
        assert max_meta["execution"]["stream"] == 0
        assert isinstance(max_meta["execution"]["stack_trace"], str)
        assert max_meta["autograd"]["phase"] == "forward"
        for index, returned in enumerate(output.args[0]):
            assert returned.target is operator.getitem
            assert returned.args == (max_node, index)
            tensor_meta = returned.meta[CUDAGRAPH_FX_METADATA_KEY]["tensor"]
            assert tensor_meta["type"] == "forward_tensor"
            returned_autograd = returned.meta[CUDAGRAPH_FX_METADATA_KEY]["autograd"]
            assert returned_autograd["phase"] == max_meta["autograd"]["phase"]
            assert returned_autograd is not max_meta["autograd"]

    @requires_gpus(1)
    def test_backward_tensor_metadata(self) -> None:
        def step(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor]:
            loss = torch.relu(x @ w).max(dim=0).values.sum()
            return torch.autograd.grad(loss, (w,))

        x = torch.randn(8, 8, device="cuda", requires_grad=True)
        w = torch.randn(8, 8, device="cuda", requires_grad=True)
        nodes = list(make_cudagraph_fx(step)(x, w).graph.nodes)
        metadata_by_id = {
            node_fx_id(node): node_metadata(node)
            for node in nodes
            if node.op != "output"
        }
        backward = [
            metadata
            for metadata in metadata_by_id.values()
            if metadata.get("autograd", {}).get("phase") == "backward"
        ]

        assert backward
        gradient_node = nodes[-1].args[0][0]
        assert node_metadata(gradient_node)["tensor"]["type"] == "backward_tensor"

    @requires_gpus(1)
    def test_backward_metadata_links_to_forward_nodes(self) -> None:
        def step(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor]:
            loss = torch.relu(x @ w).max(dim=0).values.sum()
            return torch.autograd.grad(loss, (w,))

        x = torch.randn(8, 8, device="cuda", requires_grad=True)
        w = torch.randn(8, 8, device="cuda", requires_grad=True)
        nodes = list(make_cudagraph_fx(step)(x, w).graph.nodes)
        metadata_by_id = {
            node_fx_id(node): node_metadata(node)
            for node in nodes
            if node.op != "output"
        }
        projection_ids = {
            node_metadata(node)[FX_NODE_ID_KEY]
            for node in nodes
            if node.target is operator.getitem
        }
        assert projection_ids
        backward = [
            metadata
            for metadata in metadata_by_id.values()
            if metadata.get("autograd", {}).get("phase") == "backward"
        ]

        assert any(metadata["autograd"]["forward_node_ids"] for metadata in backward)
        for metadata in backward:
            for forward_id in metadata["autograd"]["forward_node_ids"]:
                assert forward_id not in projection_ids
                forward = metadata_by_id[forward_id]["autograd"]
                assert forward["phase"] == "forward"
                assert (
                    metadata["autograd"]["executing_seq_nr"]
                    in forward["creating_seq_nrs"]
                )

    @requires_gpus(1)
    def test_traces_aten_ops_with_stream_and_stack_meta(self) -> None:
        def fn(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor]:
            return (torch.relu(torch.mm(x, w)) + x,)

        x = torch.randn(64, 64, device="cuda")
        w = torch.randn(64, 64, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x, w)
        assert isinstance(fx_trace, CudaGraphFxTrace)

        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        # stream 0
        mm_default = torch.ops.aten.mm.default(arg_0, arg_1);  arg_1 = None
        # stream 0
        relu_default = torch.ops.aten.relu.default(mm_default);  mm_default = None
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(relu_default, arg_0);  relu_default = arg_0 = None
        return (add_tensor,)""",
        )

        for node in fx_trace.graph.nodes:
            if node.op == "call_function":
                assert node_stream(node) == 0
                stack_trace = node_stack_trace(node)
                assert isinstance(stack_trace, str) and stack_trace
                assert stack_trace == node.stack_trace

    @requires_gpus(1)
    def test_forward_and_backward_are_captured(self) -> None:
        x = torch.randn(32, 32, device="cuda", requires_grad=True)
        w = torch.randn(32, 32, device="cuda", requires_grad=True)

        def step(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor]:
            loss = torch.relu(x @ w).sum()
            (grad_w,) = torch.autograd.grad(loss, [w])
            return (grad_w,)

        fx_trace = make_cudagraph_fx(step)(x, w)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        # stream 0
        mm_default = torch.ops.aten.mm.default(arg_0, arg_1);  arg_1 = None
        # stream 0
        relu_default = torch.ops.aten.relu.default(mm_default);  mm_default = None
        # stream 0
        detach_default = torch.ops.aten.detach.default(relu_default)
        # stream 0
        sum_default = torch.ops.aten.sum.default(relu_default);  relu_default = None
        # stream 0
        ones_like_default = torch.ops.aten.ones_like.default(sum_default, pin_memory = False, memory_format = torch.preserve_format);  sum_default = None
        # stream 0
        expand_default = torch.ops.aten.expand.default(ones_like_default, [32, 32]);  ones_like_default = None
        # stream 0
        detach_default_1 = torch.ops.aten.detach.default(detach_default);  detach_default = None
        # stream 0
        threshold_backward_default = torch.ops.aten.threshold_backward.default(expand_default, detach_default_1, 0);  expand_default = detach_default_1 = None
        # stream 0
        t_default = torch.ops.aten.t.default(arg_0);  arg_0 = None
        # stream 0
        mm_default_1 = torch.ops.aten.mm.default(t_default, threshold_backward_default);  t_default = threshold_backward_default = None
        return (mm_default_1,)""",
        )

        (expected,) = step(x, w)
        (got,) = fx_trace.replay(x, w)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @requires_gpus(1)
    def test_replay_supports_pytree_inputs_and_outputs(self) -> None:
        def fn(
            inputs: dict[str, tuple[torch.Tensor, torch.Tensor]],
            *,
            scale: torch.Tensor,
        ) -> dict[str, object]:
            x, y = inputs["pair"]
            return {"sum": x + y * scale, "parts": (x, y)}

        x = torch.randn(128, device="cuda")
        y = torch.randn(128, device="cuda")
        scale = torch.randn((), device="cuda")
        inputs = {"pair": (x, y)}

        fx_trace = make_cudagraph_fx(fn)(inputs, scale=scale)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1, arg_2):
        # stream 0
        mul_tensor = torch.ops.aten.mul.Tensor(arg_1, arg_2);  arg_2 = None
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(arg_0, mul_tensor);  mul_tensor = None
        return (add_tensor, arg_0, arg_1)""",
        )

        _, in_spec = tree_flatten(((inputs,), {"scale": scale}))
        assert fx_trace.in_spec == in_spec

        expected = fn(inputs, scale=scale)
        got = fx_trace.replay(inputs, scale=scale)
        assert set(got) == set(expected)
        torch.testing.assert_close(got["sum"], expected["sum"], rtol=0, atol=0)
        assert got["parts"] == (x, y)

        with pytest.raises(RuntimeError, match="pytree structure"):
            fx_trace.replay({"pair": [x, y]}, scale=scale)

    @requires_gpus(1)
    def test_replay_supports_non_tensor_outputs(self) -> None:
        marker = object()

        def fn(x: torch.Tensor) -> dict[str, object]:
            return {
                "device": x.device,
                "dtype": x.dtype,
                "layout": x.layout,
                "memory_format": torch.contiguous_format,
                "complex": 1 + 2j,
                "marker": marker,
                "scale": 2.0,
                "tensor": x + 1.0,
            }

        x = torch.randn(128, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(arg_0, 1.0);  arg_0 = None
        _constant_0 = self._constant_0
        return (device(type='cuda', index=0), torch.float32, torch.strided, torch.contiguous_format, (1+2j), _constant_0, 2.0, add_tensor)""",
        )

        got = fx_trace.replay(x)
        assert got["device"] == x.device
        assert got["dtype"] == x.dtype
        assert got["layout"] == torch.strided
        assert got["memory_format"] == torch.contiguous_format
        assert got["complex"] == 1 + 2j
        assert got["marker"] is marker
        assert got["scale"] == 2.0
        torch.testing.assert_close(got["tensor"], x + 1.0, rtol=0, atol=0)

    @requires_gpus(1)
    def test_traces_complex_scalar_as_literal(self) -> None:
        def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
            return (x + (1 + 2j),)

        x = torch.randn(16, device="cuda", dtype=torch.complex64)
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(arg_0, (1+2j));  arg_0 = None
        return (add_tensor,)""",
        )

        (got,) = fx_trace.replay(x)
        torch.testing.assert_close(got, x + (1 + 2j), rtol=0, atol=0)

    @requires_gpus(1)
    def test_traces_layout_and_memory_format_as_literals(self) -> None:
        def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
            return (
                torch.ones_like(
                    x, layout=torch.strided, memory_format=torch.channels_last
                ),
            )

        x = torch.randn(2, 3, 4, 5, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        ones_like_default = torch.ops.aten.ones_like.default(arg_0, layout = torch.strided, pin_memory = False, memory_format = torch.channels_last);  arg_0 = None
        return (ones_like_default,)""",
        )

        (got,) = fx_trace.replay(x)
        torch.testing.assert_close(
            got, torch.ones(x.shape, device="cuda"), rtol=0, atol=0
        )
        assert got.layout == torch.strided
        assert got.is_contiguous(memory_format=torch.channels_last)

    @requires_gpus(1)
    def test_closed_over_cpu_tensor_is_lifted_by_reference(self) -> None:
        scale = torch.tensor(2.0)

        def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
            return (x * scale,)

        x = torch.ones(4, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        _constant_0 = self._constant_0
        # stream 0
        mul_tensor = torch.ops.aten.mul.Tensor(arg_0, _constant_0);  arg_0 = _constant_0 = None
        return (mul_tensor,)""",
        )
        scale.fill_(3.0)

        (got,) = fx_trace.replay(x)
        torch.testing.assert_close(got, x * scale, rtol=0, atol=0)

    @requires_gpus(1)
    def test_tensor_classification_survives_inplace_outputs(self) -> None:
        constant = torch.full((4,), 2.0, device="cuda")

        def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x.add_(2)
            constant.mul_(3)
            return x, constant, x + constant

        x = torch.ones(4, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        output = next(node for node in fx_trace.graph.nodes if node.op == "output")
        assert [
            node.meta[CUDAGRAPH_FX_METADATA_KEY]["tensor"]["type"]
            for node in output.args[0]
        ] == ["input", "constant", "forward_tensor"]

        replay_input = torch.ones_like(x)
        got_input, got_constant, got_sum = fx_trace.replay(replay_input)
        assert got_input is replay_input
        assert got_constant is constant
        torch.testing.assert_close(got_input, torch.full_like(x, 3), rtol=0, atol=0)
        torch.testing.assert_close(got_constant, torch.full_like(x, 6), rtol=0, atol=0)
        torch.testing.assert_close(got_sum, torch.full_like(x, 9), rtol=0, atol=0)

    @requires_gpus(1)
    def test_tensor_classification_propagates_across_views(self) -> None:
        constant = torch.full((8,), 2.0, device="cuda")
        weight = torch.nn.Parameter(torch.randn(8, device="cuda"))

        def fn(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return (
                x[1:5].view(2, 2).t(),
                constant[1:5].view(2, 2).t(),
                weight[1:5].view(2, 2).t(),
                x.clone()[1:5].view(2, 2).t(),
            )

        x = torch.randn(8, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        output = next(node for node in fx_trace.graph.nodes if node.op == "output")
        assert [
            node.meta[CUDAGRAPH_FX_METADATA_KEY]["tensor"]["type"]
            for node in output.args[0]
        ] == ["input", "constant", "weight", "forward_tensor"]

        replay_input = torch.randn_like(x)
        results = fx_trace.replay(replay_input)
        sources = (replay_input, constant, weight, replay_input)
        for result, source in zip(results, sources, strict=True):
            torch.testing.assert_close(result, source[1:5].view(2, 2).t())
        for result, source in zip(results[:3], sources[:3], strict=True):
            assert torch._C._storage_id(result) == torch._C._storage_id(source)
        assert torch._C._storage_id(results[3]) != torch._C._storage_id(replay_input)

    @requires_gpus(1)
    def test_captured_parameter_autograd_metadata(self) -> None:
        weight = torch.nn.Parameter(torch.randn(4, device="cuda"))

        def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
            return (x * weight,)

        x = torch.randn(4, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        _constant_0 = self._constant_0
        # stream 0
        mul_tensor = torch.ops.aten.mul.Tensor(arg_0, _constant_0);  arg_0 = _constant_0 = None
        return (mul_tensor,)""",
        )

        weight_node = next(
            node for node in fx_trace.graph.nodes if node.op == "get_attr"
        )
        tensor_meta = weight_node.meta[CUDAGRAPH_FX_METADATA_KEY]["tensor"]
        self.assertEqual(tensor_meta["type"], "weight")
        self.assertTrue(tensor_meta["requires_grad"])
        (got,) = fx_trace.replay(x)
        torch.testing.assert_close(got, x * weight, rtol=0, atol=0)

    @requires_gpus(1)
    def test_autograd_sequence_metadata(self) -> None:
        sequence_numbers: list[int] = []

        def step(x: torch.Tensor) -> tuple[torch.Tensor]:
            y = x.sin()
            assert y.grad_fn is not None
            sequence_numbers.append(y.grad_fn._sequence_nr())
            return torch.autograd.grad(y.sum(), (x,))

        x = torch.randn(8, device="cuda", requires_grad=True)
        nodes = list(make_cudagraph_fx(step)(x).graph.nodes)
        sin_node = next(
            node for node in nodes if node.target is torch.ops.aten.sin.default
        )
        cos_node = next(
            node for node in nodes if node.target is torch.ops.aten.cos.default
        )
        forward = node_metadata(sin_node)["autograd"]
        backward = node_metadata(cos_node)["autograd"]
        assert forward["phase"] == "forward"
        assert forward["executing_seq_nr"] is None
        assert sequence_numbers[0] in forward["creating_seq_nrs"]
        assert backward["phase"] == "backward"
        assert backward["executing_seq_nr"] == sequence_numbers[0]

    @requires_gpus(1)
    def test_autograd_sequence_metadata_one_op_no_sequences(self) -> None:
        def step(x: torch.Tensor) -> tuple[torch.Tensor]:
            with torch.no_grad():
                return (x.sin(),)

        x = torch.randn(8, device="cuda", requires_grad=True).sin()
        nodes = [
            node
            for node in make_cudagraph_fx(step)(x).graph.nodes
            if node.op == "call_function"
        ]
        assert len(nodes) == 1
        assert nodes[0].target is torch.ops.aten.sin.default
        autograd = node_metadata(nodes[0])["autograd"]
        assert autograd["phase"] == "forward"
        assert len(autograd["creating_seq_nrs"]) == 0
        assert autograd["executing_seq_nr"] is None

    @requires_gpus(1)
    def test_autograd_sequence_metadata_one_op_multiple_sequences(self) -> None:
        def step(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            torch._foreach_mul_([x, y], 2)
            return x, y

        # In-place foreach creates one backward node per non-leaf input.
        x = torch.randn(8, device="cuda", requires_grad=True) + 0
        y = torch.randn(8, device="cuda", requires_grad=True) + 0
        nodes = [
            node
            for node in make_cudagraph_fx(step)(x, y).graph.nodes
            if node.op == "call_function"
        ]
        assert len(nodes) == 1
        assert nodes[0].target is torch.ops.aten._foreach_mul_.Scalar
        assert x.grad_fn is not None and y.grad_fn is not None
        expected_seq_nrs = [x.grad_fn._sequence_nr(), y.grad_fn._sequence_nr()]
        assert len(set(expected_seq_nrs)) == 2
        autograd = node_metadata(nodes[0])["autograd"]
        assert autograd["phase"] == "forward"
        assert len(autograd["creating_seq_nrs"]) == 2
        assert autograd["creating_seq_nrs"] == sorted(expected_seq_nrs)
        assert autograd["executing_seq_nr"] is None

    @requires_gpus(1)
    def test_autograd_sequence_metadata_multiple_ops_one_sequence(self) -> None:
        class AddThenMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                return (x + 1) * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                return grad_output * 2

        sequence_numbers: list[int] = []

        def step(x: torch.Tensor) -> tuple[torch.Tensor]:
            y = AddThenMul.apply(x)
            assert y.grad_fn is not None
            sequence_numbers.append(y.grad_fn._sequence_nr())
            return (y,)

        x = torch.randn(8, device="cuda", requires_grad=True)
        nodes = [
            node
            for node in make_cudagraph_fx(step)(x).graph.nodes
            if node.op == "call_function"
        ]
        assert [node.target for node in nodes] == [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
        ]
        assert len(sequence_numbers) == 1
        for node in nodes:
            autograd = node_metadata(node)["autograd"]
            assert autograd["phase"] == "forward"
            assert len(autograd["creating_seq_nrs"]) == 1
            assert autograd["creating_seq_nrs"] == sequence_numbers
            assert autograd["executing_seq_nr"] is None

    @requires_gpus(1)
    def test_autograd_sequence_metadata_custom_op_projections(self) -> None:
        sequence_numbers: list[int] = []

        @custom_op("cudagraph_fx_test::sequence_pair")
        def pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = x.sin(), x.cos()
            for value in outputs:
                assert value.grad_fn is not None
                sequence_numbers.append(value.grad_fn._sequence_nr())
            return outputs

        x = torch.randn(8, device="cuda", requires_grad=True)
        nodes = [
            node
            for node in make_cudagraph_fx(pair)(x).graph.nodes
            if node.op == "call_function"
        ]
        assert len(nodes) == 3
        source, *projections = nodes
        autograd = node_metadata(source)["autograd"]
        assert len(set(sequence_numbers)) == 2
        assert autograd["creating_seq_nrs"] == sorted(sequence_numbers)
        assert autograd["executing_seq_nr"] is None
        for index, projection in enumerate(projections):
            assert projection.target is operator.getitem
            assert projection.args == (source, index)
            projection_autograd = node_metadata(projection)["autograd"]
            assert projection_autograd == autograd
            assert projection_autograd is not autograd

    @requires_gpus(1)
    def test_autograd_gradient_dependencies(self) -> None:
        class Sin(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                factor = x.cos()
                result = grad_output * factor
                return result.clone()

        def step(
            x: torch.Tensor, y: torch.Tensor, grad_output: torch.Tensor
        ) -> tuple[torch.Tensor]:
            return torch.autograd.grad(y, (x,), grad_output)

        # Autograd schedules backward on the stream used by forward.
        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            patch.object(torch.cuda.graph, "default_capture_stream", stream),
        ):
            x = torch.randn(8, device="cuda", requires_grad=True)
            y = Sin.apply(x)
            grad_output = torch.randn_like(y)
            fx_trace = make_cudagraph_fx(step)(x, y, grad_output)
        stream.synchronize()
        backward = {
            node.target: node_metadata(node)["autograd"]
            for node in fx_trace.graph.nodes
            if node.op == "call_function"
            and node_metadata(node).get("autograd", {}).get("phase") == "backward"
        }
        assert backward[torch.ops.aten.cos.default]["depends_on_gradient"] is False
        assert backward[torch.ops.aten.mul.Tensor]["depends_on_gradient"] is True
        assert backward[torch.ops.aten.clone.default]["depends_on_gradient"] is True

        replay_input = torch.randn_like(x)
        replay_grad_output = torch.randn_like(x)
        (gradient,) = fx_trace.replay(
            replay_input, replay_input.sin(), replay_grad_output
        )
        torch.testing.assert_close(
            gradient, replay_grad_output * replay_input.cos(), rtol=0, atol=0
        )

    @requires_gpus(1)
    def test_gradient_seed_does_not_change_earlier_dependencies(self) -> None:
        class ReturnSavedInput(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                x.cos()
                return x

        def step(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor]:
            return torch.autograd.grad(y, (x,))

        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            patch.object(torch.cuda.graph, "default_capture_stream", stream),
        ):
            x = torch.randn(8, device="cuda", requires_grad=True)
            y = ReturnSavedInput.apply(x * 2).sum()
            fx_trace = make_cudagraph_fx(step)(x, y)
        stream.synchronize()
        backward = {
            node.target: node_metadata(node)["autograd"]
            for node in fx_trace.graph.nodes
            if node.op == "call_function"
            and node_metadata(node)["autograd"]["phase"] == "backward"
        }
        assert backward[torch.ops.aten.expand.default]["depends_on_gradient"] is True
        assert backward[torch.ops.aten.cos.default]["depends_on_gradient"] is False
        assert backward[torch.ops.aten.mul.Tensor]["depends_on_gradient"] is True

        (gradient,) = fx_trace.replay(x, y)
        torch.testing.assert_close(gradient, x * 4, rtol=0, atol=0)

    @requires_gpus(1)
    def test_closed_over_cpu_tensor_is_lifted_once(self) -> None:
        scale = torch.tensor(2.0)

        def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
            return (x * scale + scale,)

        x = torch.ones(4, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        _constant_0 = self._constant_0
        # stream 0
        mul_tensor = torch.ops.aten.mul.Tensor(arg_0, _constant_0);  arg_0 = None
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, _constant_0);  mul_tensor = _constant_0 = None
        return (add_tensor,)""",
        )

        (got,) = fx_trace.replay(x)
        torch.testing.assert_close(got, x * scale + scale, rtol=0, atol=0)

    @requires_gpus(1)
    def test_captured_objects_are_interned_by_identity(self) -> None:
        first = SimpleNamespace(value=1)
        second = SimpleNamespace(value=1)

        def fn(x: torch.Tensor) -> tuple[object, ...]:
            return x + 1, first, first, second

        x = torch.ones(4, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(arg_0, 1);  arg_0 = None
        _constant_0 = self._constant_0
        _constant_1 = self._constant_1
        return (add_tensor, _constant_0, _constant_0, _constant_1)""",
        )

        first.value = 2
        got, first_result, repeated_result, second_result = fx_trace.replay(x)
        torch.testing.assert_close(got, x + 1, rtol=0, atol=0)
        self.assertIs(first_result, first)
        self.assertIs(repeated_result, first)
        self.assertIs(second_result, second)
        self.assertEqual(first_result.value, 2)
        self.assertEqual(second_result.value, 1)

    @requires_gpus(1)
    def test_mixed_captured_constants_have_unique_names(self) -> None:
        scale = torch.tensor(2.0)
        marker = SimpleNamespace(value=1)
        opaque = OpaqueBase()

        def fn(x: torch.Tensor) -> tuple[object, ...]:
            return x * scale, marker, opaque, scale, marker, opaque

        x = torch.ones(4, device="cuda")
        fx_trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        _constant_0 = self._constant_0
        # stream 0
        mul_tensor = torch.ops.aten.mul.Tensor(arg_0, _constant_0);  arg_0 = None
        _constant_1 = self._constant_1
        _constant_2 = self._constant_2
        return (mul_tensor, _constant_1, _constant_2, _constant_0, _constant_1, _constant_2)""",
        )

        got = fx_trace.replay(x)
        torch.testing.assert_close(got[0], x * scale, rtol=0, atol=0)
        self.assertIs(got[1], marker)
        self.assertIs(got[2], opaque)
        self.assertIs(got[3], scale)
        self.assertIs(got[4], marker)
        self.assertIs(got[5], opaque)


@requires_gpus(1)
def test_warns_when_capture_completeness_check_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "torch_cudagraph_fx._tracer.annotations_available",
        lambda: False,
    )
    x = torch.randn(16, device="cuda")

    with pytest.warns(
        RuntimeWarning,
        match=r"kernel annotations are not available.*might not capture all kernel",
    ):
        make_cudagraph_fx(lambda value: (value.sin(),))(x)


@requires_gpus(1)
def test_raw_triton_kernel_under_plain_annotation_is_still_uncaptured() -> None:
    def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
        with mark_kernels({"name": "plain_raw_triton_scope"}):
            _triton_add_one_inplace[(1,)](x, x.numel(), BLOCK_SIZE=32)
        return (x,)

    x = torch.zeros(32, device="cuda")
    fn(x)
    torch.cuda.synchronize()
    x.zero_()
    torch.cuda.synchronize()

    # A plain mark_kernels annotation is not the reserved fx-node key, so the
    # launch still counts as uncaptured.
    if annotations_available():
        with pytest.raises(RuntimeError, match=r"missing from the trace"):
            make_cudagraph_fx(fn)(x)
    else:
        fx_trace = make_cudagraph_fx(fn)(x)
        expecttest.TestCase().assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        return (arg_0,)""",
        )
        assert not any(node.op == "call_function" for node in fx_trace.graph.nodes)


@requires_gpus(1)
@pytest.mark.skipif(
    not annotations_available(),
    reason="needs kernel annotations + cudaGraphNodeGetToolsId "
    "(run with LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat)",
)
def test_trace_preserves_other_graph_annotations() -> None:
    x = torch.zeros(32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with annotated_capture(graph):
        with mark_kernels({"name": "other_graph"}):
            x.add_(1)

    other_graph_ids = {
        tools_id
        for tools_id, ann_list in get_kernel_annotations().items()
        if any(annotation.get("name") == "other_graph" for annotation in ann_list)
    }
    assert other_graph_ids, "expected annotations for the other graph"

    make_cudagraph_fx(lambda value: (value.sin(),))(x)

    assert other_graph_ids.issubset(get_kernel_annotations())


@requires_gpus(1)
def test_raw_memset_is_flagged_uncaptured() -> None:
    rt = pytest.importorskip("cuda.bindings.runtime")

    def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
        stream = rt.cudaStream_t(init_value=torch.cuda.current_stream().cuda_stream)
        (err,) = rt.cudaMemsetAsync(
            x.data_ptr(), 0, x.numel() * x.element_size(), stream
        )
        assert err == rt.cudaError_t.cudaSuccess
        return (x,)

    x = torch.ones(64, device="cuda")
    fn(x)
    torch.cuda.synchronize()

    if annotations_available():
        with pytest.raises(RuntimeError, match=r"memset"):
            make_cudagraph_fx(fn)(x)
    else:
        fx_trace = make_cudagraph_fx(fn)(x)
        expecttest.TestCase().assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        return (arg_0,)""",
        )
        assert not any(node.op == "call_function" for node in fx_trace.graph.nodes)


@requires_gpus(1)
def test_raw_memcpy_is_flagged_uncaptured() -> None:
    rt = pytest.importorskip("cuda.bindings.runtime")

    def fn(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor]:
        stream = rt.cudaStream_t(init_value=torch.cuda.current_stream().cuda_stream)
        (err,) = rt.cudaMemcpyAsync(
            y.data_ptr(),
            x.data_ptr(),
            x.numel() * x.element_size(),
            rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            stream,
        )
        assert err == rt.cudaError_t.cudaSuccess
        return (y,)

    x = torch.ones(64, device="cuda")
    y = torch.zeros(64, device="cuda")
    fn(x, y)
    torch.cuda.synchronize()

    if annotations_available():
        with pytest.raises(RuntimeError, match=r"memcpy"):
            make_cudagraph_fx(fn)(x, y)
    else:
        fx_trace = make_cudagraph_fx(fn)(x, y)
        expecttest.TestCase().assertExpectedInline(
            fx_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        return (arg_1,)""",
        )
        assert not any(node.op == "call_function" for node in fx_trace.graph.nodes)


@requires_gpus(1)
@pytest.mark.skipif(
    not annotations_available(),
    reason="completeness check needs cudaGraphNodeGetToolsId "
    "(run with LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat)",
)
def test_uncaptured_kernel_raise_names_the_launch() -> None:
    @triton.jit
    def _add_one(x_ptr, out_ptr, n, block: tl.constexpr):
        offs = tl.program_id(0) * block + tl.arange(0, block)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)

    n, block = 256, 64
    grid = (triton.cdiv(n, block),)
    x = torch.randn(n, device="cuda")
    _add_one[grid](x, torch.empty_like(x), n, block)
    torch.cuda.synchronize()

    def fn(x: torch.Tensor) -> tuple[torch.Tensor]:
        out = torch.empty_like(x)
        _add_one[grid](x, out, n, block)
        return (out,)

    with pytest.raises(
        RuntimeError,
        match=r"1 kernel launch\(es\) had no fx node.*missing from the trace.*_add_one",
    ):
        make_cudagraph_fx(fn)(x)


@requires_gpus(1)
@pytest.mark.skipif(
    maybe_get_kernel_py_stacks() is None or not annotations_available(),
    reason="needs native PyTorch Python stack annotations",
)
def test_uncaptured_kernel_reports_python_launch_stack() -> None:
    def launch_raw_triton(x: torch.Tensor) -> tuple[torch.Tensor]:
        _triton_add_one_inplace[(1,)](x, x.numel(), BLOCK_SIZE=32)
        return (x,)

    x = torch.zeros(32, device="cuda")
    launch_raw_triton(x)
    torch.cuda.synchronize()

    with (
        torch.autograd.grad_mode.set_multithreading_enabled(False),
        pytest.raises(RuntimeError, match=r"test_tracer\.py:\d+:launch_raw_triton"),
    ):
        make_cudagraph_fx(launch_raw_triton)(x)


@requires_gpus(1)
def test_validation_criterion_across_stream_assignments() -> None:
    def single(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.relu(torch.mm(x, w)) + x,)

    def multi(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor]:
        side = torch.cuda.Stream()
        a = torch.mm(x, w)
        ready = new_event()
        record_event(ready)
        with torch.cuda.stream(side):
            wait_event(ready)
            b = torch.relu(a)
            done = new_event()
            record_event(done)
        wait_event(done)
        return (b + x,)

    x = torch.randn(256, 256, device="cuda")
    w = torch.randn(256, 256, device="cuda")

    single_trace = make_cudagraph_fx(single)(x, w)
    expecttest.TestCase().assertExpectedInline(
        single_trace.print_readable(print_output=False),
        """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        # stream 0
        mm_default = torch.ops.aten.mm.default(arg_0, arg_1);  arg_1 = None
        # stream 0
        relu_default = torch.ops.aten.relu.default(mm_default);  mm_default = None
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(relu_default, arg_0);  relu_default = arg_0 = None
        return (add_tensor,)""",
    )

    multi_trace = make_cudagraph_fx(multi)(x, w)
    expecttest.TestCase().assertExpectedInline(
        multi_trace.print_readable(print_output=False),
        """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        # stream 0
        mm_default = torch.ops.aten.mm.default(arg_0, arg_1);  arg_1 = None
        # stream 0
        new_event_default = torch.ops.torch_cudagraph_fx.new_event.default()
        # stream 0
        record_event_default = torch.ops.torch_cudagraph_fx.record_event.default(new_event_default);  record_event_default = None
        # stream 1
        wait_event_default = torch.ops.torch_cudagraph_fx.wait_event.default(new_event_default);  new_event_default = wait_event_default = None
        # stream 1
        relu_default = torch.ops.aten.relu.default(mm_default);  mm_default = None
        # stream 1
        new_event_default_1 = torch.ops.torch_cudagraph_fx.new_event.default()
        # stream 1
        record_event_default_1 = torch.ops.torch_cudagraph_fx.record_event.default(new_event_default_1);  record_event_default_1 = None
        # stream 0
        wait_event_default_1 = torch.ops.torch_cudagraph_fx.wait_event.default(new_event_default_1);  new_event_default_1 = wait_event_default_1 = None
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(relu_default, arg_0);  relu_default = arg_0 = None
        return (add_tensor,)""",
    )

    for fn, min_streams, fx_trace in (
        (single, 1, single_trace),
        (multi, 2, multi_trace),
    ):
        interp = fx_trace._interpreter()
        seq = assert_trace_matches(
            lambda fn=fn: fn(x, w), lambda interp=interp: interp.run_fn(x, w)
        )
        assert len(seq) >= 3
        assert len({stream for _, stream in seq}) >= min_streams


@requires_gpus(1)
def test_validation_criterion_raises_on_trace_stream_mismatch() -> None:
    side = torch.cuda.Stream()
    x = torch.randn(4096, device="cuda")

    def reference(x: torch.Tensor) -> tuple[torch.Tensor]:
        y = x + 1.0
        return (y * 2.0,)

    def candidate(x: torch.Tensor) -> tuple[torch.Tensor]:
        y = x + 1.0
        ready = new_event()
        record_event(ready)
        with torch.cuda.stream(side):
            wait_event(ready)
            out = y * 2.0
            done = new_event()
            record_event(done)
        wait_event(done)
        return (out,)

    reference_interp = make_cudagraph_fx(reference)(x)._interpreter()
    candidate_interp = make_cudagraph_fx(candidate)(x)._interpreter()

    with pytest.raises(
        AssertionError,
        match="trace does not satisfy the validation criterion",
    ) as exc_info:
        assert_trace_matches(
            lambda: reference_interp.run_fn(x),
            lambda: candidate_interp.run_fn(x),
        )

    message = str(exc_info.value)
    assert "reference" in message
    assert "candidate" in message


@requires_gpus(1)
@pytest.mark.skipif(
    not annotations_available(),
    reason="needs kernel annotations + cudaGraphNodeGetToolsId "
    "(run with LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat)",
)
def test_mark_kernels_links_cuda_graph_to_fx_metadata() -> None:
    def fn(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.relu(torch.mm(x, w)) + x,)

    x = torch.randn(32, 32, device="cuda")
    w = torch.randn(32, 32, device="cuda")
    fx_trace = make_cudagraph_fx(fn)(x, w)
    expecttest.TestCase().assertExpectedInline(
        fx_trace.print_readable(print_output=False),
        """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        # stream 0
        mm_default = torch.ops.aten.mm.default(arg_0, arg_1);  arg_1 = None
        # stream 0
        relu_default = torch.ops.aten.relu.default(mm_default);  mm_default = None
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(relu_default, arg_0);  relu_default = arg_0 = None
        return (add_tensor,)""",
    )
    interp = fx_trace._interpreter()

    existing_tools_ids = set(get_kernel_annotations())
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        interp.run_fn(x, w)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with annotated_capture(graph, stream=warmup):
        interp.run_fn(x, w)

    new_annotations = [
        ann_list
        for tools_id, ann_list in get_kernel_annotations().items()
        if tools_id not in existing_tools_ids
    ]
    assert new_annotations, "expected at least one annotated kernel"
    metadata_by_id = {
        node.meta[CUDAGRAPH_FX_METADATA_KEY][FX_NODE_ID_KEY]: node.meta[
            CUDAGRAPH_FX_METADATA_KEY
        ]
        for node in fx_trace.graph.nodes
        if node.op == "call_function"
    }
    annotated_node_ids = set()
    for ann_list in new_annotations:
        annotation = ann_list[0]
        assert annotation[FX_NODE_ANNOTATION_KEY] is True
        node_id = annotation[FX_NODE_ID_KEY]
        assert isinstance(node_id, int)
        annotated_node_ids.add(node_id)
        stack_trace = metadata_by_id[node_id]["execution"]["stack_trace"]
        assert isinstance(stack_trace, str) and stack_trace
    assert annotated_node_ids == set(metadata_by_id)


@requires_gpus(1)
@pytest.mark.parametrize("entrypoint", ["grad", "backward", "tensor_backward", "edge"])
def test_gradient_dependencies_for_graph_built_during_trace(entrypoint: str) -> None:
    def step(x: torch.Tensor, seed: torch.Tensor) -> tuple[torch.Tensor]:
        y = x.sin()
        if entrypoint == "grad":
            return torch.autograd.grad(y, x, seed)
        if entrypoint == "edge":
            return torch.autograd.grad(
                torch.autograd.graph.get_gradient_edge(y), x, seed
            )
        if entrypoint == "backward":
            torch.autograd.backward(y, seed)
        else:
            y.backward(seed)
        assert x.grad is not None
        return (x.grad,)

    x = torch.randn(8, device="cuda", requires_grad=True)
    seed = torch.randn_like(x)
    trace = make_cudagraph_fx(step)(x, seed)
    backward = {
        node.target: node_metadata(node)["autograd"]["depends_on_gradient"]
        for node in trace.graph.nodes
        if node.op == "call_function"
        and node_metadata(node)["autograd"]["phase"] == "backward"
    }
    assert backward[torch.ops.aten.cos.default] is False
    assert backward[torch.ops.aten.mul.Tensor] is True
    (gradient,) = trace.replay(x, seed)
    torch.testing.assert_close(gradient, seed * x.cos(), rtol=0, atol=0)


@requires_gpus(1)
def test_gradient_dependencies_for_recomputation_in_backward() -> None:
    class Recompute(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(x)
            return x.sin()

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            (x,) = ctx.saved_tensors
            with torch.enable_grad():
                recomputed = x.detach().requires_grad_()
                y = recomputed.sin()
                seed = torch.full_like(y, 2)
                (gradient,) = torch.autograd.grad(y, recomputed, seed)
            return gradient * grad_output

    def step(x: torch.Tensor, seed: torch.Tensor) -> tuple[torch.Tensor]:
        return torch.autograd.grad(Recompute.apply(x), x, seed)

    x = torch.randn(8, device="cuda", requires_grad=True)
    seed = torch.randn_like(x)
    trace = make_cudagraph_fx(step)(x, seed)
    backward = [
        (node.target, node_metadata(node)["autograd"]["depends_on_gradient"])
        for node in trace.graph.nodes
        if node.op == "call_function"
        and node_metadata(node)["autograd"]["phase"] == "backward"
    ]
    assert (torch.ops.aten.sin.default, False) in backward
    assert (torch.ops.aten.full_like.default, False) in backward
    assert (torch.ops.aten.cos.default, False) in backward
    assert [depends for op, depends in backward if op == torch.ops.aten.mul.Tensor] == [
        True,
        True,
    ]
    (gradient,) = trace.replay(x, seed)
    torch.testing.assert_close(gradient, 2 * x.cos() * seed, rtol=0, atol=0)


def test_autograd_hooks_are_restored_after_backward_error() -> None:
    class BrokenBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
            return x.sin()

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("backward failed")

    original_run_backward = torch.autograd._engine_run_backward
    original_graph_run_backward = torch.autograd.graph._engine_run_backward
    metadata = MetadataManager(torch.fx.Graph(), [])
    x = torch.randn(8, requires_grad=True)
    y = BrokenBackward.apply(x)
    with pytest.raises(RuntimeError, match="backward failed"):
        with metadata.enable_autograd_hooks():
            torch.autograd.grad(y, x, torch.ones_like(y))
    assert torch.autograd._engine_run_backward is original_run_backward
    assert torch.autograd.graph._engine_run_backward is original_graph_run_backward
    assert not metadata._gradient_dependency_metadata._autograd_hooks


class _Sin(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.sin().clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output * ctx.saved_tensors[0].cos()


class _Recompute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.fn = fn
        return fn(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        with torch.enable_grad():
            x = ctx.saved_tensors[0].detach().requires_grad_()
            y = ctx.fn(x)
            (gradient,) = torch.autograd.grad(y, (x,), grad_output)
        return gradient, None


@requires_gpus(1)
@pytest.mark.parametrize("fn, expected_nodes", [(torch.sin, 1), (_Sin.apply, 2)])
def test_recomputation_records_forward_sequences(
    fn: Callable[[torch.Tensor], torch.Tensor], expected_nodes: int
) -> None:
    outer_seq_nr = -1

    def step(x: torch.Tensor) -> tuple[torch.Tensor]:
        nonlocal outer_seq_nr
        y = _Recompute.apply(x, fn)
        assert y.grad_fn is not None
        outer_seq_nr = y.grad_fn._sequence_nr()
        return torch.autograd.grad(y, (x,), torch.ones_like(y))

    x = torch.linspace(-1, 1, 8, device="cuda", requires_grad=True)
    trace = make_cudagraph_fx(step)(x)
    (gradient,) = trace.replay(x)
    torch.testing.assert_close(gradient, x.cos(), rtol=0, atol=0)
    recomputed = [
        node_metadata(node)["autograd"]
        for node in trace.graph.nodes
        if node.target in {torch.ops.aten.sin.default, torch.ops.aten.clone.default}
        and node_metadata(node)["autograd"]["phase"] == "backward"
    ]
    inner_backward = [
        node_metadata(node)["autograd"]
        for node in trace.graph.nodes
        if node.target in {torch.ops.aten.cos.default, torch.ops.aten.mul.Tensor}
    ]
    assert len(recomputed) == expected_nodes
    assert len(inner_backward) == 2
    inner_seq_nr = inner_backward[0]["executing_seq_nr"]
    assert inner_seq_nr != outer_seq_nr
    for metadata in recomputed:
        assert metadata["executing_seq_nr"] == outer_seq_nr
        assert metadata["creating_seq_nrs"] == [inner_seq_nr]
    for metadata in inner_backward:
        assert metadata["executing_seq_nr"] == inner_seq_nr
        assert metadata["creating_seq_nrs"] == []


@requires_gpus(1)
def test_sequence_reuse_stops_between_backward_runs() -> None:
    def step(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x.sin()
        grad_output = torch.ones_like(y)
        (first_gradient,) = torch.autograd.grad(
            y, (x,), grad_output, create_graph=True, retain_graph=True
        )
        (second_gradient,) = torch.autograd.grad(y, (x,), grad_output)
        return first_gradient, second_gradient

    x = torch.linspace(-1, 1, 8, device="cuda", requires_grad=True)
    trace = make_cudagraph_fx(step)(x)
    backward = [
        node
        for node in trace.graph.nodes
        if node.op == "call_function"
        and node_metadata(node)["autograd"]["phase"] == "backward"
    ]
    assert len(backward) == 4
    first_run = backward[:2]
    second_run = backward[2:]
    assert any(
        node_metadata(node)["autograd"]["creating_seq_nrs"] for node in first_run
    )
    for node in second_run:
        assert node_metadata(node)["autograd"]["creating_seq_nrs"] == []
    first_gradient, second_gradient = trace.replay(x)
    torch.testing.assert_close(first_gradient, x.cos(), rtol=0, atol=0)
    torch.testing.assert_close(second_gradient, x.cos(), rtol=0, atol=0)


@requires_gpus(1)
@pytest.mark.parametrize("fn, expected_nodes", [(torch.sin, 1), (_Sin.apply, 2)])
def test_nested_backward_links_to_recomputed_forward(
    fn: Callable[[torch.Tensor], torch.Tensor], expected_nodes: int
) -> None:
    def step(x: torch.Tensor) -> tuple[torch.Tensor]:
        y = _Recompute.apply(x, fn)
        return torch.autograd.grad(y, (x,), torch.ones_like(y))

    x = torch.linspace(-1, 1, 8, device="cuda", requires_grad=True)
    trace = make_cudagraph_fx(step)(x)
    (gradient,) = trace.replay(x)
    torch.testing.assert_close(gradient, x.cos(), rtol=0, atol=0)
    recomputed = [
        node
        for node in trace.graph.nodes
        if node.target in {torch.ops.aten.sin.default, torch.ops.aten.clone.default}
        and node_metadata(node)["autograd"]["phase"] == "backward"
    ]
    inner_backward = [
        node
        for node in trace.graph.nodes
        if node.target in {torch.ops.aten.cos.default, torch.ops.aten.mul.Tensor}
    ]
    assert len(recomputed) == expected_nodes
    assert len(inner_backward) == 2
    for node in inner_backward:
        assert node_metadata(node)["autograd"]["forward_node_ids"] == [
            node_fx_id(source) for source in recomputed
        ]


@requires_gpus(1)
def test_nested_backward_links_to_each_recomputed_operator() -> None:
    def step(x: torch.Tensor) -> tuple[torch.Tensor]:
        y = _Recompute.apply(x, lambda value: value.sin().sin())
        return torch.autograd.grad(y, (x,), torch.ones_like(y))

    x = torch.linspace(-1, 1, 8, device="cuda", requires_grad=True)
    trace = make_cudagraph_fx(step)(x)
    (gradient,) = trace.replay(x)
    torch.testing.assert_close(gradient, x.sin().cos() * x.cos(), rtol=0, atol=0)
    recomputed = [
        node
        for node in trace.graph.nodes
        if node.target is torch.ops.aten.sin.default
        and node_autograd_meta(node)["phase"] == "backward"
    ]
    assert len(recomputed) == 2
    first_recomputed_sin, second_recomputed_sin = recomputed
    inner_backward = [
        node
        for node in trace.graph.nodes
        if node.target in {torch.ops.aten.cos.default, torch.ops.aten.mul.Tensor}
    ]
    assert [node.target for node in inner_backward] == [
        torch.ops.aten.cos.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.cos.default,
        torch.ops.aten.mul.Tensor,
    ]
    assert [
        node_autograd_meta(node)["forward_node_ids"] for node in inner_backward
    ] == [
        [node_fx_id(second_recomputed_sin)],
        [node_fx_id(second_recomputed_sin)],
        [node_fx_id(first_recomputed_sin)],
        [node_fx_id(first_recomputed_sin)],
    ]


@requires_gpus(1)
def test_double_backward_links_to_first_backward_operators() -> None:
    def step(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x.sin().sum()
        (gradient,) = torch.autograd.grad(y, (x,), create_graph=True)
        (second_gradient,) = torch.autograd.grad(gradient.sum(), (x,))
        return gradient, second_gradient

    x = torch.linspace(-1, 1, 8, device="cuda", requires_grad=True)
    trace = make_cudagraph_fx(step)(x)
    nodes = list(trace.graph.nodes)
    gradient_node = nodes[-1].args[0][0]
    first_backward_seq_nr = node_metadata(gradient_node)["autograd"]["executing_seq_nr"]
    first_backward = [
        node
        for node in nodes
        if node.target in {torch.ops.aten.cos.default, torch.ops.aten.mul.Tensor}
        and node_metadata(node)["autograd"]["executing_seq_nr"] == first_backward_seq_nr
    ]
    gradient, second_gradient = trace.replay(x)
    torch.testing.assert_close(gradient, x.cos(), rtol=0, atol=0)
    torch.testing.assert_close(second_gradient, -x.sin(), rtol=0, atol=0)
    assert len(first_backward) == 2
    for source in first_backward:
        sequences = node_metadata(source)["autograd"]["creating_seq_nrs"]
        assert len(sequences) == 1
        second_backward = [
            node
            for node in nodes
            if node.op == "call_function"
            and node_metadata(node)["autograd"]["phase"] == "backward"
            and node_metadata(node)["autograd"]["executing_seq_nr"] == sequences[0]
        ]
        assert second_backward
        for node in second_backward:
            assert node_metadata(node)["autograd"]["forward_node_ids"] == [
                node_fx_id(source)
            ]


@requires_gpus(1)
def test_retained_capture_uses_caller_owned_events() -> None:
    event = new_event(external=True)

    def fn(x: torch.Tensor) -> torch.Tensor:
        value = x + 1
        record_event(event)
        return value

    x = torch.ones(4, device="cuda")
    trace = make_cudagraph_fx(fn, retain_cudagraph=True)(x)
    cudagraph_state = trace.cudagraph_states()
    try:
        assert cudagraph_state is not None
        cudagraph_state.cuda_graph.replay()
        torch.testing.assert_close(
            cudagraph_state.flat_outputs[0], x + 1, rtol=0, atol=0
        )
    finally:
        torch.cuda.synchronize()
        trace.release_cudagraph_states()


@requires_gpus(1)
def test_retain_cudagraph_mutates_pytree_inputs() -> None:
    def fn(
        inputs: list[torch.Tensor], *, bias: dict[str, torch.Tensor]
    ) -> dict[str, tuple[list[torch.Tensor], None]]:
        inputs[0].add_(bias["value"])
        return {"result": ([inputs[0], inputs[0][1:], inputs[0] * 3], None)}

    inputs = [torch.ones(4, device="cuda")]
    bias = {"value": torch.full((4,), 2.0, device="cuda")}
    trace = make_cudagraph_fx(fn, retain_cudagraph=True)(inputs, bias=bias)
    cudagraph_state = trace.cudagraph_states()
    try:
        assert cudagraph_state is not None
        torch.testing.assert_close(
            inputs[0], torch.ones_like(inputs[0]), rtol=0, atol=0
        )
        assert cudagraph_state.flat_inputs[0] is inputs[0]
        assert cudagraph_state.flat_inputs[1] is bias["value"]
        assert cudagraph_state.flat_outputs[0] is inputs[0]
        assert cudagraph_state.flat_outputs[1].data_ptr() == inputs[0][1:].data_ptr()
        for input_value, bias_value in [(1, 2), (4, 5)]:
            inputs[0].fill_(input_value)
            bias["value"].fill_(bias_value)
            expected_inputs = [inputs[0].clone()]
            expected_outputs, _ = tree_flatten(fn(expected_inputs, bias=bias))
            cudagraph_state.cuda_graph.replay()
            torch.testing.assert_close(inputs, expected_inputs, rtol=0, atol=0)
            torch.testing.assert_close(
                cudagraph_state.flat_outputs, expected_outputs, rtol=0, atol=0
            )
    finally:
        torch.cuda.synchronize()
        trace.release_cudagraph_states()
    assert trace.cudagraph_states() is None
