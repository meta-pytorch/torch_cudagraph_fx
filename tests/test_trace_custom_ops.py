# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any
from unittest.mock import patch

import expecttest
import pytest
import torch
import torch.fx as fx
import torch.utils._pytree as pytree
import triton
import triton.language as tl
from torch._library.opaque_object import is_opaque_type, register_opaque_type
from torch._opaque_base import OpaqueBase
from torch.utils._python_dispatch import TorchDispatchMode
from torch_cudagraph_fx import (
    custom_op,
    make_cudagraph_fx,
    OpaqueRef,
    register_dataclass_pytree,
    trace_dataclass,
    trace_leaf,
)
from torch_cudagraph_fx._cudagraph_annotations import annotations_available
from torch_cudagraph_fx._metadata import node_fx_id, node_metadata, node_tensor_meta
from torch_cudagraph_fx._tracer import _CudagraphFxDispatchMode
from torch_cudagraph_fx.custom_ops.arg_path import parse_arg_path
from torch_cudagraph_fx.custom_ops.schema import (
    decode_overload_name,
    encode_overload_name,
)
from utils import requires_gpus


@dataclass
class Box:
    tensor: torch.Tensor


pytree.register_dataclass(
    Box,
    serialized_type_name="torch_cudagraph_fx.tests.test_trace_custom_ops.Box",
)


@dataclass(frozen=True)
class Buffer:
    tensor: torch.Tensor
    handle: object
    snapshot: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot", self.tensor.clone())


register_dataclass_pytree(Buffer, opaque_ref_fields=("handle",))


class Scale(OpaqueBase):
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Scale) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __fx_repr__(self) -> tuple[str, dict[str, type[Scale]]]:
        return f"Scale({self.value!r})", {"Scale": Scale}


if not is_opaque_type(Scale):
    register_opaque_type(Scale, typ="value")


class Reference(OpaqueBase):
    pass


if not is_opaque_type(Reference):
    register_opaque_type(Reference, typ="reference")


class ScaleFactor(IntEnum):
    DOUBLE = 2
    TRIPLE = 3


class PlainScaleFactor(Enum):
    DOUBLE = 2
    TRIPLE = 3


@triton.jit
def _add_one_inplace(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr) -> None:
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(x_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + 1.0, mask=mask)


def _overload_name(op: Any) -> str:
    overload = next(iter(op._cache.values()))
    name = overload.__dict__.get("_custom_op_overload_name")
    assert isinstance(name, str)
    return name


def test_arg_path_rejects_dynamic_accesses() -> None:
    values = {"state": {"buf": [1, 2]}, "box": type("Holder", (), {"x": 3})()}
    assert parse_arg_path("state['buf'][1]").resolve(values) == 2
    assert parse_arg_path("box.x").resolve(values) == 3
    for path in ("state[key]", "state[1:2]", "fn()"):
        with pytest.raises(ValueError, match="unsupported mutation path"):
            parse_arg_path(path)


class TestTraceCustomOps(expecttest.TestCase):
    maxDiff = None

    def test_trace_leaf_disabled_preserves_function(self) -> None:
        for tracing in (None, "0"):
            with self.subTest(tracing=tracing), patch.dict(os.environ):
                if tracing is None:
                    os.environ.pop("CGFX_TRACING", None)
                else:
                    os.environ["CGFX_TRACING"] = tracing

                def fn(x: int) -> int:
                    return x + 1

                decorated = trace_leaf(
                    "cudagraph_fx_test::disabled", mutates_args=("x",)
                )(fn)
                self.assertIs(decorated, fn)
                self.assertEqual(decorated(2), 3)
                os.environ["CGFX_TRACING"] = "1"
                self.assertEqual(decorated(3), 4)
                self.assertFalse(hasattr(torch.ops.cudagraph_fx_test, "disabled"))

    @requires_gpus(1)
    def test_trace_leaf_enabled_traces_and_replays(self) -> None:
        with patch.dict(os.environ, CGFX_TRACING="1"):

            @trace_leaf("cudagraph_fx_test::enabled", mutates_args=("x",))
            def fn(x: torch.Tensor) -> torch.Tensor:
                x.add_(1)
                return x * 2

            os.environ["CGFX_TRACING"] = "0"
            x = torch.ones(4, device="cuda")
            trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        enabled = torch.ops.cudagraph_fx_test.enabled(arg_0);  arg_0 = None
        getitem = enabled[0];  enabled = None
        return (getitem,)""",
        )
        node = next(node for node in trace.graph.nodes if node.name == "enabled")
        self.assertTrue(node.target._schema.arguments[0].alias_info.is_write)

        replay_x, eager_x = x.clone(), x.clone()
        actual = trace.replay(replay_x)
        expected = fn(eager_x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(replay_x, eager_x, rtol=0, atol=0)

    @requires_gpus(1)
    def test_trace_leaf_disabled_traces_inner_operators(self) -> None:
        with patch.dict(os.environ, CGFX_TRACING="0"):

            @trace_leaf("cudagraph_fx_test::disabled_trace", mutates_args=("x",))
            def fn(x: torch.Tensor) -> torch.Tensor:
                x.add_(1)
                return x * 2

            x = torch.ones(4, device="cuda")
            trace = make_cudagraph_fx(fn)(x)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        add__tensor = torch.ops.aten.add_.Tensor(arg_0, 1);  arg_0 = None
        # stream 0
        mul_tensor = torch.ops.aten.mul.Tensor(add__tensor, 2);  add__tensor = None
        return (mul_tensor,)""",
        )

        replay_x, eager_x = x.clone(), x.clone()
        actual = trace.replay(replay_x)
        expected = fn(eager_x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(replay_x, eager_x, rtol=0, atol=0)

    @requires_gpus(1)
    def test_dispatchless_overload_flattens_runtime_outputs(self) -> None:
        @custom_op("cudagraph_fx_test::flat_mapping_output")
        def flat_mapping_output(
            tensor: torch.Tensor,
        ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
            return {"values": (tensor + 1, tensor * 2)}

        @custom_op("cudagraph_fx_test::flat_tensor_output")
        def flat_tensor_output(tensor: torch.Tensor) -> torch.Tensor:
            return tensor + 3

        def fn(tensor: torch.Tensor) -> dict[str, object]:
            return {
                "mapping": flat_mapping_output(tensor),
                "tensor": flat_tensor_output(tensor),
            }

        tensor = torch.randn(4, device="cuda")
        expected = fn(tensor)
        trace = make_cudagraph_fx(fn)(tensor)
        graph_state = tuple(
            (node, node.target, node.args, node.kwargs) for node in trace.graph.nodes
        )
        readable = trace.print_readable(print_output=False)
        self.assertExpectedInline(
            readable,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        flat_mapping_output = torch.ops.cudagraph_fx_test.flat_mapping_output(arg_0)
        getitem = flat_mapping_output[0]
        getitem_1 = flat_mapping_output[1];  flat_mapping_output = None
        # stream 0
        flat_tensor_output = torch.ops.cudagraph_fx_test.flat_tensor_output(arg_0);  arg_0 = None
        getitem_2 = flat_tensor_output[0];  flat_tensor_output = None
        return (getitem, getitem_1, getitem_2)""",
        )
        self.assertEqual(trace.print_readable(print_output=False), readable)
        graph_text = str(trace.graph)
        self.assertIn("args = (%flat_mapping_output, 0)", graph_text)
        self.assertIn("args = (%flat_tensor_output, 0)", graph_text)
        self.assertEqual(str(trace.graph), graph_text)
        self.assertEqual(
            tuple(
                (node, node.target, node.args, node.kwargs)
                for node in trace.graph.nodes
            ),
            graph_state,
        )

        actual = trace.replay(tensor)
        actual_flat, actual_spec = pytree.tree_flatten(actual)
        expected_flat, expected_spec = pytree.tree_flatten(expected)
        assert actual_spec == expected_spec
        for actual_leaf, expected_leaf in zip(actual_flat, expected_flat, strict=True):
            assert isinstance(actual_leaf, torch.Tensor)
            assert isinstance(expected_leaf, torch.Tensor)
            assert torch.equal(actual_leaf, expected_leaf)

    @requires_gpus(1)
    def test_basic_trace_is_opaque_and_replays(self) -> None:
        @custom_op("cudagraph_fx_test::viewy")
        def viewy(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return t.view(-1), t * 2

        @custom_op("cudagraph_fx_test::mutator", mutates_args=("tensors",))
        def mutator(tensors: list[torch.Tensor], value: float) -> torch.Tensor:
            for tensor in tensors:
                tensor.add_(value)
            return tensors[0].view(-1)

        def fn(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor]:
            view, doubled = viewy(a)
            mutated = mutator([a, b], 1.0)
            return (view + doubled + mutated + b,)

        a = torch.randn(3, device="cuda")
        b = torch.randn(3, device="cuda")
        trace = make_cudagraph_fx(fn)(a, b)
        viewy_node = next(node for node in trace.graph.nodes if node.name == "viewy")
        view_node, doubled_node = viewy_node.users
        source_meta = node_tensor_meta(viewy_node.args[0])
        view_meta, doubled_meta = map(node_tensor_meta, (view_node, doubled_node))
        assert view_meta["type"] == "input"
        assert doubled_meta["type"] == "forward_tensor"
        assert view_meta["storage_id"] == source_meta["storage_id"]
        assert doubled_meta["storage_id"] != source_meta["storage_id"]
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        # stream 0
        viewy = torch.ops.cudagraph_fx_test.viewy(arg_0)
        getitem = viewy[0]
        getitem_1 = viewy[1];  viewy = None
        # stream 0
        mutator = torch.ops.cudagraph_fx_test.mutator([arg_0, arg_1], 1.0);  arg_0 = None
        getitem_2 = mutator[0];  mutator = None
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        # stream 0
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, getitem_2);  add_tensor = getitem_2 = None
        # stream 0
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, arg_1);  add_tensor_1 = arg_1 = None
        return (add_tensor_2,)""",
        )

        replay_a, replay_b = a.clone(), b.clone()
        eager_a, eager_b = a.clone(), b.clone()
        (actual,) = trace.replay(replay_a, replay_b)
        (expected,) = fn(eager_a, eager_b)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(replay_a, eager_a, rtol=0, atol=0)
        torch.testing.assert_close(replay_b, eager_b, rtol=0, atol=0)

    @requires_gpus(1)
    def test_overload_identity_codec_and_reconstruction(self) -> None:
        @custom_op("cudagraph_fx_test::identity")
        def identity(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return tensor.view(-1), tensor * 2

        tensor = torch.randn(3, device="cuda")
        trace = make_cudagraph_fx(identity)(tensor)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        identity = torch.ops.cudagraph_fx_test.identity(arg_0);  arg_0 = None
        getitem = identity[0]
        getitem_1 = identity[1];  identity = None
        return (getitem, getitem_1)""",
        )
        actual_view, actual_doubled = trace.replay(tensor)
        expected_view, expected_doubled = identity(tensor)
        torch.testing.assert_close(actual_view, expected_view, rtol=0, atol=0)
        torch.testing.assert_close(actual_doubled, expected_doubled, rtol=0, atol=0)

        overload = next(iter(identity._cache.values()))
        name = _overload_name(identity)
        metadata = decode_overload_name(name)
        assert name.startswith("pt_")
        assert metadata["schema"] == "(Tensor(a0) tensor) -> (Tensor(a0), Tensor)"
        assert encode_overload_name(metadata) == name

        identity._cache.clear()
        rebuilt = getattr(torch.ops.cudagraph_fx_test.identity, name)
        assert str(rebuilt._schema) == str(overload._schema)
        with pytest.raises(AttributeError):
            torch.ops.cudagraph_fx_test.identity.pt_garbage

    @requires_gpus(1)
    def test_dataclass_output_is_flattened_and_reconstructed(self) -> None:
        @custom_op("cudagraph_fx_test::dataclass_output")
        def dataclass_output(tensor: torch.Tensor) -> Box:
            return Box(tensor + 1)

        tensor = torch.randn(4, device="cuda")
        trace = make_cudagraph_fx(dataclass_output)(tensor)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        dataclass_output = torch.ops.cudagraph_fx_test.dataclass_output(arg_0);  arg_0 = None
        getitem = dataclass_output[0];  dataclass_output = None
        return (getitem,)""",
        )

        replay_tensor = tensor + 2
        actual = trace.replay(replay_tensor)
        self.assertIsInstance(actual, Box)
        torch.testing.assert_close(actual.tensor, replay_tensor + 1, rtol=0, atol=0)

    def test_dataclass_roundtrip_preserves_allocations_and_opaque_identity(
        self,
    ) -> None:
        buffer = Buffer(torch.zeros(2), object())
        leaves, spec = pytree.tree_flatten(buffer)
        restored = pytree.tree_unflatten(
            leaves, pytree.treespec_loads(pytree.treespec_dumps(spec))
        )
        self.assertIsNot(restored, buffer)
        self.assertIs(restored.tensor, buffer.tensor)
        self.assertIs(restored.snapshot, buffer.snapshot)
        self.assertIs(restored.handle, buffer.handle)
        paths, _ = pytree.tree_flatten_with_path(buffer)
        self.assertEqual(
            [path[0].name for path, _ in paths], ["tensor", "handle", "snapshot"]
        )

    def test_dataclass_registration_rejects_unknown_fields(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "opaque_ref_fields names not in dataclass"
        ):
            register_dataclass_pytree(Buffer, opaque_ref_fields=("missing",))
        with self.assertRaisesRegex(TypeError, "is not a dataclass"):
            register_dataclass_pytree(object)

    @requires_gpus(1)
    def test_dataclass_none_and_opaque_outputs_replay(self) -> None:
        @custom_op("cudagraph_fx_test::buffer_step", mutates_args=("buffer.tensor",))
        def buffer_step(buffer: Buffer) -> tuple[torch.Tensor, None, OpaqueRef]:
            buffer.tensor.add_(1)
            return buffer.tensor, None, OpaqueRef(buffer.handle)

        @custom_op("cudagraph_fx_test::consume_handle", mutates_args=("tensor",))
        def consume_handle(handle: OpaqueRef, tensor: torch.Tensor) -> torch.Tensor:
            return tensor.add_(handle.value)

        def fn(tensor: torch.Tensor) -> tuple[torch.Tensor, None]:
            output, absent, handle = buffer_step(Buffer(tensor, 3))
            return consume_handle(handle, output), absent

        tensor = torch.zeros(2, device="cuda")
        trace = make_cudagraph_fx(fn, retain_cudagraph=True)(tensor)
        try:
            state = trace.cudagraph_states()
            assert state is not None
            state.cuda_graph.replay()
            state.cuda_graph.replay()
            torch.testing.assert_close(
                tensor, torch.full_like(tensor, 8), rtol=0, atol=0
            )
            actual, absent = trace.replay(torch.zeros_like(tensor))
            self.assertIsNone(absent)
            torch.testing.assert_close(
                actual, torch.full_like(tensor, 4), rtol=0, atol=0
            )
        finally:
            torch.cuda.synchronize()
            trace.release_cudagraph_states()

    def test_dataclass_registration_is_conditional(self) -> None:
        with patch.dict(os.environ):

            @dataclass
            class ConditionalBuffer:
                tensor: torch.Tensor
                handle: object

            buffer = ConditionalBuffer(torch.ones(2), object())
            for tracing in (None, "0", "true"):
                with self.subTest(tracing=tracing):
                    if tracing is None:
                        os.environ.pop("CGFX_TRACING", None)
                    else:
                        os.environ["CGFX_TRACING"] = tracing
                    trace_dataclass(ConditionalBuffer, opaque_ref_fields=("handle",))
                    leaves, _ = pytree.tree_flatten(buffer)
                    self.assertEqual(len(leaves), 1)
                    self.assertIs(leaves[0], buffer)

            os.environ["CGFX_TRACING"] = "1"
            trace_dataclass(ConditionalBuffer, opaque_ref_fields=("handle",))
            leaves, spec = pytree.tree_flatten(buffer)
            self.assertEqual(len(leaves), 2)
            self.assertIs(leaves[0], buffer.tensor)
            self.assertIsInstance(leaves[1], OpaqueRef)
            restored = pytree.tree_unflatten(leaves, spec)
            self.assertIs(restored.tensor, buffer.tensor)
            self.assertIs(restored.handle, buffer.handle)

            os.environ["CGFX_TRACING"] = "0"
            trace_dataclass(ConditionalBuffer)
            self.assertEqual(len(pytree.tree_leaves(buffer)), 2)

    @requires_gpus(1)
    def test_structured_inputs_and_outputs(self) -> None:
        @custom_op("cudagraph_fx_test::structured")
        def structured(
            tensor: torch.Tensor,
            pair: tuple[torch.Tensor, torch.Tensor],
            *,
            box: Box,
            scale: float,
        ) -> dict[str, torch.Tensor]:
            total = tensor + pair[0] + pair[1] + box.tensor
            return {"total": total * scale, "view": tensor.view(-1)}

        def fn(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> tuple[torch.Tensor]:
            out = structured(x, (y, z), box=Box(x), scale=2.0)
            return (out["total"] + out["view"],)

        x = torch.randn(4, device="cuda")
        y = torch.randn(4, device="cuda")
        z = torch.randn(4, device="cuda")
        trace = make_cudagraph_fx(fn)(x, y, z)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1, arg_2):
        # stream 0
        structured = torch.ops.cudagraph_fx_test.structured(arg_0, (arg_1, arg_2), box = Box(tensor=arg_0), scale = 2.0);  arg_0 = arg_1 = arg_2 = None
        getitem = structured[0]
        getitem_1 = structured[1];  structured = None
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return (add_tensor,)""",
        )
        (actual,) = trace.replay(x, y, z)
        (expected,) = fn(x, y, z)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @requires_gpus(1)
    def test_mutation_paths_unknown_and_metadata_rejection(self) -> None:
        @custom_op(
            "cudagraph_fx_test::path_mutator",
            mutates_args=("state['buf']", "buffers[0]", "box.tensor"),
        )
        def path_mutator(
            state: dict[str, torch.Tensor],
            buffers: list[torch.Tensor],
            box: Box,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            state["buf"].add_(1)
            buffers[0].add_(1)
            box.tensor.add_(1)
            return state["buf"].view(-1), buffers[1], box.tensor.view(-1)

        state = {
            "buf": torch.randn(3, device="cuda"),
            "other": torch.randn(3, device="cuda"),
        }
        buffers = [torch.randn(3, device="cuda"), torch.randn(3, device="cuda")]
        box = Box(torch.randn(3, device="cuda"))
        trace = make_cudagraph_fx(path_mutator)(state, buffers, box)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        # stream 0
        path_mutator = torch.ops.cudagraph_fx_test.path_mutator({'buf': arg_0, 'other': arg_1}, [arg_2, arg_3], Box(tensor=arg_4));  arg_0 = arg_1 = arg_2 = arg_3 = None
        getitem = path_mutator[0]
        getitem_1 = path_mutator[1]
        getitem_2 = path_mutator[2];  path_mutator = None
        return (getitem, getitem_1, getitem_2)""",
        )

        replay_state = {key: value.clone() for key, value in state.items()}
        eager_state = {key: value.clone() for key, value in state.items()}
        replay_buffers = [value.clone() for value in buffers]
        eager_buffers = [value.clone() for value in buffers]
        replay_box = Box(box.tensor.clone())
        eager_box = Box(box.tensor.clone())
        actual = trace.replay(replay_state, replay_buffers, replay_box)
        expected = path_mutator(eager_state, eager_buffers, eager_box)
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        for key in replay_state:
            torch.testing.assert_close(
                replay_state[key], eager_state[key], rtol=0, atol=0
            )
        for actual_tensor, expected_tensor in zip(
            replay_buffers, eager_buffers, strict=True
        ):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        torch.testing.assert_close(replay_box.tensor, eager_box.tensor, rtol=0, atol=0)

        schema = str(next(iter(path_mutator._cache.values()))._schema)
        assert "Tensor(a0!) state0" in schema
        assert "Tensor state1" in schema
        assert "Tensor(a1!) buffers0" in schema
        assert "Tensor(a2) buffers1" in schema
        assert "Tensor(a3!) box0" in schema

        @custom_op("cudagraph_fx_test::unknown_mutator", mutates_args="unknown")
        def unknown_mutator(tensor: torch.Tensor) -> torch.Tensor:
            tensor.add_(1)
            return tensor.view(-1)

        unknown_trace = make_cudagraph_fx(unknown_mutator)(state["buf"])
        self.assertExpectedInline(
            unknown_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        unknown_mutator = torch.ops.cudagraph_fx_test.unknown_mutator(arg_0);  arg_0 = None
        getitem = unknown_mutator[0];  unknown_mutator = None
        return (getitem,)""",
        )
        replay_tensor = state["buf"].clone()
        eager_tensor = state["buf"].clone()
        actual = unknown_trace.replay(replay_tensor)
        expected = unknown_mutator(eager_tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(replay_tensor, eager_tensor, rtol=0, atol=0)

        @custom_op("cudagraph_fx_test::metadata_mutator", mutates_args="tensor")
        def metadata_mutator(tensor: torch.Tensor) -> torch.Tensor:
            tensor.t_()
            return tensor

        with pytest.raises(RuntimeError, match="metadata-changing input mutations"):
            make_cudagraph_fx(metadata_mutator)(torch.randn(2, 3, device="cuda"))

    @requires_gpus(1)
    def test_none_optional_and_divergent_overloads(self) -> None:
        @custom_op("cudagraph_fx_test::optional")
        def optional(
            tensor: torch.Tensor, values: list[torch.Tensor] | None
        ) -> torch.Tensor:
            return tensor + 1 if values is None else tensor + sum(values)

        def none_fn(tensor: torch.Tensor) -> torch.Tensor:
            return optional(tensor, None)

        def list_fn(tensor: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return optional(tensor, [value])

        x = torch.randn(3, device="cuda")
        y = torch.randn(3, device="cuda")
        none_trace = make_cudagraph_fx(none_fn)(x)
        list_trace = make_cudagraph_fx(list_fn)(x, y)
        self.assertExpectedInline(
            none_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        optional = torch.ops.cudagraph_fx_test.optional(arg_0, None);  arg_0 = None
        getitem = optional[0];  optional = None
        return (getitem,)""",
        )
        self.assertExpectedInline(
            list_trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        # stream 0
        optional = torch.ops.cudagraph_fx_test.optional(arg_0, [arg_1]);  arg_0 = arg_1 = None
        getitem = optional[0];  optional = None
        return (getitem,)""",
        )
        torch.testing.assert_close(none_trace.replay(x), none_fn(x), rtol=0, atol=0)
        torch.testing.assert_close(
            list_trace.replay(x, y), list_fn(x, y), rtol=0, atol=0
        )

    @requires_gpus(1)
    def test_opaque_input_and_output(self) -> None:
        @custom_op("cudagraph_fx_test::opaque")
        def opaque(tensor: torch.Tensor, scale: Scale) -> tuple[torch.Tensor, Scale]:
            return tensor * scale.value, scale

        tensor = torch.randn(3, device="cuda")
        scale = Scale(3)
        trace = make_cudagraph_fx(opaque)(tensor, scale)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        # stream 0
        opaque = torch.ops.cudagraph_fx_test.opaque(arg_0, arg_1);  arg_0 = arg_1 = None
        getitem = opaque[0]
        getitem_1 = opaque[1];  opaque = None
        return (getitem, getitem_1)""",
        )
        actual_tensor, actual_scale = trace.replay(tensor, scale)
        expected_tensor, expected_scale = opaque(tensor, scale)
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        assert actual_scale == expected_scale

    @requires_gpus(1)
    def test_enum_input_and_output(self) -> None:
        @custom_op("cudagraph_fx_test::enum_graph")
        def enum_op(tensor: torch.Tensor, scale: Enum) -> tuple[torch.Tensor, Enum]:
            return tensor * scale.value, scale

        tensor = torch.randn(3, device="cuda")
        trace = make_cudagraph_fx(enum_op)(tensor, PlainScaleFactor.DOUBLE)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        enum_graph = torch.ops.cudagraph_fx_test.enum_graph(arg_0, PlainScaleFactor.DOUBLE);  arg_0 = None
        getitem = enum_graph[0];  enum_graph = None
        _constant_0 = self._constant_0
        return (getitem, _constant_0)""",
        )
        actual_tensor, actual_scale = trace.replay(tensor, PlainScaleFactor.TRIPLE)
        torch.testing.assert_close(actual_tensor, tensor * 2, rtol=0, atol=0)
        assert actual_scale is PlainScaleFactor.DOUBLE

    @requires_gpus(1)
    def test_enum_argument_default_and_output(self) -> None:
        @custom_op("cudagraph_fx_test::enum")
        def enum_op(
            tensor: torch.Tensor,
            scale: Enum = ScaleFactor.DOUBLE,
            *,
            bias: torch.Tensor,
        ) -> tuple[torch.Tensor, Enum]:
            return tensor * scale.value + bias, scale

        tensor = torch.arange(1, 5, device="cuda", dtype=torch.float32)
        bias = torch.ones_like(tensor)
        for kwargs in (
            {},
            {"scale": ScaleFactor.TRIPLE},
            {"scale": PlainScaleFactor.TRIPLE},
        ):
            with self.subTest(kwargs=kwargs):
                for _ in range(3):
                    enum_op(tensor, **kwargs, bias=bias)
                trace = make_cudagraph_fx(enum_op, retain_cudagraph=True)(
                    tensor, **kwargs, bias=bias
                )
                try:
                    tensor.add_(1)
                    bias.add_(2)
                    expected_tensor, expected_scale = enum_op(
                        tensor, **kwargs, bias=bias
                    )
                    replay_kwargs = (
                        {"scale": type(kwargs["scale"]).DOUBLE} if kwargs else {}
                    )
                    actual_tensor, actual_scale = trace.replay(
                        tensor, **replay_kwargs, bias=bias
                    )
                    torch.testing.assert_close(
                        actual_tensor, expected_tensor, rtol=0, atol=0
                    )
                    assert actual_scale is expected_scale
                    state = trace.cudagraph_states()
                    assert state is not None
                    state.cuda_graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        state.flat_outputs[0], expected_tensor, rtol=0, atol=0
                    )
                finally:
                    trace.release_cudagraph_states()

    @requires_gpus(1)
    def test_opaque_reference_input_and_output(self) -> None:
        @custom_op("cudagraph_fx_test::opaque_reference")
        def opaque_reference(
            tensor: torch.Tensor, reference: Reference
        ) -> tuple[torch.Tensor, Reference]:
            return tensor + 1, reference

        tensor = torch.randn(3, device="cuda")
        trace = make_cudagraph_fx(opaque_reference)(tensor, Reference())
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        # stream 0
        opaque_reference = torch.ops.cudagraph_fx_test.opaque_reference(arg_0, arg_1);  arg_0 = arg_1 = None
        getitem = opaque_reference[0]
        getitem_1 = opaque_reference[1];  opaque_reference = None
        return (getitem, getitem_1)""",
        )

        replay_reference = Reference()
        actual_tensor, actual_reference = trace.replay(tensor, replay_reference)
        torch.testing.assert_close(actual_tensor, tensor + 1, rtol=0, atol=0)
        assert actual_reference is replay_reference

    @requires_gpus(1)
    def test_void_mutation_is_preserved(self) -> None:
        @custom_op("cudagraph_fx_test::void_mutator", mutates_args="tensor")
        def void_mutator(tensor: torch.Tensor) -> None:
            tensor.add_(2)

        def fn(tensor: torch.Tensor) -> tuple[torch.Tensor]:
            void_mutator(tensor)
            return (tensor * 3,)

        tensor = torch.randn(4, device="cuda")
        trace = make_cudagraph_fx(fn)(tensor)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        void_mutator = torch.ops.cudagraph_fx_test.void_mutator(arg_0);  void_mutator = None
        # stream 0
        mul_tensor = torch.ops.aten.mul.Tensor(arg_0, 3);  arg_0 = None
        return (mul_tensor,)""",
        )

        void_node = next(
            node for node in trace.graph.nodes if node.name == "void_mutator"
        )
        assert node_fx_id(void_node) == 1
        execution = node_metadata(void_node)["execution"]
        assert isinstance(execution, dict)
        assert execution["stream"] == 0

        replay_input = tensor.clone()
        eager_input = tensor.clone()
        (actual,) = trace.replay(replay_input)
        (expected,) = fn(eager_input)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(replay_input, eager_input, rtol=0, atol=0)

    @requires_gpus(1)
    def test_nested_custom_op_is_absorbed_and_mode_restored(self) -> None:
        @custom_op("cudagraph_fx_test::inner")
        def inner(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.sin() + 1

        @custom_op("cudagraph_fx_test::outer")
        def outer(tensor: torch.Tensor) -> torch.Tensor:
            return inner(tensor) * 2

        def fn(tensor: torch.Tensor) -> tuple[torch.Tensor]:
            return (outer(tensor) + tensor.cos(),)

        tensor = torch.randn(8, device="cuda")
        trace = make_cudagraph_fx(fn)(tensor)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        outer = torch.ops.cudagraph_fx_test.outer(arg_0)
        getitem = outer[0];  outer = None
        # stream 0
        cos_default = torch.ops.aten.cos.default(arg_0);  arg_0 = None
        # stream 0
        add_tensor = torch.ops.aten.add.Tensor(getitem, cos_default);  getitem = cos_default = None
        return (add_tensor,)""",
        )
        (actual,) = trace.replay(tensor)
        (expected,) = fn(tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        @custom_op("cudagraph_fx_test::raises")
        def raises(tensor: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("expected custom-op failure")

        with pytest.raises(RuntimeError, match="expected custom-op failure"):
            make_cudagraph_fx(raises)(tensor)
        (actual,) = make_cudagraph_fx(fn)(tensor).replay(tensor)
        (expected,) = fn(tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @requires_gpus(1)
    def test_shadowed_and_nested_cudagraph_modes_fail_loudly(self) -> None:
        @custom_op("cudagraph_fx_test::mode_check")
        def mode_check(tensor: torch.Tensor) -> torch.Tensor:
            return tensor + 1

        class ForeignMode(TorchDispatchMode):
            def __torch_dispatch__(
                self,
                func: Any,
                types: Any,
                args: tuple[Any, ...] = (),
                kwargs: dict[str, Any] | None = None,
            ) -> Any:
                return func(*args, **(kwargs or {}))

        tensor = torch.randn(3, device="cuda")

        def shadowed(value: torch.Tensor) -> torch.Tensor:
            with ForeignMode():
                return mode_check(value)

        with pytest.raises(RuntimeError, match="shadowing"):
            make_cudagraph_fx(shadowed)(tensor)

        def nested(value: torch.Tensor) -> torch.Tensor:
            with _CudagraphFxDispatchMode(fx.Graph()):
                return mode_check(value)

        with pytest.raises(RuntimeError, match="nested"):
            make_cudagraph_fx(nested)(tensor)

    @requires_gpus(1)
    def test_more_than_dispatcher_argument_limit(self) -> None:
        @custom_op("cudagraph_fx_test::many_inputs")
        def many_inputs(tensors: list[torch.Tensor]) -> torch.Tensor:
            return sum(tensors[1:], tensors[0].clone())

        inputs = [torch.randn(2, device="cuda") for _ in range(70)]
        trace = make_cudagraph_fx(many_inputs)(inputs)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, arg_27, arg_28, arg_29, arg_30, arg_31, arg_32, arg_33, arg_34, arg_35, arg_36, arg_37, arg_38, arg_39, arg_40, arg_41, arg_42, arg_43, arg_44, arg_45, arg_46, arg_47, arg_48, arg_49, arg_50, arg_51, arg_52, arg_53, arg_54, arg_55, arg_56, arg_57, arg_58, arg_59, arg_60, arg_61, arg_62, arg_63, arg_64, arg_65, arg_66, arg_67, arg_68, arg_69):
        # stream 0
        many_inputs = torch.ops.cudagraph_fx_test.many_inputs([arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, arg_27, arg_28, arg_29, arg_30, arg_31, arg_32, arg_33, arg_34, arg_35, arg_36, arg_37, arg_38, arg_39, arg_40, arg_41, arg_42, arg_43, arg_44, arg_45, arg_46, arg_47, arg_48, arg_49, arg_50, arg_51, arg_52, arg_53, arg_54, arg_55, arg_56, arg_57, arg_58, arg_59, arg_60, arg_61, arg_62, arg_63, arg_64, arg_65, arg_66, arg_67, arg_68, arg_69]);  arg_0 = arg_1 = arg_2 = arg_3 = arg_4 = arg_5 = arg_6 = arg_7 = arg_8 = arg_9 = arg_10 = arg_11 = arg_12 = arg_13 = arg_14 = arg_15 = arg_16 = arg_17 = arg_18 = arg_19 = arg_20 = arg_21 = arg_22 = arg_23 = arg_24 = arg_25 = arg_26 = arg_27 = arg_28 = arg_29 = arg_30 = arg_31 = arg_32 = arg_33 = arg_34 = arg_35 = arg_36 = arg_37 = arg_38 = arg_39 = arg_40 = arg_41 = arg_42 = arg_43 = arg_44 = arg_45 = arg_46 = arg_47 = arg_48 = arg_49 = arg_50 = arg_51 = arg_52 = arg_53 = arg_54 = arg_55 = arg_56 = arg_57 = arg_58 = arg_59 = arg_60 = arg_61 = arg_62 = arg_63 = arg_64 = arg_65 = arg_66 = arg_67 = arg_68 = arg_69 = None
        getitem = many_inputs[0];  many_inputs = None
        return (getitem,)""",
        )
        torch.testing.assert_close(
            trace.replay(inputs), many_inputs(inputs), rtol=0, atol=0
        )

    @requires_gpus(1)
    def test_structured_display(self) -> None:
        @custom_op("cudagraph_fx_test::display")
        def display(values: list[torch.Tensor]) -> torch.Tensor:
            return values[0] + values[1]

        x = torch.randn(3, device="cuda")
        y = torch.randn(3, device="cuda")
        trace = make_cudagraph_fx(display)([x, y])
        expected_graph = """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0, arg_1):
        # stream 0
        display = torch.ops.cudagraph_fx_test.display([arg_0, arg_1]);  arg_0 = arg_1 = None
        getitem = display[0];  display = None
        return (getitem,)"""
        self.assertExpectedInline(
            trace.print_readable(print_output=False), expected_graph
        )

        torch.testing.assert_close(
            trace.replay([x, y]), display([x, y]), rtol=0, atol=0
        )

    @requires_gpus(1)
    @pytest.mark.skipif(
        not annotations_available(),
        reason="capture completeness needs cudaGraphNodeGetToolsId",
    )
    def test_raw_kernel_inside_custom_op_is_capture_complete(self) -> None:
        @custom_op("cudagraph_fx_test::raw_kernel")
        def raw_kernel(tensor: torch.Tensor) -> torch.Tensor:
            _add_one_inplace[(1,)](tensor, tensor.numel(), BLOCK_SIZE=32)
            return tensor

        tensor = torch.zeros(32, device="cuda")
        raw_kernel(tensor)
        torch.cuda.synchronize()
        tensor.zero_()
        torch.cuda.synchronize()

        trace = make_cudagraph_fx(raw_kernel)(tensor)
        self.assertExpectedInline(
            trace.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg_0):
        # stream 0
        raw_kernel = torch.ops.cudagraph_fx_test.raw_kernel(arg_0);  arg_0 = None
        getitem = raw_kernel[0];  raw_kernel = None
        return (getitem,)""",
        )
        replay_tensor = tensor.clone()
        eager_tensor = tensor.clone()
        actual = trace.replay(replay_tensor)
        expected = raw_kernel(eager_tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(replay_tensor, eager_tensor, rtol=0, atol=0)

    @requires_gpus(1)
    def test_none_output_leaves_replay(self) -> None:
        for return_tensor in (False, True):
            with self.subTest(return_tensor=return_tensor):

                @custom_op(
                    "cudagraph_fx_test::none_output_leaves", mutates_args=("tensor",)
                )
                def optional_output(
                    tensor: torch.Tensor, return_tensor: bool = return_tensor
                ) -> dict[str, tuple[torch.Tensor | None, None]]:
                    tensor.add_(1)
                    return {"values": (tensor if return_tensor else None, None)}

                trace = make_cudagraph_fx(optional_output)(
                    torch.zeros(3, device="cuda")
                )
                replay_tensor = torch.ones(3, device="cuda")
                actual = trace.replay(replay_tensor)

                assert isinstance(actual, dict)
                assert isinstance(actual["values"], tuple)
                assert actual["values"][1] is None
                if return_tensor:
                    assert actual["values"][0] is replay_tensor
                else:
                    assert actual["values"][0] is None
                torch.testing.assert_close(
                    replay_tensor, torch.full_like(replay_tensor, 2), rtol=0, atol=0
                )

                overload = next(iter(optional_output._cache.values()))
                assert str(overload._schema.returns[1].type) == "NoneType"

    def test_opaque_ref_wrapper_is_a_pytree_leaf(self) -> None:
        payload = {"tensor": torch.ones(2), "handle": object()}
        reference = OpaqueRef(payload)
        leaves, spec = pytree.tree_flatten(reference)

        assert is_opaque_type(OpaqueRef)
        assert len(leaves) == 1
        assert leaves[0] is reference
        assert pytree.tree_unflatten(leaves, spec).value is payload

    @requires_gpus(1)
    def test_opaque_ref_wrapper_replays_with_new_payload(self) -> None:
        @custom_op("cudagraph_fx_test::opaque_ref_wrapper")
        def apply_offset(
            tensor: torch.Tensor, reference: OpaqueRef
        ) -> tuple[torch.Tensor, OpaqueRef]:
            return tensor + reference.value["offset"], OpaqueRef(reference.value)

        tensor = torch.ones(3, device="cuda")
        trace = make_cudagraph_fx(apply_offset)(tensor, OpaqueRef({"offset": 1}))
        replay_payload = {"offset": 4}
        actual_tensor, actual_reference = trace.replay(
            tensor, OpaqueRef(replay_payload)
        )

        assert isinstance(actual_reference, OpaqueRef)
        assert actual_reference.value is replay_payload
        torch.testing.assert_close(actual_tensor, tensor + 4, rtol=0, atol=0)
