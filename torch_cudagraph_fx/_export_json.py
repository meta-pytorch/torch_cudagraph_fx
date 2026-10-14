# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math
import operator
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch.fx as fx
from torch_cudagraph_fx._metadata import (
    JsonValue,
    node_autograd_meta,
    node_fx_id,
    node_tensor_meta,
)


def _json_value(value: Any) -> JsonValue:
    if isinstance(value, fx.Node):
        return {"fx_node_id": node_fx_id(value)}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value).removeprefix("torch.")


def _tensor_tree(nodes: list[fx.Node]) -> JsonValue:
    children: dict[int, list[fx.Node]] = {}
    for node in nodes:
        if node_tensor_meta(node):
            return {"fx_node_id": node_fx_id(node)}
        for user in node.users:
            if user.target is operator.getitem:
                children.setdefault(cast(int, user.args[1]), []).append(user)
    values = {
        index: value
        for index, projections in children.items()
        if (value := _tensor_tree(projections)) is not None
    }
    return [values.get(index) for index in range(max(values) + 1)] if values else None


def _tensor_returns(node: fx.Node) -> dict[str, JsonValue]:
    tree = _tensor_tree([node])
    if tree is None:
        return {}
    schema = getattr(node.target, "_schema", None)
    if schema is None or not isinstance(tree, list):
        return {"output": tree}
    if len(schema.returns) == 1 and schema.returns[0].type.kind() == "ListType":
        return {"output": tree}
    # Custom ops return flat lists, including when their schema has one return.
    return {
        "output" if len(schema.returns) == 1 else f"output_{index}": value
        for index, value in enumerate(tree)
        if value is not None
    }


def _operator_schema(node: fx.Node) -> tuple[str, dict[str, Any]]:
    if node.op in ("placeholder", "get_attr"):
        result_type = "Tensor" if node.op == "placeholder" else "Any"
        return f"{node.op}(str name) -> {result_type}", {"name": node.target}
    if node.op == "call_function" and node.target is operator.getitem:
        return "getitem(Any input, Any index) -> Any", {
            "input": node.args[0],
            "index": node.args[1],
        }
    schema = getattr(node.target, "_schema", None)
    if node.op != "call_function" or schema is None:
        raise TypeError(
            f"cannot export FX node {node.name!r} without an operator schema"
        )
    arguments = {arg.name: value for arg, value in zip(schema.arguments, node.args)}
    arguments.update(node.kwargs)
    for arg in schema.arguments:
        if arg.name not in arguments and arg.has_default_value():
            arguments[arg.name] = arg.default_value
    schema_text = str(schema)
    if hasattr(node.target, "_custom_op_overload_name"):
        schema_text = schema.name + schema_text.removeprefix(
            f"{schema.name}.{schema.overload_name}"
        )
    return schema_text, arguments


def _operator_record(node: fx.Node) -> dict[str, JsonValue]:
    schema, arguments = _operator_schema(node)
    record: dict[str, JsonValue] = {
        "schema": schema,
        "arguments": _json_value(arguments),
        "returns": _tensor_returns(node),
    }
    autograd = node_autograd_meta(node)
    if node.op == "call_function" and autograd:
        record.update(
            invoked_in=autograd["phase"],
            depends_on_gradient=autograd.get("depends_on_gradient"),
            executing_seq_nr=autograd.get("executing_seq_nr"),
            creating_seq_nrs=autograd.get("creating_seq_nrs", []),
            forward_operators=autograd.get("forward_node_ids", []),
        )
    return record


def export_graph(graph: fx.Graph, path: str | PathLike[str]) -> None:
    operators = {}
    tensors = {}
    outputs = None
    for node in graph.nodes:
        if node.op == "output":
            outputs = _json_value(node.args[0])
            continue
        node_id = str(node_fx_id(node))
        operators[node_id] = _operator_record(node)
        tensor = node_tensor_meta(node)
        if not tensor:
            continue
        producer = node
        while producer.target is operator.getitem:
            producer = cast(fx.Node, producer.args[0])
        tensors[node_id] = {
            "fx_producer_id": node_fx_id(producer),
            "storage_id": tensor["storage_id"],
            "tensor_type": tensor["type"],
            "tensor_meta": {
                key: value
                for key, value in tensor.items()
                if key not in ("type", "storage_id")
            },
        }
    with Path(path).open("w", encoding="utf-8") as output:
        json.dump(
            {
                "format_version": 1,
                "operators": operators,
                "tensors": tensors,
                "outputs": outputs,
            },
            output,
            allow_nan=False,
        )
        output.write("\n")
