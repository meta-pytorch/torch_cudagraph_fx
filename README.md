# torch_cudagraph_fx

`torch_cudagraph_fx` is a `make_fx`-style tracer and interpreter oriented to graph
capture for CUDA-graphable training steps. It is a prototype of the design in
[`DESIGN.md`](DESIGN.md).

Your code must be CUDA-graphable. What the tracer gives you is a higher-level FX
IR that you can inspect and manipulate *before* recording CUDA graphs on it —
useful for reordering (e.g. prefetching) across module boundaries, fusions, and
global transforms like memory planning.

## API

```python
from torch_cudagraph_fx import make_cudagraph_fx

fx_trace = make_cudagraph_fx(step)(x, w)
out = fx_trace.replay(x, w)
```

`make_cudagraph_fx(fn)` returns a callable with the same signature as `fn` that
produces a `CudaGraphFxTrace`. By default, tracing runs `fn` inside a **throwaway
record-relaxed CUDA-graph capture**: kernel launches are recorded into the
(immediately discarded) graph rather than executed, so no real GPU compute runs
— the tracer wires the graph by tensor `id` and op, never by tensor data. If
`fn` performs `.backward()` / `torch.autograd.grad`, the graph contains **both
forward and backward** ops. Nodes target ATen / custom ops — in the limit a
single kernel launch is one node.

For inspection, `fx_trace.print_readable()` prints the underlying FX
`GraphModule` output with `# stream ...` comments above nodes
that carry stream metadata. Pass `include_stack_trace=True` to also include FX's
stacktrace summaries; the default is `False`.

Because tracing happens under capture, `fn` must be CUDA-graphable: no host
syncs or data-dependent control flow on this step's values, and any side stream
must be joined to the capturing stream (`side.wait_stream(main)`) before use and
joined back afterward.

### End-to-end example

```python
import torch
from torch_cudagraph_fx import make_cudagraph_fx


def step(x, w):
    return (torch.relu(x @ w) + x,)


x = torch.ones((2, 2), device="cuda")
w = torch.eye(2, device="cuda")
fx_trace = make_cudagraph_fx(step)(x, w)

for node in fx_trace.graph.nodes:
    if node.op == "call_function":
        metadata = node.meta["cudagraph_fx"]
        print(f"{node.target} stream={metadata['execution']['stream']}")

(got,) = fx_trace.replay(x, w)
print(got)
```

Expected output:

```text
aten.mm.default stream=0
aten.relu.default stream=0
aten.add.Tensor stream=0
tensor([[2., 2.],
        [2., 2.]], device='cuda:0')
```

Each `call_function` node carries:

- `node.meta["cudagraph_fx"]` — JSON-compatible metadata containing the stable
  `fx_node_id` and execution metadata.
- `node.meta["cudagraph_fx"]["execution"]["stream"]` — the semantic CUDA stream
  id the op ran on. Stream id `0` is the stream current when tracing began; side
  streams get `1, 2, …` in first-seen order.
- `node.stack_trace` — the source stack that launched the op.

### Tracing leaves

Use `trace_leaf` to provide an interception point for the kernels launched by a
CUDA-graphable Python function, allowing the tracer to capture them in the
graph as one FX node:

```python
import torch
from torch_cudagraph_fx import trace_leaf


@trace_leaf("my_ops::add")
def add(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    _triton_kernel_add[(a.numel(),)](a, b, c, out)
    return out
```

Set `CGFX_TRACING=1` before importing decorated functions. Otherwise, the
decorator returns the original function unchanged. A tracing leaf may launch
multiple kernels.
Currently, this annotation is supported by CGFX.

See the [`torch_cudagraph_fx.trace_leaf` implementation](torch_cudagraph_fx/custom_ops/custom_ops.py)
for argument and mutation annotations.

### Events

Native event calls (`torch.cuda.Event.record()`, `Stream.wait_event()`) are
plain C calls that never dispatch, so the tracer exposes event effects as
first-class **traceable custom ops**. During `make_cudagraph_fx` tracing,
`torch.cuda.Event` and `torch.cuda.streams.Event` are temporarily monkey-patched
so event construction returns a `CudaGraphEvent`; the original constructors are
restored when tracing exits. Code that constructs and uses native PyTorch CUDA
events inside the traced function is recorded as `new_event` / `record_event` /
`wait_event` nodes. You can also call the ops directly:

```python
from torch_cudagraph_fx import new_event, record_event, wait_event

def step(x, w):
    a = x @ w
    e = new_event()
    record_event(e)                 # on the current (main) stream
    with torch.cuda.stream(side):
        wait_event(e)               # side stream waits on the event
        b = a.relu()
        done = new_event(); record_event(done)
    wait_event(done)                # main waits, joining side back
    return (b + a,)
```

A CUDA event is modeled as an **opaque object** (`CudaGraphEvent`, a torch opaque
*reference* type). `new_event` mints one and `record_event` / `wait_event`
consume it, so the event's identity is carried *in the graph* by ordinary
dataflow: `record_event` and `wait_event` reference the same `new_event` node.
The ops take no effect token and no stream argument — the stream is
implicit (the current stream, captured into the node's metadata), which the
interpreter enforces. Because the event nodes are in the graph and replayed in
order on their streams, multi-stream graphs replay with the cross-stream
dependencies intact, so the result is correct — not just the stream assignment.

## Completeness check

The tracer only sees ops that go through `__torch_dispatch__`; a kernel launched
outside the dispatcher (a raw Triton kernel, a C call, a bare `cudaMemcpy`)
produces no fx node and would be silently missing. `make_cudagraph_fx` checks
for those missing launches when CUDA graph annotations are available. In that
mode, every captured FX node is annotated in the throwaway CUDA graph; after
capture, any annotatable launch (`kernel`, `memcpy`, `memset`, or
`batch_mem_op`) without that annotation is reported as missing and
`make_cudagraph_fx` raises.

```python
# Raises RuntimeError (naming the offending launches) if a kernel bypassed
# __torch_dispatch__ and is missing from the trace.
fx_trace = make_cudagraph_fx(step)(x, w)
```

The error automatically includes each missing launch's Python stack when
PyTorch's CUDA graph stack recording is available. Stacks show
`filename:line:function`, innermost first, with framework frames filtered by
PyTorch. Recording requires
`torch.cuda.graph_annotations.get_kernel_py_stacks()`, a compatible CUDA
driver, `cupti-python`, and CUPTI >= 13.3. If the API is absent, missing launches
are reported by name with `Python launch stack not captured.` PyTorch reports
any errors initializing stack recording directly.

To enable the check, run with cuda-compat on `LD_LIBRARY_PATH` so the
`cudaGraphNodeGetToolsId` driver API is available:

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat:$LD_LIBRARY_PATH python ...
```

Without cuda-compat, `make_cudagraph_fx` still traces and replays, but it skips
the completeness check and does not guarantee that dispatch-bypassing launches
were caught.

### JSON export

```python
trace = make_cudagraph_fx(step)(x, w)
trace.export_json("trace.json")
```

`export_json(path)` writes a JSON file without executing the graph
or requiring `retain_cudagraph=True`. The document has `format_version: 1` and
two tables keyed by FX node ID:

- `operators`: schema, named arguments (including defaults), and tensor returns.
  Calls also include `invoked_in`, `depends_on_gradient`, `executing_seq_nr`,
  `creating_seq_nrs`, and `forward_operators`. Placeholders and constants
  omit autograd fields. Both sequence fields are preserved, including
  multiple creating sequence numbers.
- `tensors`: `fx_producer_id`, `storage_id`, `tensor_type`, and `tensor_meta`
  containing shape, stride, dtype, device, layout, element size, byte count,
  `requires_grad`, and `is_leaf`. For projected outputs, the producer is the
  original operator.

Storage IDs are assigned in first-seen order within each trace. Tensors sharing
the same storage have the same ID, even when their views have different offsets
or shapes. IDs identify storage objects, not memory addresses or overlapping
regions, and do not retain their allocations.

Tensor types belong to individual values: a parameter is `weight`, while a plain
buffer sharing its storage can remain `constant` or `forward_tensor`. Views
inherit their source's type when the alias source is unambiguous; computations
that allocate new storage get a forward/backward tensor type. Operator alias
annotations distinguish sources such as an `out` buffer from other operands.

The top-level `outputs` field contains the graph's flattened return values,
including tensor references and literals. The output node does not get an ID
or an entry in `operators`.

Tensor references use `{"fx_node_id": ...}`. Multiple returns are named
`output_0`, `output_1`, etc.; single returns use `output`, with lists for tensor
list results. Non-tensor operator return values are omitted; unrecorded positions
within lists appear as `null`. Dtypes, devices and other non-JSON argument values are
rendered as strings, as are non-finite floating-point arguments. The file contains
metadata, not tensor contents or a serialized executable.

### Replay

To retain the original CUDA capture, pass `retain_cudagraph=True`. Replay it
through `trace.cudagraph_states()`, which exposes the native graph and flattened
input/output buffers:

```python
trace = make_cudagraph_fx(step, retain_cudagraph=True)(x, w)
cudagraph_state = trace.cudagraph_states()
try:
    assert cudagraph_state is not None
    cudagraph_state.cuda_graph.replay()
    out_flat = cudagraph_state.flat_outputs
finally:
    torch.cuda.synchronize()
    trace.release_cudagraph_states()
```

Inputs are captured at their original addresses, without cloning or copying.
Update GPU tensor contents in place between replays; mutations in the captured
step affect those same buffers. CPU scalar tensor operands can be baked into
kernel arguments and require a new trace when their values change.
Outputs are reused and overwritten by subsequent
replays, so clone any results that must survive another replay. CUDA replay
does not create a fresh autograd tape; capture backward operations as part of
the step when needed.

Callers must order input writes, replays and output reads, and finish outstanding
GPU work before release. The trace does not intercept native replay calls to
synchronize them. Keep the trace alive while using its native graph, since the
trace owns the retained input and output buffers.

Captured constants and opaque objects are owned by the caller and assumed to
remain alive for the process lifetime. Retention keeps input and output buffers
and the CUDA graph's allocation pool alive.
`release_cudagraph_states()` is idempotent and leaves the FX module usable, but its
constants and caller-held state or tensors can keep storage alive. After release,
`cudagraph_states()` returns `None`; previously obtained states retain their
buffers, but their native graph must not be replayed.

`fx_trace.replay(...)` replays the graph, and for every node:

1. switches to the node's semantic stream, and
2. wraps the op in `mark_kernels` carrying the node's stacktrace, so that when
   replay runs *inside a CUDA graph capture*, every recorded kernel
   node is annotated with the source line that produced it.

## End-to-end validation

For tests and debugging, a trace can also be checked end to end with the PyTorch
profiler: the original function and the FX replay should produce **exactly the
same kernels, in exactly the same order, on exactly the same streams**.
`torch_cudagraph_fx` ships that checker as a private prototype helper:

```python
from torch_cudagraph_fx._validation import assert_trace_matches

interp = fx_trace._interpreter()
assert_trace_matches(lambda: step(x, w), lambda: interp.run_fn(x, w))
```

This check does not require cuda-compat and can catch some missing-kernel bugs
in tests, but it is separate from the trace-time completeness check above.

## Status / prototype limitations

This is an early prototype. Known gaps (see `DESIGN.md` for context):

- **Real tensor allocations** — tracing is record-relaxed (kernels recorded, not
  run), but tensors are still really allocated, so a trace costs the step's peak
  memory.
- **Events use `torch_cudagraph_fx` ops in the FX graph.** Native PyTorch CUDA
  events constructed inside the traced function are temporarily monkey-patched
  into `new_event` / `record_event` / `wait_event` nodes. Pre-existing native
  events and raw CUDA event calls outside PyTorch are still invisible to the
  tracer.
- **The completeness check reports, it does not fix.** `uncaptured_kernels`
  tells you which launches have no fx node, but you still have to wrap them in a
  custom op so the tracer sees them. It also only covers launches that landed in
  the throwaway capture.
- **Kernel annotations depend on recent torch, with a polyfill fallback.**
  `mark_kernels` and CUDA-graph inspection come from `torch.cuda._graph_annotations`
  / `CUDAGraph.get_graph_data` when present; on older torch that predates them
  the annex falls back to a self-contained polyfill over `cuda.bindings`
  (`_cudagraph_annotations_polyfill.py`). Either way the annex imports only `torch`
  plus optional `cuda.bindings`, so downstream projects compose over torch rather
  than depending on each other. Both paths need the `cudaGraphNodeGetToolsId`
  driver API; without it `annotations_available()` is false and the completeness
  check degrades to a no-op.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
