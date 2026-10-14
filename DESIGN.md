# `torch_cudagraph_fx`

`torch_cudagraph_fx` is a tracer and IR design oriented to graph capture for
CUDA graphable training steps.  Your code must be CUDA graphable, and what
this tracer gives you is a higher level IR representation which you can
manipulate and inspect before actually recording CUDA graphs on it.  You
might find it useful for:

1. Reordering operations (e.g., prefetching) in a way that is inconvenient
   to do in the original source code as they span across module boundaries.

2. Fusions that are inconvenient to write in source code; e.g., fusing
   accumulate gradients into their preceding operations.

3. Global transformations like memory planning; ensuring operations are
   written out to symmetric memory before a collective that needs it.
   (Note that better allocator behavior can potentially be done at the layer
   below, you don't need to do it as a graph pass.)

The trace contains both forwards and backwards operation.  The IR nodes aim
to target ATen operations and custom ops; in complicated situations, it
is always correct to represent a *single kernel launch* in the graph as
a custom op (since a CUDA graph only ever consists of kernel launches.)

The IR also takes a particular opinionated stance about streams: every FX node
has a stream associated with it (via metadata), and runners of the graph must
respect these streams (we're not doing with special stream control ops or
HOPs.)  It doesn't matter that you have to switch stream when interpreting
every op, because streams evaporate entirely when you do a CUDA graph record.
(NB: events still have to be ops in the graph.)

Auxiliary goal: we would like this graph being serializable, so that it can be
used as a way of reduce startup time / potentially directly editable for
hotfixes.

NB: Doing graph manipulations is NOT trivial, because there can be
mutations/aliasing in the graph, and there can be stream events in the graph.
You will need to respect these things.

The validation criteria of a trace, is that when run, it produces exactly the
same kernels, in exactly the same order on exactly the same streams, per
PyTorch profiler.  (NB: you shouldn't profile the CUDA graph replay, as you
will lose stream information in that case--profile what you would have CUDA
graph recorded.)  NB: obviously, the Python CUDA stream ids are not stable
run-to-run, but there is a fixed semantic set of streams which you map each
stream to.  NB: stream id as displayed in the profile actually is always the
same, e.g., the default stream is always seven!  The exact numbering of these
stream IDs is internally assigned by CUDA but reproducible.

Question: what to do about warmup?

## Tracer

How do we trace operations?  Primarily:

- Via normal torch dispatch `make_fx` tracer on (custom) ops; every op is a
  `custom_op`

- GPU trace hooks, which let us intercept all PyTorch level stream changes /
  events

The primary change to downstream model code is introducing custom ops, anywhere
there is a kernel launch that isn't already inside of a custom op.

## Fake tensor

We won't use fake tensor for tracing.  All real tensors.  We will do it under
a cuda graph record relaxed (with the intention of throwing out the cuda graph
when we're done).  This prevents real kernels from being run and is reasonably
fast.  The cuda graph can also be used to identify places where we didn't have
an operator in our graph corresponding to a kernel launch.

Problem: The above strategy doesn't catch things launched on
non-cuda graphed side streams.  Might need to hook up CUPTI with the things
that CUPTI has seen.

NB: When we have cuda graph recording on, there is not an easy way to tell
what source code kernels correspond to.  Luckily, there is `mark_kernels`
which can be used to annotate the nodes in the CUDA graph.  So once you get
your CUDA graph, and you have had `mark_kernels` turned on for every traceable
op (e.g., turn it on before you run the op you're tracing--might need to patch
the tracer to be able to do that, or maybe a TorchDispatchMode is enough), you
can walk the CUDA graph and pull out the names of the kernels that didn't have
any `mark_kernels` associated with it.  This won't directly tell you where the
kernel launch is in source code, but we should have a pretty strong prior on
what kernels we need to do from the spike, so that should enough to figure it
out.

NB to use `mark_kernels` you need to run with `LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat/`

## Custom ops

Some kernel launches may be inconvenient to custom op'ify because the custom
op signature cannot support them.  There's support for custom objects in
operators and we should use that facility.

---

## Prototype notes (this annex)

This directory is a first prototype of the design above. See
[`README.md`](README.md) for the implemented API and the current set of
limitations. In short, what exists today:

- `make_cudagraph_fx` — a `make_fx`-style tracer built on a `TorchDispatchMode`
  (rather than `make_fx` itself, because stream context is ambient and must be
  captured live). Records per-node `stream` metadata and `stack_trace`, and
  captures forward + backward.
- `CudaGraphFxTrace` — owns the FX graph, exposes replay, and reports capture
  completeness.
- `CudaGraphFxInterpreter` — internal replay engine that respects per-node
  streams and wraps each op in `mark_kernels` carrying the originating
  stacktrace, so a recorded CUDA-graph kernel can be mapped back to source.
- `assert_trace_matches` — the validation criterion as an executable check.

Stream **events** are implemented as explicit `new_event` / `record_event` /
`wait_event` custom ops. Native `torch.cuda.Event` calls and CUPTI-based
side-stream detection remain outside this prototype.
