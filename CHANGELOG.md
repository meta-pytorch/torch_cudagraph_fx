# Changelog

## Unreleased

### Added

- Initial prototype of FX tracing and replay for CUDA-graphable PyTorch code.
- Traceable CUDA events, custom operations, and stream metadata.
- CUDA graph capture completeness checks when graph annotations are available.
- JSON export of operator and tensor metadata.
- `substitute_with_kernels`: editable per-specialization kernel caches for fused
  regions.
- `SchedulePlan` with `fuse`: outline operator regions into `fused_op` calls,
  lowered deterministically to graph transformations.
- `CudaGraphFxTrace.indexed_graph`: operators addressable by phase, module path and
  operator name, with module attribution for backward operators.
