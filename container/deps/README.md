# Container Python Dependencies

Python dependency files for Dynamo container images, split by component
so each image only installs what it needs.

## Files

| File | Purpose |
|------|---------|
| `requirements.common.txt` | Core deps shared by all containers |
| `requirements.aisimulate.txt` | Exact AISimulate wheel version staged into wheelhouses |
| `requirements.planner.txt` | Planner, profiler, and global_planner deps |
| `requirements.frontend.txt` | Frontend deps |
| `requirements.benchmark.txt` | Benchmark and profiling tools |
| `requirements.test.txt` | Test-only deps |
| `requirements.dev.txt` | Dev-only tools |

## Version Pinning Strategy

- Use `==` for packages that are pure Python and well-tested.
- Use `<=` or `<` for packages that may have platform-specific builds
  (CUDA, system packages) — the max available version can differ across
  x86_64 / aarch64 and CUDA versions.
- **Never use `>=`** as it allows untested future versions that may introduce
  breaking changes, create non-reproducible builds, and cause dependency
  conflicts. Every installed version should be explicitly tested.

## AISimulate nightly availability

`stage_aisimulate.py` first downloads the exact wheel from NVIDIA's public index.
If unavailable, it builds only a version listed in `aisimulate-sources.json`.
The fallback uses a full source commit and SHA-256 archive checksum, applies
the upstream nightly version script, and builds with locked Rust dependencies.
It checks both Dynamo Rust pins and the final wheel metadata against the Python
pin; a different version is never substituted.

The `0.13.0.dev202610060000000067` source commit comes from the published
`aisimulate-core` crate's `.cargo_vcs_info.json`. Its `src/` files were compared
with that crate and match exactly. This fallback is needed because the matching
Python wheel is absent from the public NVIDIA index.

When updating the dependency, update the Python and both Rust pins together.
For an unpublished wheel, also review the corresponding source commit and archive
checksum before adding its mapping. Do not point the fallback at a moving branch
or relabel an older wheel. Source builds use the existing wheel-builder toolchain
and are cached by Docker after a successful build.
