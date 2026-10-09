<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo — Agent Guide

## Overview

Dynamo is NVIDIA's open-source, datacenter-scale distributed inference framework. It is
the orchestration layer **above** inference engines (SGLang, TensorRT-LLM, vLLM), not a
replacement for them: it turns a cluster of GPUs into one coordinated inference system.
Core capabilities are disaggregated prefill/decode serving, KV-aware routing, multi-tier
KV cache management (KVBM: GPU → CPU → SSD → remote), SLA-driven autoscaling (Planner),
in-flight fault tolerance, and a Kubernetes operator for deployment.

The stack is deliberately layered and large. A **Rust core** (a Cargo workspace of
twenty-plus crates, mostly under `lib/`) holds the runtime, LLM, routing, and
KV-block-manager engines. A **Python
extensibility layer** (the `ai-dingo` wheel, bound to the Rust core through PyO3/maturin)
holds the frontend, backends, planner, and profiler. A **Kubernetes layer** (`deploy/`)
holds the operator, Helm charts, and gateway integration. Treat any change that crosses
these boundaries as non-trivial. Dynamo also sits inside a wider `ai-dynamo` ecosystem of
sibling repos (below) that it integrates with rather than vendors.

## Skills

This fork does not vendor the upstream agent skill bundles. Use installed skills when applicable.

If repository-local skills are added, use kebab-case directory names matching YAML frontmatter `name`, a third-person `description` of at most 1024 characters, `license: Apache-2.0`, and `metadata` with `author` and nonempty `tags`. List every local skill exactly once in this Skills section; `scripts/validate_skills.py` checks the frontmatter and index.

## Ecosystem

Sibling repositories this repo integrates with:

| Repo | Role |
|------|------|
| [NIXL](https://github.com/ai-dynamo/nixl) | High-throughput inference data-transfer library (KV-cache transfer over RDMA/NVLink) that underpins disaggregated serving |
| [AIPerf](https://github.com/ai-dynamo/aiperf) | Benchmarking and load-generation tool used by the benchmarking guides |
| [AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) | Simulates thousands of deployment configs to find an optimal serving config before spending GPU-hours |
| [ModelExpress](https://github.com/ai-dynamo/modelexpress) | Streams model weights GPU-to-GPU via NIXL for fast replica cold-start |
| [Grove](https://github.com/ai-dynamo/grove) | Kubernetes operator for topology-aware gang scheduling |

## Repository Map

| Path | Contents |
|------|----------|
| `lib/` | Rust workspace crates: `runtime`, `llm`, `kv-router`, `kvbm-*`, `mocker`, and more (see the root [`Cargo.toml`](Cargo.toml) `[workspace] members`), plus `bindings/python` — the PyO3 extension crate, built via maturin and deliberately excluded from the workspace |
| `dingo/` | Python packages: `frontend`, `planner`, `router`, `vllm`/`sglang` backends, `mocker`, `profiler`, and more |
| `deploy/` | Kubernetes `operator`, Helm charts, `inference-gateway` ext-proc, `observability` |
| `container/` | Dockerfiles and build scripts for runtime and dev images |
| `docs/`, `fern/` | Documentation sources and the Fern docs-site config — read [`docs/AGENTS.md`](docs/AGENTS.md) before editing |
| `examples/`, `recipes/` | Runnable examples and deployment recipes — also covered by [`docs/AGENTS.md`](docs/AGENTS.md) |
| `benchmarks/`, `tests/` | Benchmark harnesses and the top-level pytest suite |

## Build

System prerequisites (Rust toolchain, `uv`, system libraries) and the VS Code / Cursor
devcontainer are covered in [`docs/contribution-guide.md`](docs/contribution-guide.md).

Python dev build (bindings + wheel, editable):

```bash
uv venv .venv && source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
cd lib/bindings/python && maturin develop --uv && cd -
uv pip install -e lib/gpu_memory_service
uv pip install -e .
python3 -m dingo.frontend --help   # verify
```

Rust-only:

```bash
cargo build                 # whole workspace
cargo build -p dynamo-llm   # one crate
```

## Test

```bash
cargo test                  # Rust
pytest -m unit tests/       # Python unit tests
```

Markers are strict (`--strict-markers`); the full marker list lives in
[`pyproject.toml`](pyproject.toml) `[tool.pytest.ini_options]`, including GPU gating
(`gpu_0` … `gpu_8`). Follow the existing pytest markers and model/GPU requirements when writing tests.

## Lint

```bash
pre-commit run --all-files            # all hooks (run `pre-commit install` first; it also installs the DCO commit-msg hook)
cargo fmt --all && cargo clippy --workspace
```

## PR and Commit Conventions

- Keep changes focused and reviewable.
- Use Conventional Commit PR titles: `type(scope): summary`. Accepted types:
  `feat`, `fix`, `docs`, `test`, `ci`, `refactor`, `perf`, `chore`, `revert`,
  `style`, and `build`.
- PR descriptions must include `Summary` and `Validation`.
- Sign every commit with DCO: `git commit -s`.
- Do not hand-edit root `CODEOWNERS`; edit `.github/codeowners/areas.yaml` and regenerate `CODEOWNERS` and `CONTRIBUTORS.md`. See `.github/codeowners/README.md` for coverage/drift validation and offline tree input.
- Dingo CI is defined in `.github/workflows/dingo-router-ci.yml` and
  `dingo-router-tests.yml`; external fork labels do not automatically approve CI.
  Keep optional KVBM post-merge execution opt-in.
- Architecture changes require a Dynamo Enhancement Proposal (DEP), filed as a GitHub
  issue on `ai-dynamo/dynamo` with `dep:*` labels.

See [`docs/contribution-guide.md`](docs/contribution-guide.md) for the full workflow
(issue sizing, CODEOWNERS, review process).

## Docs, Examples, Recipes

Any change under `docs/`, `examples/`, or `recipes/` must follow
[`docs/AGENTS.md`](docs/AGENTS.md) and the
[documentation style guide](docs/documentation-style-guide.md): SPDX headers, Fern
frontmatter (no body `# H1`), GitHub-style admonitions, and backend casing
(vLLM / SGLang / TensorRT-LLM). The deterministic subset is enforced pre-merge.
