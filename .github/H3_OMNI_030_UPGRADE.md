# H3 / Omni 0.30 candidate

This change follows the image pipeline on `DingoRouter-base`, based on
`77a1716960867bf1aba0af53c1993aa352506e9e`. It is a build candidate, not a GPU
qualification report.

The original 0.30 upgrade was rebased onto this maintained branch. The three
later local network experiments (Extras-only official source, all-Alma official
HTTPS sources, and the explicit hd04 8080 proxy override) were dropped. Keep
the new base's native builder, package mirrors, retries, and Runner configuration
unchanged. This does not deploy or update the shared Runner.

## Version inputs

- vLLM `0.30.0`, source tag commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- Omni `0.30.0`, source tag commit `a8576ccb725c4e21cd13c3eb5f9a546b21149d2b`.
- Official `vllm/vllm-openai:v0.30.0-ubuntu2404`, multi-platform index pinned
  in `container/context.yaml`; CI still targets `linux/amd64`.
- Omni's released wheel is SHA-256 pinned in `install_vllm_omni.sh`.
- Keep the compiled core and all installed `nvidia-*` libraries constrained to
  their inherited versions. The shared Transformers/tokenizers API pair may
  resolve against both vLLM and Omni requirements.

## H3 dependency repair

This branch builds an H3 tuning image. `enable_kvbm` and
`enable_modelexpress` are false in the vLLM context, as requested. These are
optional Dynamo plugins, not Omni default dependencies. Do not use this image
with DynamoConnector/KVBM or a ModelExpress `mx` load format. Keep the upstream
NIXL wheels: Omni's own NIXL paths are independent of the disabled KVBM plugin.

The installer extracts the installed default requirements of ai-dingo,
ai-dingo-runtime and vLLM, and resolves them together with the full hash-pinned
Omni wheel. This fills Kubernetes/zstandard and respects the runtime's existing
Pydantic upper bound, including matching pydantic-core. It does not request the
older ai-dingo[vllm] extra pins. All Omni default dependencies are retained,
including s3tokenizer and ONNX; no ModelExpress Protobuf cap is applied.

Snapshots and constraints before/after installation are stored in build-info.
Any change to an inherited protected GPU library stops the build. This is a
candidate dependency solve; the local metadata tests do not execute the real
Linux image installation. Optional Omni extras such as `[vsa]`, `[fa4]` and
`[dev]` are outside the default profile and require separate qualification.

## Existing build and publishing contract

The executable workflow/configuration takes precedence over older CI guides:
`DingoRouter-*` pushes trigger `.github/workflows/dingo-router-ci.yml`, using
the `[self-hosted, linux, x64, dingo]` runner. The current configured target is
`registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm`, with tags derived
from the upstream runtime tag plus the first 12 characters of the source commit.
The workflow builds **and publishes** in one run; confirm the destination and
authorization before pushing a matching branch. Registry authentication remains
in `REGISTRY_USERNAME` / `REGISTRY_PASSWORD` repository secrets.

The context change invalidates the reusable builder fingerprint, so budget for
a fresh native builder. This is a CPU Linux build; no H100 allocation is needed.
The new base pins manylinux to `2026.09.05-1` and per-architecture digests, uses
the Huawei HTTPS mirror for AlmaLinux/EPEL, and the Aliyun HTTP mirror for the
Ubuntu jq transaction. HTTP here is the upstream package-source choice, not a
request to disable package signature verification. Do not restore the old
official-source rewrite or the `.61:8080` build arguments. Builds inherit the
Runner's Docker client proxy configuration and keep upstream pip/uv index args.
The historic builder digest from the old branch is not forced onto this changed
native recipe. The upstream matrix selects the builder for the new inputs.
Preserve both existing CPU probes: compiler/Inductor and media processing
(`ffmpeg`, `ffprobe`, `libx264rgb`, lossless reference-video round trip).

## Evidence and remaining gates

The workflow records the real commit SHA and final image digest in its summary
and `image-evidence-*` artifact, together with the rendered Dockerfile and
configuration. It records the builder reference, not a resolved builder digest.
The runtime records package versions, H3 source hashes and protected constraints
under `/opt/dynamo/build-info`; export that directory during image acceptance.
These records are not a complete transitive input lock. Existing mutable native
builder inputs and the older compliance baseline remain outside this update.

The complete `uv pip check` output and exit code are retained for review. The
collector audits the installed environment and reports separate default
dependency closures for vLLM and Omni, including requested extras. The sole
permitted mismatch is the fixed official base's Torch 2.13.0+cu130 metadata
request for NCCL 2.29.7 with installed NCCL 2.30.7. Its source reference,
upstream override file and pre-install protected versions must all match.
Upstream deliberately uses NCCL 2.30.7 for DeepEPv2; downgrading it to satisfy
the older Torch metadata would break that upstream choice. The reviewed
exception is also explicitly recorded when reached through Omni's dependency
closure. All direct Omni requirements must be satisfied.

There is no KVBM/NIXL mismatch exemption. Distribution path evidence distinguishes
repeated enumeration from the reviewed Ubuntu/system shadowing cases; other
duplicates and unknown conflicts still fail. Review dependency-audit.json,
the two framework reports, and before/after package records before deployment.
An accepted upstream metadata override is not GPU collective-communication
qualification.

Runtime, builder and cache image names carry no personal prefix. The proposed
publishing branch is `DingoRouter-h3-omni030-base-20260929`; the local preparation
branch is `h3/omni030`. Kubernetes Pod naming is independent of image naming.
Continue pushing new commits to this existing rebased publishing branch; do not
reuse the pre-rebase `DingoRouter-h3-omni030-20260929` branch or force-push it.
The inherited Runner/GC deployment
files are not applied by this change; actual Runner deployment state is not
validated by rebasing the repository.

## H3 protocol and deployment

Do not widen the old image-overlay version gate (0.27.1 / 0.27.0rc1), or mount
the old deployment's `pipeline_minimax_h3.py`, `vae.py`, `quality_policy.py`
over 0.30. Start with the native released implementation, keeping old baseline
packages and media immutable.

Omni 0.30 counts denoiser evaluations. For ordinary uniform H3 schedules, old
`num_inference_steps=50` means 50 sigma points / 49 evaluations; new `49`
restores that grid/work count when both flow shifts match. New `50` means 50
evaluations and is a different protocol. Record actual sigmas and NFE; neither
choice promises identical output across framework/kernel upgrades. Do not apply
this conversion to distilled models with an explicit schedule.

After publication, qualify the digest on the authorized hd04 H100 environment:
image pull, GPU kernel execution, model loading, one FL2VA case and one Ref2VA
case with video/audio reference processing, complete MP4 decode and export of
media/logs/manifests. Then assess quality/performance separately. Use only new
`jirx-` Pods, nodeSelector-based scheduling and approved model access. Model
weights, reference/test videos, output reports and credentials are not image
inputs. VBench and FastH3 remain separate follow-up image projects.
