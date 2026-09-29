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
- Keep compiled packages inherited from the base image constrained. Only the
  reviewed Transformers/tokenizers API pair may be resolved by Omni.

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
collector audits the whole installed environment, including dependency extras
and Python-version requirements. The only metadata exception is precisely KVBM
1.3.0 requiring `nixl[cu13]==1.0.1` while NIXL 1.3.1 is installed. This preserves
the branch's existing `--no-deps` installation policy; do not downgrade NIXL to
satisfy that historical metadata. Other missing or incompatible dependencies,
invalid metadata and checker execution errors stop the build. An allowed
metadata exception does not establish KVBM/NIXL ABI compatibility. Review the
saved dependency audit before accepting the image for deployment.

Runtime, builder and cache image names carry no personal prefix. The proposed
publishing branch is `DingoRouter-h3-omni030-base-20260929`; the local preparation
branch is `h3/omni030`. Kubernetes Pod naming is independent of image naming.
Use a new publishing branch after the rebase so the previous failed branch
remains available without force-pushing. The inherited Runner/GC deployment
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
