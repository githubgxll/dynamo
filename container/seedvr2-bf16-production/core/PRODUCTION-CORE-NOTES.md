# SeedVR2 DiT-only BF16 production core

This is a local production extraction of the GPU-tested diagnostic helper at
`b8255f7cfd9c46062d7b9c1842c05caa5fc224ad`, on the unchanged `26d3767f...`
Omni 0.32 / Torch 2.13 parent. It is **not a new GPU validation result**.

The diagnostic run passed the original P001 failing window, full P001 and C02:
three cases, 14 business windows, SP4 / W33. Production SR8 / W65, this new
installed helper, performance, peak memory and human image-quality comparison
require fresh acceptance against the resulting image digest.

## Installed boundary

Install only `bf16_runtime.py` as
`vllm_omni.diffusion.models.seedvr2.bf16_runtime`. There is no bootstrap,
sitecustomize, meta-path loader, diagnostic dependency, global torch default or
non-SeedVR2 model patch. The pinned NaDiT already casts video/text inputs to the
DiT weight dtype, so changing DiT parameters makes its main activations BF16;
explicit source FP32 RMS/RoPE calculations remain unchanged.

`SEEDVR2_DIT_BF16=1` is the default. Exactly `0` selects the original call and
tensor identity path: no conversion, no autocast override, no velocity cast.
The first helper use freezes a process-local Boolean snapshot. Later environment
changes cannot change precision; restart the model process to apply a new flag.
Invalid initial values fail. Configuration is deployment-wide: every rank of a
healthy SP group must start with the same flag and topology.

`configure_once` remains the first statement of `SeedVR2Pipeline.forward`, before
VAE work, generator setup, posterior sampling or DiT collectives. It checks the
original FP16 parameter/text boundary and CUDA BF16 support; clones original
runtime RoPE frequencies **before** `.to(bfloat16)`, restores those exact values
as FP32 and clears each `_axis_cache`; converts only DiT and `self.text`; checks
that VAE parameters remain FP16. It does not recover any bits historically lost
before these runtime values were loaded.

Only `run_transformer` disables ambient autocast, and only while calling DiT.
VAE, posterior/noise, timestep and solver source and contexts are unchanged.
`velocity_to_fp16` checks native BF16 output for finite values and FP16 range,
globally agrees success/failure over the actual SP device group, explicitly
casts to FP16, checks finite again and globally agrees before solver/VAE.
No clamp, nan_to_num, solver replacement, RNG access or weight-file writes exist.

## Deliberate differences from diagnostic implementation

- Removed `sr_numeric_probe`, observation AST composition, source import hooks,
  request-associated tensor summaries and per-tensor diagnostic logs.
- Removed all file writes, run IDs, output directories, source-file runtime
  hashing and diagnostic contract/vendor loading. The build owns pinned source
  hash verification, helper installation and exact wheel RECORD row updates.
- Removed receipt-success/failure-specific extra all-reduces. Each retained
  numerical boundary has one failure-MAX all-reduce: configuration precheck and
  postcheck once per pipeline, native velocity and FP16 velocity per call.
- Logging is one small model-configuration report per model/rank, after successful
  setup (or a passthrough identity report when the flag is `0`).
  The report is cached on the pipeline and public reads return an independent
  metadata copy. It inventories dtypes/sizes only, never tensor value statistics.
  Metadata/logging errors are swallowed and never change collective ordering.
- `_blocks` and `_numeric_stats` preserve the diagnostic finite/absmax semantics;
  noncontiguous views narrow before reshape, with at most 262144 elements per
  statistics block. No whole-output FP32/contiguous diagnostic copy is made.

All-rank agreement applies to healthy participating ranks and an initialized,
matching SP group. It does not promise recovery from a crashed worker, failed
transport, invalid topology or fatal CUDA error. A configuration conversion
failure marks no pipeline configured and fails the call; discard/restart that
process rather than reusing a potentially partially converted model.

## Source preservation and checks

`source_patch.apply_source(bytes)` has no I/O and returns patched bytes plus a
JSON-safe proof. It rejects any parent other than SHA256
`bc25563a73707078626826cd665cf2eb7612a7212950a57c2bba32a3f9210658`.
Exactly three small byte spans change: helper import, first-forward setup and
velocity call/bridge. All other source bytes remain identical; original and
patched code compile, and an independent AST transformation proves that only
those three semantic changes occurred. A second application fails closed.

Run standard-library static/protocol checks from this production directory:

```text
python -m unittest discover -s tests -p test_core.py -v
```

Run real CPU helper checks with the parent Linux Torch 2.13 environment:

```text
python tests/cpu_selfcheck.py --output /tmp/seedvr2-bf16-cpu-selfcheck.json
```

The CPU script performs real toy-tensor conversion, RoPE value preservation,
autocast restoration, bounded noncontiguous statistics, finite/range rejection
and RNG/VAE non-mutation. It makes no CUDA calls or distributed/GPU/media claim.
On a Windows host without Torch it exits 2 with `NOT_RUN_TORCH_UNAVAILABLE`;
do not install a large Torch package merely to make that status disappear.
