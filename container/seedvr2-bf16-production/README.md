# SeedVR2 DiT-only BF16 production derivative

This recipe preserves the accepted H3/SR framework, native libraries, Gateway and model files. It installs one SeedVR2 helper and changes three exact source anchors in `pipeline_seedvr2.py`. It does not install packages, regenerate weights, change the global model dtype, or copy the current repository's Dingo implementation into the image.

## Numerical boundary

- Materialized SeedVR2 DiT parameters and its fixed text condition use BF16.
- RoPE frequency values are cloned **before** conversion, restored in FP32, and their axis caches cleared. This preserves the values present at runtime; it does not recover precision lost in an earlier checkpoint conversion.
- Only the DiT call disables an ambient autocast. The caller's surrounding context is restored.
- Native BF16 velocity must be nonempty, finite and within the FP16 finite range. Healthy ranks in the matching SP group agree before conversion. The explicit FP16 result is checked and agreed again before entering the unchanged solver.
- VAE, posterior/noise generation and solver retain their FP16 path and RNG order. There is no clamp, `nan_to_num`, replacement sample, local-text-sum patch or hidden window reduction.

BF16 has a wider exponent range and different rounding than FP16. Passing a finite result through an explicit FP16 bridge retains the established downstream dtype contract; it does not establish bitwise equality or unchanged quality relative to a successful FP16 run.

## Activation and receipt

`SEEDVR2_DIT_BF16` accepts exactly `1` or `0`, defaults to `1`, and is frozen on first use in a process. Restart to change it. With `0`, transformer invocation and velocity are passed through without the new conversion or numerical gates.

The accepted production launcher sanitizes inherited `SEEDVR2_*` variables. Deployment profiles explicitly set `sr.stage_environment.SEEDVR2_DIT_BF16=1`; H3 and Gateway stages do not set it. Setting only the outer Pod variable to `0` is not a reliable rollback: use the SR stage environment or the previous image digest. Direct native SeedVR2 entrypoints are covered by the helper's default-on behavior.

Once configured, each model/rank attempts one small `SEEDVR2_BF16_PRECISION ` JSON log. Its schema is `seedvr2-dit-bf16-production/1`; it includes PID, global/SP rank, actual/configured SP size, helper version/path/SHA and parameter dtype metadata. It has no per-request tensor logging, diagnostic directory requirement, import hook or `SRDIAG_ENABLE` dependency. Logging failures cannot change the numerical collective sequence. A missing receipt is insufficient acceptance evidence, rather than proof of inference failure or success.

`precision_report(pipeline)` returns a copy of cached configuration metadata. Its GPU status describes configuration only. A receipt before first forward need not be configured.

## Preserved service contract

H3 retains BF16, Cache-DiT and null quantization. FL2VA retains no offload; Ref2VA retains its accepted module offload. The production SR profile retains eight-way Ulysses/VAE height sharding, FP16 checkpoint preparation, native `--dtype float16`, W33/W65 windows, and existing standalone SR / H3-to-SR interfaces. The BF16 helper is specific to SeedVR2; it does not change H3 precision.

The existing complete source verifier is retained byte-for-byte. Apply verifies the parent under its inherited identity, preserves its inputs/receipts/metadata/RECORD evidence, then projects only the two reviewed source changes into the complete expected manifest and exact RECORD rows. The same verifier must pass under the child identity. New release identities describe a derivative source tree, not a rebuild of all inherited files.

## Validation boundaries

The diagnostic implementation passed the original 33-frame failure window, a 192-frame complete clip and a 158-frame normal clip at SP4/W33. Fourteen business windows were finite; six raw/presentation videos, 766 decoded frames, had no fully black frames in the recorded checks.

This production extraction changes installation and observation behavior and removes diagnostic receipt-related reductions. Those historical results do **not** qualify the new production image, SP8, W65/1440p, H3-to-SR, performance or comprehensive quality. Validate those on the final published digest.

Local standard-library tests exercise source splices, metadata/RECORD preservation and protocol failures. `tests/cpu_selfcheck.py` uses real Torch toy tensors for conversion, exact RoPE restoration, autocast, finite/range/cast rejection and RNG checks. Missing Torch is `NOT_RUN`, not a pass; final Linux image validation must run with the inherited Torch installation. GPU communication and real-model acceptance remain separate.

Only healthy participating ranks with compatible configuration and SP topology are covered by the collective agreement. This mechanism does not recover from dead ranks, fatal CUDA failures, invalid process topology or arbitrary failures inside DiT execution. Reconfigure/reload or failed partial conversion requires a fresh service lifecycle and acceptance.
