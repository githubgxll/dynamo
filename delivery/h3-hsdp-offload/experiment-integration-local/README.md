# Offload experiment integration patch

Status: **LOCAL_PATCH_TESTED_NOT_DEPLOYABLE**. This directory is a reviewable patch bundle, not a runnable experiment.

`experiment.patch` uses zero-context unified diff. Verify the source hashes in integration.json before applying with `git apply --unidiff-zero` in a new package. It changes only the controller/runtime integration and adds seven local contract tests. The input package, its existing RUNs, image bindings and frozen configs are unchanged.

After publication, create a new code-only experiment package, apply this diff, refresh the new image binding, source locks, repair gate hashes and package seal using the actual build evidence. The old image identity is not runnable as a new candidate. No image digest or commit has been fabricated here.

Prepare controls (available after integration): `--offload-fastpath 0|1` and `--offload-timing 0|1`. They default to absent, preserving compatibility with old configs. For a controlled A/B, explicitly freeze both flags. Only the H3 child receives them; SR and the converter receive neither. Telemetry goes to the existing H3 log as `H3_OFFLOAD_TIMING` records; `H3_OFFLOAD_TIMING_DIR` is deliberately not passed through.

The new `launch-argv.json` records explicit `h3_offload_env`; `config.json` and ConfigMap include the same frozen controls. Admission checks and launch/sampling/parallelism arguments are unchanged.

Local tests mock image binding only while generating disposable manifests. Their success does not confirm GPU correctness, performance or quality. See `LOCAL-VALIDATION.log` and `integration.json`.
