# Native build prerequisite: unavailable AISimulate pin

The existing workspace and Python binding pinned `aisimulate-core=0.13.0-dev.202610060000000067`. The public crates.io index did not contain that version and its crate download returned HTTP 404, preventing even a feature-disabled native extension build from resolving dependencies.

With explicit user approval, both manifests now pin the locally available published `0.13.0-dev.202610290000000087`; Cargo regenerated both lockfiles. Its dependency closure updates parquet from 55.2.0 to 59.3.2, adds num-bigint 0.5.1 and removes obsolete closure members as reported by Cargo. This is a separate build-prerequisite change, not a KVV protocol workaround.

The KVV frontend build does not enable the optional AIS forward-pass feature. Its AIS runtime behavior is therefore not verified by KVV; any deployment enabling that feature must run its own AIS-specific compatibility checks. No placeholder crate, fake version or disabled protocol validation is used.
