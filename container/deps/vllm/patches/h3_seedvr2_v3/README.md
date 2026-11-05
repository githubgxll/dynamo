<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# H3 and SeedVR2 source overlays for Omni 0.32

This bundle preserves thirteen native `vllm_omni` source files from the verified
H3 plus SeedVR2 service. Gateway and Worker business code lives under `dingo/`.
No weights, credentials, deployment manifests or generated media are included.

`source-lock.json` pins vLLM and vLLM-Omni 0.32.0, the official Omni wheel checksum,
each upstream/replacement checksum, and the tested parent image digest. Newly
introduced modules must be absent or already match their exact reviewed bytes.
The two latest delivery files also accept the known earlier parent-image hashes.
Unknown or modified inputs fail before any writes.

The native files contain H3 module-offload recovery and memory preparation,
SeedVR2 long-video windows, telemetry and transport, the fourteen output canvases,
centered crops, two 1080P Lanczos exceptions, one final lossy encode, 32 FRAME
encoder threads and 16 FRAME full-output validation threads.

The CUDA image template invokes `install_vllm_omni_patches.sh`, which selects
this bundle only for the 0.32.0 pair. Older version-specific patch behavior stays
in its existing branch. The current default CUDA runtime base is the pinned
upstream vLLM 0.32 Ubuntu 24.04 image, and the released Omni 0.32 wheel is installed
with the tested transformers/tokenizers pair before these source files.

To verify or apply explicitly in a compatible image:

```bash
python3 container/deps/vllm/patches/h3_seedvr2_v3/apply.py \
  /usr/local/lib/python3.12/dist-packages --check
python3 container/deps/vllm/patches/h3_seedvr2_v3/apply.py \
  /usr/local/lib/python3.12/dist-packages
```

The installer is idempotent. It does not install Python packages or launch models.
The source checks reproduce the reviewed native code; they do not establish GPU
correctness or guarantee that a newly built image has the same full dependency
closure as the existing service. Build and GPU smoke validation remain required.

Gateway dimensions and capability negotiation use
`dingo/video_gateway/adapters/h3_output_policy.py`. Gateway and Worker must agree
on `minimax_h3_seedvr2.v3`. The Worker separately needs a verified FP16 SeedVR2
model path, one execution slot, and `DINGO_VIDEO_SEEDVR2_CONFIG`. Installing the
bundle alone does not enable SR. H3 module offloading is configured through its
explicit model startup options and `H3_OFFLOAD_FASTPATH`; model configuration is
not copied into this bundle.

The complete output/base/internal canvas table, crop boxes and resize exceptions
are maintained in [the output contract](../../../../../docs/video-gateway/super-resolution.md).
