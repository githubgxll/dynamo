<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# Deployment recipes

[Inkling](inkling/README.md)

| Model | Runtime | Topology | GPUs | K8s | Tested upstream | Notes | Other |
|---|---|---|---|---|---|---|---|
| **[GPT-OSS-120B](gpt-oss-120b/vllm/)** | vLLM | Agg + Disagg | 8x B200 / 8x H200 | ✅ | ✅ | MXFP4 MoE + FP8 KV, 8x TP1 agg / decode-heavy single-node disagg (2P6D B200, 4P4D H200), EAGLE3 spec decode, KV-aware routing, harmony reasoning + tool calling; agentic profile | ❌ |
| **[GLM-5.2](glm-5.2/)** | SGLang | Aggregated + Disaggregated | 16x/20x B200 or 24x/16x H200 | ✅ | ✅ | B200 NVFP4 or H200 FP8 with FP8 KV, KV-aware routing, EAGLE, B200 HiCache CPU offload, agentic trace profile | ❌ |
| **[DeepSeek-V4-Flash](deepseek-v4/deepseek-v4-flash/)** | vLLM | Agg + Disagg | 4x B200 / 4x H200 | ✅ | ✅ | Text — MoE 284B / 13B active, NVFP4 (B200) / public FP8 (H200) + FP8 KV, agg TP4 (B200) / DP4+TP1+EP (H200), MTP (H200), KV-aware routing, agentic trace profile, reasoning + tool calling; plus disagg 2P1D (12x B200) / 4P3D (28x H200) | ❌ |
| **[DeepSeek-V4-Pro](deepseek-v4/deepseek-v4-pro/)** | vLLM | Agg + Disagg | 8x B200 / 8x H200 | ✅ | ✅ | Text — MoE 1.6T / 49B active (1M ctx; 86k on H200), NVFP4 (B200) / public FP8 (H200) + FP8 KV, TP8 + EP, MTP-2 (B200), KV-aware routing, agentic trace profile, reasoning + tool calling; plus disagg 1P1D (16x B200) / 1P3D (32x H200) | ❌ |
