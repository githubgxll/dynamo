<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro 部署方案

本文提供在 Dynamo 上部署 **DeepSeek-V4-Pro** 的方案，涵盖两个后端（**vLLM**、**SGLang**）和两种目标硬件（**B200**、**GB200**）。单节点聚合式服务会占满一台 B200 服务器的 8 块 GPU；在 GB200 上，该模型无法装入单个 4-GPU NVL4 托盘，因此 V4-Pro 需通过 NVLink72（MNNVL）跨两个 GB200 托盘运行，可选择**跨节点 TP=8 的聚合式部署**或 **Prefill/Decode 分离式部署**。

| 变体 | 后端 | 硬件 | 清单 | 拓扑 | 容器 |
|---------|---------|----------|----------|----------|-----------|
| **vllm-agg-b200**       | vLLM   | 1 个节点，8× B200 | [`vllm/agg/b200/deploy.yaml`](vllm/agg/b200/deploy.yaml)         | TP=8 + 专家并行（单节点）                                            | 预构建 NGC 镜像（`...1.2.0-deepseek-v4-cuda13-dev.3`，多架构） |
| **vllm-agg-gb200**      | vLLM   | 2 个节点，每节点 4× GB200（共 8 块） | [`vllm/agg/gb200/deploy.yaml`](vllm/agg/gb200/deploy.yaml)       | 跨节点 TP=8 + 专家并行，通过 ComputeDomain 使用 MNNVL（NVLink72）           | 预构建 NGC 镜像（`...1.2.0-deepseek-v4-cuda13-dev.3`，多架构） |
| **vllm-disagg-gb200**   | vLLM   | 2 个节点，每节点 4× GB200（8 块用于 Prefill + 8 块用于 Decode，共 16 块） | [`vllm/disagg/gb200/deploy.yaml`](vllm/disagg/gb200/deploy.yaml) | 1P + 1D，每个 Worker 使用 DP=8 + 专家并行，通过 ComputeDomain 使用 MNNVL（NVLink72） | 预构建 NGC 镜像（`...1.2.0-deepseek-v4-cuda13-dev.3`，多架构） |
| **sglang-agg**          | SGLang | 1 个节点，8× B200 | [`sglang/agg/deploy.yaml`](sglang/agg/deploy.yaml)               | TP=8，通过 FlashInfer 使用 MXFP4 MoE，EAGLE MTP 3/4                                   | 预构建 NGC 镜像；也可选择[自定义构建](../container/) |

部署文件旁同时提供了用于 GB200 分离式变体的性能基准测试 Job：

| 性能测试 Job | 说明 |
|---|---|
| [`vllm/disagg/gb200/perf.yaml`](vllm/disagg/gb200/perf.yaml) | 针对 `dsv4-pro-disagg-frontend:8000` 运行 `aiperf profile`，以 8K 输入/1K 输出进行并发扫描（256/512/1024）。如需执行规模较小的冒烟测试，可覆盖 Job 环境变量中的 `CONCURRENCIES`。 |

状态：**实验性**（Day-0）。模态：仅文本。

## 前置条件

1. **已安装 Dynamo Platform**——参见 [Kubernetes 部署指南](../../../docs/kubernetes/README.md)。
2. **GPU 集群。**硬件要求因变体而异：
   - **B200 变体**（`vllm-agg-b200`、`sglang-agg`）：单个节点（x86_64）上有 8 块可用的 B200 GPU。TP=8 会占满整台服务器。
   - **GB200 变体**（`vllm-agg-gb200`、`vllm-disagg-gb200`）：需要 **2 个 GB200 节点**，每个节点配备 4 块 GPU（各为一个 NVL4 托盘），并连接至**同一个 NVLink72 clique**。节点必须带有 `nvidia.com/gpu.product=NVIDIA-GB200` 标签和 `kubernetes.io/arch=arm64:NoSchedule` 污点。集群必须安装 **DRA/ComputeDomain 控制器**（可通过 `kubectl get crd | grep computedomain` 验证）；各清单中的 `ComputeDomain` CR 和 `resourceClaims` 用于让 Operator 将一组 Worker Pod 调度到同一个 NVLink72 互连域中（聚合式变体放置 2 个 Pod，分离式变体放置 4 个 Pod）。
3. **Hugging Face Token**，且有权访问 `deepseek-ai/DeepSeek-V4-Pro`。

## 快速开始

通用设置（只需运行一次，适用于所有变体）：

```bash
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# HuggingFace token secret (consumed by the download Job and, as a convenience, by the worker)
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Download model into the model-cache PVC.
# Edit model-cache/model-cache.yaml and set storageClassName to a RWX class in your cluster.
# The PVC requests 1500Gi; DeepSeek-V4-Pro is ~865 GB on disk (64 safetensors shards,
# FP4+FP8 mixed) and typically takes 1.5-3 hours to download on first apply.
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=14400s
```

### 部署——vLLM B200（`vllm-agg-b200`）

```bash
kubectl apply -f vllm/agg/b200/deploy.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~90 minutes (TP=8 weight load +
# FlashInfer autotune + cudagraph warmup). The startup probe is sized for this.
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv4-pro-agg \
  -n ${NAMESPACE} --timeout=5400s
```

### 部署——vLLM GB200 聚合式（`vllm-agg-gb200`）

```bash
kubectl apply -f vllm/agg/gb200/deploy.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~90 minutes (TP=8 weight load
# + NCCL bring-up over MNNVL + cudagraph capture across 2 nodes).
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv4-pro-agg \
  -n ${NAMESPACE} --timeout=5400s
```

### 部署——vLLM GB200 分离式（`vllm-disagg-gb200`）

```bash
kubectl apply -f vllm/disagg/gb200/deploy.yaml -n ${NAMESPACE}

# First launch of each leader takes up to ~90 minutes (DP=8 weight load +
# NIXL/UCX setup + NCCL bring-up over MNNVL + cudagraph capture).
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv4-pro-disagg \
  -n ${NAMESPACE} --timeout=5400s

# Optional: run the perf benchmark Job (8K input / 1K output sweep at c=256/512/1024)
# kubectl apply -f vllm/disagg/gb200/perf.yaml -n ${NAMESPACE}
```

### 部署——SGLang（`sglang-agg`）

```bash
kubectl apply -f sglang/agg/deploy.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~60 minutes (TP=8 weight load +
# DeepGEMM warmup + cudagraph warmup). The startup probe is sized for this.
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=sglang-dsv4-pro \
  -n ${NAMESPACE} --timeout=3600s
```

## 测试部署

为已部署的变体设置端口转发：

```bash
# vLLM B200 agg or GB200 agg (same DGD/service name — only one of these
# variants can be deployed in a given namespace at a time)
kubectl port-forward svc/dsv4-pro-agg-frontend 8000:8000 -n ${NAMESPACE}

# vLLM GB200 disagg
kubectl port-forward svc/dsv4-pro-disagg-frontend 8000:8000 -n ${NAMESPACE}

# SGLang
kubectl port-forward svc/sglang-dsv4-pro-frontend 8000:8000 -n ${NAMESPACE}
```

所有变体的请求结构均相同。根据上述 Day-0 注意事项，使用 vLLM 变体时请发送 `thinking: false`：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "chat_template_kwargs": {"thinking": false}
  }'
```

## 部署方案详情

### vLLM B200 聚合式（`vllm/agg/b200/deploy.yaml`）

| 参数 | 作用 |
|------|---------|
| `--tokenizer-mode deepseek_v4` | 选择 DeepSeek-V4 分词器 |
| `--dyn-reasoning-parser deepseek_v4` | 将思维链提取到 `message.reasoning_content` 中 |
| `--dyn-tool-call-parser deepseek_v4` | 输出与 OpenAI 兼容的结构化 `tool_calls` |
| `--attention-config '{"use_fp4_indexer_cache":true}'` | 为 CSA+HCA 注意力启用 Blackwell FP4 索引器缓存 |
| `--kv-cache-dtype fp8` + `--block-size 256` | 使用 FP8 KV 缓存；块大小与上游方案一致 |
| `--tensor-parallel-size 8 --enable-expert-parallel` | 在单节点的 8 块 GPU 上使用 TP=8，并为 MoE 专家启用 EP |
| `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'` | 采用适合较大 Pro 模型的保守 cudagraph 模式（与上游 V4-Pro 示例一致） |
| `--no-enable-flashinfer-autotune` | 启动时跳过针对各张量形状的 FlashInfer 自动调优；dsv4 要获得正确精度必须使用该参数 |
| `--max-num-seqs 256` | 并发数上限 |

### vLLM GB200 聚合式（`vllm/agg/gb200/deploy.yaml`）

V4-Pro 在磁盘上约为 865 GB，无法装入单个 GB200 NVL4 托盘（4 块 GPU 合计约 768 GB HBM），因此 GB200 聚合式方案将一个张量并行组扩展到**两个**托盘。跨节点 TP 的 all-reduce/all-gather 通过 NVLink72（MNNVL）而非 RoCE 传输。DRA `ComputeDomain` 控制器会将两个 Pod 调度到同一个 NVLink72 clique 中。

| 参数/环境变量 | 作用 |
|---|---|
| `--tensor-parallel-size 8 --enable-expert-parallel` | 跨 2 个节点（每节点 4 块 GPU，共 2 个节点）使用 TP=8 + EP，不使用 DP。 |
| `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"pass_config":{"fuse_allreduce_rms":false}}'` | 使用 FULL_AND_PIECEWISE cudagraph 和全部自定义算子；`fuse_allreduce_rms:false` 可避免启动时出现非致命的 FlashInfer trtllm allreduce-norm 工作区警告。 |
| `--attention-config '{"use_fp4_indexer_cache":true}'` + `--moe-backend deep_gemm_mega_moe` | 使用 Blackwell FP4 索引器缓存和 DeepGEMM “mega MoE”内核，与 B200 聚合式变体采用相同内核。 |
| `--no-enable-flashinfer-autotune` | 启动时跳过针对各张量形状的 FlashInfer 自动调优；dsv4 要获得正确精度必须使用该参数 |
| `NCCL_MNNVL_ENABLE=1`、`UCX_CUDA_IPC_ENABLE_MNNVL=y`、`UCX_TLS=cuda_copy,cuda_ipc,tcp`、`NCCL_NVLS_ENABLE=1`、`NCCL_P2P_LEVEL=NVL` | 启用跨节点 NVLink72/MNNVL 互连。由于 TP=8 进程组横跨 2 个节点，因此必须启用。 |
| `ComputeDomain` CR + `resourceClaimTemplate`（清单顶部） | DRA 原语，用于请求调度器按需分配 MNNVL 通道，并将一组 2 个 Pod 放置在同一个 NVLink72 clique 中。 |
| （无 `--data-parallel-rpc-port`） | 仅使用 TP——torch.distributed master 绑定 `MASTER_PORT`（29500）进行跨节点 rendezvous，同时满足 Operator 的 `wait-for-leader-mp` TCP 探测。 |

### vLLM GB200 分离式（`vllm/disagg/gb200/deploy.yaml`）

V4-Pro 在磁盘上约为 865 GB，无法装入单个 GB200 NVL4 托盘（4 块 GPU 合计约 768 GB HBM），因此 GB200 方案采用 **Prefill/Decode 分离式**结构：一个 Prefill 副本横跨 2 个 GB200 节点（DP=8 + EP），一个 Decode 副本也横跨 2 个 GB200 节点（DP=8 + EP）。DRA `ComputeDomain` 控制器会将全部 4 个 Pod 放置在同一个 NVLink72 clique 中。

| 参数/环境变量 | 作用 |
|---|---|
| `--data-parallel-size 8 --enable-expert-parallel --tensor-parallel-size 1` | 每个 Worker 跨 2 个节点（每节点 4 块 GPU，共 2 个节点）使用 DP=8 + EP，TP=1。 |
| `--data-parallel-rpc-port 29500` | 将 vLLM 的 DP 协调器绑定到 `:29500`。Dynamo Operator 的 `wait-for-leader-mp` init container 会探测 `<leader>:29500`，并在该端口接受连接前阻止 Worker 启动；将 DP 协调端口固定为 29500，可使真实 RPC 服务器直接满足探测要求，比设置一个占位监听器更简洁。 |
| `--disaggregation-mode prefill`（仅 Prefill）+ `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'` | Prefill 通过 NIXL 写入 KV 块，Decode 读取这些块。NIXL 的 UCX active-messages 控制面通过 TCP（`UCX_TLS=cuda_ipc,cuda_copy,tcp`）传输，大批量 KV 数据则通过 MNNVL 传输。 |
| `NCCL_MNNVL_ENABLE=1`、`UCX_CUDA_IPC_ENABLE_MNNVL=y`、`NCCL_NVLS_ENABLE=1`、`NCCL_P2P_LEVEL=NVL` | 启用跨节点 NVLink72/MNNVL 互连。由于 Prefill 和 Decode Worker 均横跨 2 个节点，因此必须启用。 |
| `ComputeDomain` CR + `resourceClaimTemplate`（清单顶部） | DRA 原语，用于请求调度器按需分配 MNNVL 通道，并将一组 4 个 Pod 放置在同一个 NVLink72 clique 中。缺少该配置时，跨 Pod 的 NCCL 初始化会失败；对于 DP=8 的跨 Pod all-reduce，仅使用 TCP 的回退方案不可行。 |
| `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`（Decode）、`--enforce-eager`（Prefill） | 保守的编译/计算图配置，与 B200 聚合式变体的 V4-Pro 调优一致。 |
| `--no-enable-flashinfer-autotune`（Prefill + Decode） | 启动时跳过针对各张量形状的 FlashInfer 自动调优；dsv4 要获得正确精度必须使用该参数。 |
| `--max-model-len 9280`、`--max-num-seqs 16`（Prefill）/`128`（Decode） | 上限与 8K 输入/1K 输出的基准测试形态相匹配。 |

### SGLang（`sglang/agg/deploy.yaml`）

| 参数 | 作用 |
|------|---------|
| `--dyn-reasoning-parser deepseek_v4` | 将思维链提取到 `message.reasoning_content` 中 |
| `--dyn-tool-call-parser deepseek_v4` | 输出与 OpenAI 兼容的结构化 `tool_calls` |
| `--trust-remote-code` | V4 架构的自定义建模代码需要该参数 |
| `--tp 8` | 在单节点全部 8 块 GPU 上进行张量并行 |
| `--moe-runner-backend flashinfer_mxfp4` | 通过 FlashInfer 为 V4 专家权重使用 MXFP4 MoE 内核 |
| `--speculative-algo EAGLE` + `--speculative-num-steps 3` + `--speculative-eagle-topk 1` + `--speculative-num-draft-tokens 4` | EAGLE MTP 推测解码（3 个草稿步骤、EAGLE Head 上取 top-1、每步 4 个草稿 Token） |
| `--chunked-prefill-size 4096` | 以 4K Token 为单位切分长提示词，以便与稳态 Decode 交错执行 |
| `--disable-flashinfer-autotune` | 启动时跳过针对各张量形状的自动调优；dsv4 基础镜像已提供预调优默认值 |

### 为什么使用 TP=8（而不是像 Flash 那样使用 DP=4）？

DeepSeek-V4-Pro 的磁盘占用约为 Flash 的 5.5 倍（约 865 GB 对约 160 GB）。采用 FP4+FP8 混合权重后，在典型批处理形态下仍无法装入 4 个 Rank，因此上游针对 Pro 在两个后端上测试的形态均为 **TP=8**：使用单个 B200 节点的全部 8 块 GPU，或使用由 NVLink72 连接的两个 GB200 NVL4 托盘。在 vLLM 上，专家并行叠加在 TP 之上：TP 对稠密权重（Attention/Router/Norm）进行分片，EP 对专家进行分片。在 SGLang 上，MXFP4 MoE 后端在同一 TP=8 进程组内处理专家分片。

## 模型详情

以下信息来源于 [`deepseek-ai/DeepSeek-V4-Pro` 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)（预览版本）：

| | |
|---|---|
| **模型** | `deepseek-ai/DeepSeek-V4-Pro`（MoE，总参数量 1.6T，每个 Token 激活 49B） |
| **上下文长度** | 1M Token |
| **检查点** | 混合精度——MoE 专家权重采用 FP4，大部分其他参数采用 FP8 |
| **注意力机制** | 混合压缩稀疏注意力（CSA）+ 高度压缩注意力（HCA）。vLLM 变体通过 `--attention-config '{"use_fp4_indexer_cache":true}'` 启用 Blackwell FP4 索引器缓存 |
| **残差路径** | 流形约束超连接（Manifold-Constrained Hyper-Connections，mHC） |
| **推理模式** | 通过 `chat_template_kwargs` 提供三个推理强度级别：`{}`（不思考）、`{"thinking":true,"reasoning_effort":"high"}`（高强度思考）、`{"thinking":true,"reasoning_effort":"max"}`（最大强度思考，需要 `--max-model-len >= 393216`） |
| **长上下文效率** | 根据模型卡，在 1M 上下文下，每 Token 推理 FLOPs 约为 DeepSeek-V3.2 的 27%，KV 缓存约为其 10% |
| **许可证** | MIT |

部署方案级别（各变体）的设置：

| | vLLM（`vllm-agg`） | SGLang（`sglang-agg`） |
|---|---|---|
| **后端镜像** | 预构建的 `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3`（多架构） | 预构建的 `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-deepseek-v4-cuda12-dev.3` |
| **并行方式** | TP=8，启用专家并行 | TP=8 |
| **MoE 后端** | vLLM 的 V4 专家内核（FP4） | FlashInfer MXFP4 |
| **KV 缓存** | FP8，块大小为 256 | 引擎默认值 |
| **推测解码** | — | EAGLE MTP（3 步/4 个草稿 Token） |

## 验证推理解析

两个变体的流程相同：使用同一个模型和相同的 `--dyn-reasoning-parser deepseek_v4`。根据 Day-0 注意事项，在 vLLM 变体上应省略 `chat_template_kwargs.thinking`，或将其设置为 `false`：

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
    "max_tokens": 200
  }' | python3 -m json.tool
```

预期结果：

- `choices[0].message.reasoning_content` 包含模型的思维链。
- `choices[0].message.content` 仅包含最终答案。
- 两个字段中均不应出现原始 `</think>` 标签。

如果 `reasoning_content` 为 `null`，且 `content` 中出现 `</think>`，说明推理解析器未正确接入。请确认 Worker 命令中包含 `--dyn-reasoning-parser deepseek_v4`。

## 验证工具调用

两个变体使用相同的验证流程：

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 300
  }' | python3 -m json.tool
```

预期结果：

- `choices[0].message.tool_calls` 是包含 `function.name`、`function.arguments` 和 `id` 的结构化数组。
- `choices[0].finish_reason` 为 `"tool_calls"`。
- `choices[0].message.reasoning_content` 可能包含模型选择工具时的推理过程。

如果缺少 `tool_calls`，且 `content` 中出现原始工具调用标记，请确认 Worker 命令中包含 `--dyn-tool-call-parser deepseek_v4`。

## 注意事项

### 通用

- **存储类。** 将 `model-cache/model-cache.yaml` 中的 `storageClassName` 更新为一个 RWX 存储类，使前端和 Worker Pod 均可访问该 PVC。
- **模型大小。** `deepseek-ai/DeepSeek-V4-Pro` 的磁盘占用约为 865 GB（64 个 FP4+FP8 混合格式的 safetensors 分片）。1500Gi PVC 为 HF 缓存元数据和一个备用修订版本预留了约 1.7 倍余量。
- **解析器参数。** 在 Worker 上使用 Dynamo 版本的参数（`--dyn-reasoning-parser`、`--dyn-tool-call-parser`）。各引擎原生的 `--reasoning-parser`/`--tool-call-parser` 在引擎侧工作，不会将结果传给 Dynamo OpenAI 渲染器。
- **离线模型缓存。** 两类 Worker 均设置了 `HF_HUB_OFFLINE=1`，因此引擎从 PVC 读取缓存权重，启动时不会连接 HF Hub。为安全起见仍挂载了 HF Token Secret；下载 Job 完成后，运行时并不需要它。
- **首次启动较慢。** Decode Worker 首次启动时需要在 8 个 TP Rank 间加载权重，并预热 CUDA Graph/DeepGEMM 内核；清单中的启动探针允许约 60～90 分钟的准备时间，之后才会判定就绪失败。

### vLLM 特有事项

- **预构建镜像。** 三个 vLLM 清单（`vllm/agg/b200/`、`vllm/agg/gb200/`、`vllm/disagg/gb200/`）均引用多架构镜像 `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3`。如需从源码重新构建（例如使用自定义 Dynamo 分支或不同的 vLLM 基础镜像），参见 [`<repo_root>/container/README.md`](../../../container/README.md)。
- **引擎就绪超时。** `VLLM_ENGINE_READY_TIMEOUT_S=5400` 与启动探针的时间预算一致（`failureThreshold: 540`、`periodSeconds: 10`）。
- **FlashInfer 自动调优。** 每个 vLLM Worker（Prefill 和 Decode）均设置了 `--no-enable-flashinfer-autotune`，以跳过启动时针对各张量形状的 FlashInfer 自动调优。dsv4 必须设置该参数：当前自动调优器生成的调优结果会降低 GSM8k 精度。跳过自动调优还可缩短首次启动的预热时间。
- **GB200：聚合式与分离式。** 两者都通过 MNNVL/ComputeDomain 将 V4-Pro 分布到两个 GB200 NVL4 托盘。聚合式变体跨两个节点运行一个 TP=8 进程组（延迟更低、拓扑更简单，共 2 个 Pod）；分离式变体分别运行 DP=8 的 Prefill 和 Decode Worker（高并发时稳态吞吐量更高，共 4 个 Pod）。通用服务场景建议使用聚合式变体；当工作负载能从 Prefill/Decode 分离中获益时，使用分离式变体。

### SGLang 特有事项

- **预构建镜像。** `sglang/agg/deploy.yaml` 已引用公开的 NGC Tag `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-deepseek-v4-cuda12-dev.3`。如需重新构建（例如使用自定义 Dynamo 分支或不同的 SGLang 基础镜像），参见 [`recipes/deepseek-v4/container/README.md`](../container/README.md)。
- **DeepGEMM/FlashInfer 预热。** `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` + `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` 会跳过耗时的预编译并使用快速预热路径。`--disable-flashinfer-autotune` 会跳过启动时针对各张量形状的 FlashInfer 自动调优；dsv4 基础镜像已提供预调优默认值。
- **NCCL/Gloo。** 为 Blackwell 上的 V4 NCCL 集合通信设置了 `NCCL_CUMEM_ENABLE=1`。`GLOO_SOCKET_IFNAME=eth0` 将 Gloo 固定到标准 Pod 网络接口。

## 相关部署方案

[DeepSeek-V4-Flash](../deepseek-v4-flash/) 是规模较小的同系列模型（总参数量 284B/每 Token 激活 13B，使用 4× B200），并与本方案共用相同的 dsv4 vLLM 和 SGLang 容器镜像。
