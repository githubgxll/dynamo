# 构建前复核：最新版 skill 与 Dynamo 基础版本

结论：**本次不更新 Dynamo 基础版本，不 rebase 到 main 或 DingoRouter-base。** 继续从成功的 `87365073cbdf95fa17ca352a24a93cb74fde144a47c96135e91011d1a03df72e` 镜像派生，保留 Omni 0.31.0rc1 / vLLM 0.31.0 / Torch 2.13.0+cu130 / Cache-DiT 1.5.0，只验证这次 HSDP 参数迁移修复。

这是对**原生 Omni Ref2VA8 实验**的构筑复核。候选镜像尚未构建、发布或经过真实 GPU 验收；不能写成已完成 Gateway 配置 A 接入或已降低显存峰值。

## 1. 远端版本实时核查

2026-10-27 通过 GitHub 分支 API 核对 head，再以 Git 获取固定 Base 提交检查代码。核查结果和源码 SHA256 保存在 [base-review-20261027.json](base-review-20261027.json)。下表日期是提交日期，不是镜像发布时间。

| 分支 | 远端 HEAD | 提交日期 | 项目版本 | 本次结论 |
|---|---|---|---|---|
| `DingoRouter-video-gateway` | `9ccee2f56d0122be94d7436ae2f5fec96d15ea9b` | 2026-10-19 | ai-dingo 1.3.0 | 与当前父镜像锁定的 Gateway 来源一致，无需追新 |
| `DingoRouter-base` | `f0a976ba3bbd633ca7af421564f062ce4af99213` | 2026-10-21 | ai-dingo 1.6.0 | 有更新，但与 Gateway 分支分叉；本次不迁移 |
| `main` | `049eece3f32f8d38dc1d8c52d4b793cb30e394bb` | 2026-10-08 | ai-dynamo 1.6.0 | 不是当前完整 Dingo Gateway 服务的直接替代 |
| `Dingorouter-H3-vllm-omni-jirx` | `0b11a41b083facf74d3387696a72893d0d3e40e1` | 2026-10-26 | ai-dingo 1.3.0 | 与本地 HEAD 和已成功 C14 构筑提交一致；继续使用此上传分支 |

Gateway 相对 Base 的 Git 比较为分叉：Base 有 49 个独有提交，Gateway 有 20 个独有提交，不能直接做一次快进更新。这些数量是提交图差异，包含历史/cherry-pick 差异，不代表 49 项新的必要功能。

Base 最近提交主要涉及 GLM/SGLang、推理输出以及 AISimulate 构筑；两分支的直接差异还涉及 Omni handler、Gateway 任务和性能监测代码、构筑模板与 1.3→1.6 的依赖迁移。贸然换基础会把多个因素加入本轮卸载对照。

具体核查了 Gateway、Base、main 以及实验分支的 Omni 参数文件：都没有 `diffusion_offload_config` 字段或 `--diffusion-offload-config` CLI。升级到目前这些分支头本身不能补齐配置 A 的 Dingo 参数链。底层 CPU/CUDA 存储错误发生在安装的 Omni offloader/FSDP 参数迁移中，仍需当前候选补丁和真实 GPU 验证。

这次没有查询新的 Harbor tag 来替换用户指定成功 digest；远端代码分支无变化也不能据此断言所有 Harbor tag 均未变化。构筑仍明确绑定已验证父 digest。

## 2. 按 2026-10-27 新版 skill 的检查结果

使用了 `D:/AI/skill/h3-image-handoff/SKILL.md`、其 `references/identity-and-integration.md`，以及 `h3-experiment-operations/SKILL.md`。文件哈希已记录，历史 Playbook 仅作溯源。

| 检查项 | 结果 |
|---|---|
| 固定成功父镜像，避免附带框架升级 | 符合；FROM 固定完整 digest，不安装包或下载权重 |
| 父子源码标记分别校验 | 符合；保留 lite/C14 标记，新增 HSDP 标记；禁止用 Pod env 伪装 |
| 源码来源、RECORD、依赖与既有证据保留 | 配方具备完整检查；真实镜像运行仍待 CI |
| 不关闭 HSDP、不变更 A 的组件策略 | 只替换迁移和回滚路径，选择/阶段调用保留；实际阶段行为待八卡验收 |
| 严格区分本地、CPU、GPU、媒体和生产验收 | 候选标记 GPU_NOT_RUN；先真实八卡，再 C02，再 Ref7 |
| 新实验运行身份和挂载覆盖 | 需按下节派生新实验包；不能只改旧 YAML 的 image |
| Dingo Gateway 参数透传 | 配置 A 仍未接入；属于单独的服务适配工作，不阻塞原生实验候选构建 |

当前实验的 `launch-argv.json` 明确使用原生 `vllm-omni serve`。其 Pod 只挂载 `/opt/h3-sr` 控制器、工作目录、模型和共享内存，并有可选 FP16 转换 init；本次包没有 `/sr-patch` 或 Omni 源码覆盖。不能把历史生产 overlay 风险写成本轮已发生的故障。

## 3. 新实验包必须采用的身份与来源检查

等本次候选发布后，从本次发布证据读取**新的镜像 digest 和 source_commit**，创建独立实验目录、ConfigMap、Pod 和 RUN。旧失败 Run、blocked 状态和旧 ConfigMap 保留。

| 字段 | 新实验应核对的值 |
|---|---|
| 部署 image | 本次 `deployment-image.txt` 的完整 `repo@sha256:...` |
| `H3_EXPERIMENTAL_SOURCE_COMMIT` | 保留父层 `2362a3241c03a286b98e51b8003a51f494fab4f1` |
| `H3_C14_SOURCE_COMMIT` | 保留父层 `0b11a41b083facf74d3387696a72893d0d3e40e1` |
| `H3_HSDP_PARENT_IMAGE` | `registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:87365073cbdf95fa17ca352a24a93cb74fde144a47c96135e91011d1a03df72e` |
| `H3_HSDP_SOURCE_COMMIT` | 本次发布回执的完整 `source_commit`，不是旧 C14 提交 |
| 实际卸载实现 | `vllm_omni/diffusion/offloader/sequential_backend.py` 的 import 路径和新 SHA256 |
| 实际迁移 helper | `vllm_omni/diffusion/offloader/hsdp_migration.py` 的 import 路径和 SHA256 |

两个文件的新哈希取本配方 `container/h3-hsdp-offload/source-patches.lock.json`，同时与最终镜像 `source-verification.json` 对照。可直接调用 `/opt/h3-hsdp-offload/runtime.py` 的 `verify_installed()` 完整核验；八卡验证脚本已调用该检查。

旧实验控制器只验证 lite/C14 标记，其 `SOURCE_HASHES` 清单没有这两个新文件，因此不能直接复用为新候选验收。只加新 BATCH 或只换 image 不足以证明补丁被加载。新包需保留原先对 H3 pipeline、并行、精度、模型、输入和请求的约束，再增加以上检查。

## 4. 同事 Gateway 接入边界

本镜像保留既有 Dingo 服务代码与依赖，**没有新增 Gateway 配置 A 的参数链**。如果后续通过 `python -m dingo.vllm.omni` 使用 A，需要在独立服务适配中完成：

1. `OmniDiffusionKwargs` 增加默认 None 的 `diffusion_offload_config` 字段。
2. `OmniArgGroup` 增加 JSON 对象参数；拒绝损坏 JSON、数组和字符串。
3. 核验 namespace → dataclass → `_build_omni_kwargs` → `AsyncOmni`，以及最终 Omni stage 配置。
4. 核验与 `enable_cpu_offload` 的互斥，覆盖无卸载/A/B 解析、透传与失败用例。
5. 审查同事部署实际的 initContainer、ConfigMap、PYTHONPATH/源码 overlay，避免旧覆盖替换本次修复；保留原/新哈希，不删除旧保护。
6. 在该服务组合下重新执行 C02、媒体和 Gateway 压力验收。现有 B 开关存在也不代表其底层卸载已通过。

本轮检查没有修改生产服务或实验控制器，也没有提交、推送、构建、发布或启动 Pod。后续若要统一到 Dynamo/Dingo 1.6，应单独安排版本迁移与完整服务回归。
