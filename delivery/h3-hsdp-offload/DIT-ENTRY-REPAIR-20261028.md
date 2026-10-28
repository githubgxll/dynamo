# 2026-10-28：H3 DiT 入口修复镜像交接

**结论：保留上一版 HSDP 镜像内容，新增 H3 pipeline 的 DiT 外层激活上下文。无需升级框架、Dynamo、Torch 或任何依赖。** 本次针对真实 C02 在首次 DiT forward 出现的 CPU/CUDA 混合驻留，保持原 A 策略、8×H100、BF16、U8/HSDP8、TE TP8、VAE patch8、Cache-DiT、50步、输入与 seed。

构建沿用 `Dingorouter-H3-vllm-omni-jirx` 分支和 `delivery/h3-hsdp-offload` 入口。同一配方从原 C14 固定父层重新应用已经验证过来源的迁移补丁，加上此次用户提供的 pipeline 候选。**这不是把旧卸载补丁重复应用到上一版镜像，也不是重新搭建依赖环境。**

## 已证实的问题与尚未验收的部分

上一版来源提交为 `4a5d1340cf496aeec11cc33c11416b58fe125808`，运行镜像为：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:18e218d394345c57bf837d627b7a97b6d9e1ee80c9b068583e82280398c4a5b1
```

用户的 `C02-MIXED-RESIDENCY-20261028.md` 和完整失败 ZIP 记录：八卡合成迁移预检通过，真实 Ref2VA C02 在 0/50 的首次 DiT 调用报 `Component must have homogeneous original residency; got cpu, cuda:N`。HTTP 500 的输出是错误 JSON，没有生成成功视频；不能使用这次请求耗时计算卸载性能，也不能以 Pod Ready 认定生成恢复。

源码解释是原生 FSDP pre-forward 比 Omni forward 内部卸载 hook 更早运行。等内部 hook 才恢复 DiT 时，root 已可能进入 FORWARD/unshard，而子组件仍在 CPU。原 helper 拒绝混合驻留和 active-FSDP 是必要保护；将保护放宽不会修复调用顺序。

用户候选把现成的 `_component_on_device(transformer)` 放在整个去噪循环外：先完整恢复本次选择的 DiT，再进入原 DLO 上下文与首次 FSDP forward。50步期间保持 DiT 驻 GPU，按原规则在结束后释放。FL/Ref 选择、baseline/DLO、TE-only 的语义保留。

本地源码/控制流检查不运行真实模型。新的 Linux 镜像构建、安装模块 CPU 检查、冷态八卡合成迁移、C02 cold/repeat、Ref7、显存收益和媒体质量是不同的验收层，后续结果必须分别记录。新镜像尚无可交付 digest 时，不可把上一版的 digest 写成新修复镜像。

## 构建方式与来源

Docker `FROM` 继续固定为：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:87365073cbdf95fa17ca352a24a93cb74fde144a47c96135e91011d1a03df72e
```

其来源为 `0b11a41b083facf74d3387696a72893d0d3e40e1`，Gateway 为 `9ccee2f56d0122be94d7436ae2f5fec96d15ea9b`。本次沿用 vLLM 0.31.0、Omni 0.31.0rc1、Torch 2.13.0+cu130、Cache-DiT 1.5.0，以及既有 Gateway/C14/RMSNorm 修复；不安装包、不下载模型或适配器。Base 权重仍由现有只读挂载提供。

保持原父层是为了遵守旧配方的 create-once 校验。原 `apply` 检查的是 C14 中未修改的 backend/pipeline，并要求 helper 尚不存在。若把 `FROM` 改成已经含 helper 的 `18e218…`，会正确拒绝重复补丁。本次选择复用原配方，不另行建立不同的增量父子契约。

| 安装文件 | 本次处理与最终 SHA256 |
|---|---|
| `vllm_omni/diffusion/offloader/sequential_backend.py` | 保留上一版：`b7f65e771d755f6f897ac4c8fd6753153ba0d114b5bc03578b3259ff4e987ed0` |
| `vllm_omni/diffusion/offloader/hsdp_migration.py` | 保留上一版：`1bec61bb616e797aee598f49a9ba0b4fe400d6cc41e2cf069bf599723ef587af` |
| `vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py` | 新增用户候选：`9a2b75e9989a4fdb3dcd09094250f82a39d8cc9956c6773481e3c9cf969c0024` |
| `vllm_omni/diffusion/distributed/hsdp.py` | 保持原文：`7eb25d685e9ef1d50b42c06530b065ba7a4601c5cd78f213ad7ec559e001ed0a` |

pipeline 的原文 SHA256 为 `e8d37d9e086b72930d33b419efadbf25ff695bcef295cba8f3c0c2b59edc85e0`，最小 diff SHA256 为 `fb25c7b745528f0bb7a76a005b3e34ad291f2c9c18f1e21bf11c38436027f18d`。新文件与用户的 `pipeline_minimax_h3.candidate.py` 完全一致，不附带其他模型行为修改。

`source-patches.lock.json`、安装验证和完整源码清单把 pipeline 纳入 `changed_sources`，Omni RECORD 同步记录新字节及长度；CI 发布与本地导入器也核验这三份变更源码。不能通过删除记录、改环境变量或沿用旧 pipeline 哈希让检查通过。Torch 使用上一轮修复后的跨 Python 规范指纹，保留精确版本约束。

## 本地与镜像测试

候选的 CPU 控制流测试已移植到维护配方，以仓库内的相对 vendor 路径读取原始源码，不再依赖交接机 `D:\AI` 的绝对位置。原候选生成器的输出保护测试改为验证纯转换不写回原件，实际安装的 create-once 保护由独立安装测试覆盖；新增 DiT-only 选择和“去噪成功但退出卸载失败时仍须抛错”，形成15项候选契约。安装测试使用可写临时目录，避免最终镜像的 `dynamo` 用户写 `/opt`。

本次实际完成的本地检查：

- Python 3.12 和 3.14 各执行50项定向测试，全部通过；包含来源指纹、DiT入口、模拟安装/RECORD以及GPU脚本控制流。
- `01-check.cmd` 完整通过：341项回归中实际执行325项，16项因本机没有 Torch 跳过；22个 Python 文件及4段工作流脚本检查通过。
- 完整证据：`.h3-experimental-evidence/hsdp/local/checks-20261028T034252.353015Z/summary.json`。这不是安装后的真实 Linux/Torch 验证；安装模块的CPU测试仍会在最终镜像中运行，CI不接受其失败或跳过。

之后只补充本文检查结果并重新封存，代码未再改变。未自动提交、推送、构建或操作集群。

更新的 `gpu_roundtrip.py` 保留上一版 warm 迁移检查，并增加：

- 独立冷态嵌套 FSDP fixture，在首次 FSDP forward 前启用初始 CPU parking，不能用隐藏 warmup 提前初始化。
- 绑定最终镜像实际安装的 `MiniMaxH3Pipeline.diffuse()` 和组件上下文；每轮50步，至少三轮。真实 FSDP/TP8 forward 使用合成小模型和输入，去噪数值循环与 Cache-DiT refresh 为替身。
- 检查实际参数身份、DTensor/local device、FSDP cached storage、别名、循环引用、输出数值及搬运计数，要求每轮仅一次 DiT H2D，步间不重复搬权重。
- 在脚本内建立 TP8 `VllmConfig`，贯穿构造、forward 和销毁。正向测试不复用旧入口失败后的 FSDP 实例。

GPU 脚本准备完成不代表已经运行。若冷态检查失败，应保留每个 rank 的第一处失败和状态，继续定位，不能移除保护或强行提交真实 C02。旧 warm PASS 也不能替代新 cold PASS。

## Windows CMD 构建命令

直接提交本次已封存的新配方：

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\h3-hsdp-offload\02-submit-build.cmd
```

该入口先执行本地检查，再提交和推送；新的 push 触发 `h3-hsdp-offload` 作业。仍使用原分支，不创建新分支。无需先单独执行 `01-check.cmd`；若只想本地复核，可单独运行 `01`。本次有配方变更，不能仅用 `03-run-build.cmd` 重跑旧提交。

Actions 地址：<https://github.com/githubgxll/dynamo/actions/workflows/dingo-router-ci.yml>。

新镜像标签仍为：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm:omni031rc1-gateway-hsdp-offload-<提交前12位>
```

构建发布成功后，下载本次 `image-evidence-h3-hsdp-offload-<完整提交>-<attempt>` ZIP 并导入：

```bat
call delivery\h3-hsdp-offload\04-import-image-evidence.cmd "C:\Users\admin\Downloads\image-evidence-h3-hsdp-offload.zip"
```

部署镜像取导入产物 `deployment-image.txt` 的完整 `repo@sha256:...`，不能使用 ZIP 哈希代替镜像 digest。发布只代表镜像 CPU/来源/依赖检查及发布拉回核验完成，`gpu_validation_status` 仍是 `NOT_RUN`。

## 新镜像部署交接

发布后另行派生新的实验包、ConfigMap、Pod 与 RUN。保留 `ref8-a-1028-b` 的失败 ZIP、日志、blocked 状态和旧 Pod，不手改旧运行记录，不重复生成已经有完整证据的失败请求。

新包须根据本次发布证据同步以下身份与来源：

1. 新完整镜像 digest、本次完整 `source_commit`；`H3_HSDP_SOURCE_COMMIT` 对应本次提交，不能继续写 `4a5d1340…`。`H3_HSDP_PARENT_IMAGE` 仍是上文 `873650…`。
2. 保留父层 `H3_EXPERIMENTAL_SOURCE_COMMIT=2362a3241c03a286b98e51b8003a51f494fab4f1` 与 `H3_C14_SOURCE_COMMIT=0b11a41b083facf74d3387696a72893d0d3e40e1`；不要在 Pod env 伪造这些镜像标记。
3. 在 `image-repair-lock.json` 和实际运行器 `SOURCE_HASHES` 中核验 backend、helper、新 pipeline 三文件；distributed/hsdp 原哈希继续保留。核对实际 import 路径，确认不存在覆盖旧源码的挂载/PYTHONPATH。
4. 从新镜像/发布证据读取 `runtime.py`、`gpu_roundtrip.py`、`source-patches.lock.json`、`inputs/source-lock.json`、`inputs/input-seal.json` 等 repair-files 哈希。它们可能已改变，不能复制旧 `18e218…` 的锁值。可调用 `/opt/h3-hsdp-offload/runtime.py` 的 `verify_installed()` 完整核验。
5. 旧 ConfigMap `gpu_roundtrip_context.py` 同时锁定旧镜像脚本哈希。新脚本已内置 TP8 VllmConfig，上述入口应重新审核并更新为新脚本调用；不能删除旧 hash 检查来强行兼容。
6. 旧实验包 `repair_gate.py::PHASES` 精确匹配17个阶段。新脚本在 `remove-hooks` 与 `destroy-vllm-fixture-model-parallel` 之间加入 `cold-installed-h3-diffuse-before-native-fsdp-prehook`，三轮时总计18个阶段。新部署包必须同步阶段清单，并逐rank核验 `cold_pipeline_entry.status=PASS_SYNTHETIC_INSTALLED_DIFFUSE_ENTRY_ONLY`、新 pipeline 哈希、`cycles=3`、`forwards=150`、`step_component_h2d_transitions=0`。不能保留只接受旧17阶段的判断，也不能仅看总PASS而忽略新增冷态证据。直接调用镜像脚本后，旧 `harness_context` 对wrapper哈希的记录也应改为实际使用的入口身份。

八卡测试在独立新 Pod、没有已加载 H3/SR 的条件下运行；每次使用新输出目录：

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  /opt/h3-hsdp-offload/gpu_roundtrip.py \
  --output-dir /tmp/h3-hsdp-gpu-gate --cycles 3
```

归档所有 rank 与 summary 后，再执行真实 Ref2VA C02 cold 和同输入 repeat，确认完整50步、VAE、可解码音视频，再提交完整 Ref7。实际 FL2VA、无卸载 baseline、仅 TE 卸载也需各自回归；同事 Gateway 参数透传和生产并发验收另行交接。原生 Omni 的合成测试不证明 Gateway 已接入 A 策略。

本轮构建配方不修改实验 Pod、旧控制器或运行状态；新部署包需要以上独立交接，不能只将旧 YAML 中的 image 换成新标签。
