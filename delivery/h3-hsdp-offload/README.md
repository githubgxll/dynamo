# H3 八卡 HSDP CPU 卸载候选镜像

结论：这次需要修改 Omni 的卸载实现。交接包记录的是 CPU/CUDA 参数存储设备不一致，发生于文本编码前切换组件，不是 OOM。这里从已经成功发布的 C14 轻量镜像生成独立候选镜像，保留 vLLM 0.31.0、vLLM-Omni 0.31.0rc1、PyTorch 2.13.0+cu130、Cache-DiT 1.5.0、现有 video-gateway 与 SeedVR2 补丁；不重新安装依赖或下载权重。

**构建成功只代表候选镜像通过 CPU、来源和依赖保留检查。实际 HSDP 卸载、八卡 C02 生成、显存与质量验收尚未完成，发布证据始终记录 `gpu_validation_status=NOT_RUN`。**

2026-10-27 构建中的 `Native Torch method differs` 已定位为 Python 3.12/3.14 的 AST 文本表示差异；本次改用跨版本规范指纹，保持 Torch、依赖和迁移 helper。见 [根因与修复说明](TORCH-CONTRACT-FIX-20261027.md)。请用 `02-submit-build.cmd` 提交新配方，单独运行 `03-run-build.cmd` 不会上传本次修改。

2026-10-27 已按新版 skill 复核并实时核对远端分支：Gateway 仍为本镜像锁定的 `9ccee2f…`，本次无需升级 Dynamo 基础。较新的 `DingoRouter-base` 已到 ai-dingo 1.6.0，但与 Gateway 分叉，且仍没有配置 A 的 Dingo CLI 参数链。完整结论、新实验身份清单与服务接入边界见 [构建前复核](BASE-AND-SKILL-REVIEW-20261027.md)。

**本配方针对原生 Omni 实验。** 同事通过 Dingo Gateway 使用配置 A 时，仍需单独增加 `diffusion_offload_config` 的解析、dataclass 与 handler 透传；本次底层迁移补丁不包含这一服务适配。

父镜像固定为：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:87365073cbdf95fa17ca352a24a93cb74fde144a47c96135e91011d1a03df72e
```

对应父源码提交 `0b11a41b083facf74d3387696a72893d0d3e40e1`；video-gateway 为 `9ccee2f56d0122be94d7436ae2f5fec96d15ea9b`。本次不自动 rebase 到 main 或变化中的 Gateway 分支。原因是要对已知失败环境进行单变量修复。后续合入同事新版本，应另做差异及依赖审查。

## Windows CMD 操作

沿用现有目录及分支 `Dingorouter-H3-vllm-omni-jirx`，不新建分支：

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\h3-hsdp-offload\01-check.cmd
call delivery\h3-hsdp-offload\02-submit-build.cmd
```

`01-check.cmd` 只做本地检查；`02-submit-build.cmd` 会检查封存输入、提交并推送，触发 Linux Runner 构建并向现有 Harbor 仓库发布候选镜像。构建名称带 `h3-hsdp-offload`、分支、完整提交号。只有该作业运行；不同时触发原 C14、lite 或全量框架镜像。

进度地址：<https://github.com/githubgxll/dynamo/actions/workflows/dingo-router-ci.yml>。

如果提示相同配方已经上传，先查看该完整提交号对应的 Actions。确需再次构建时：

```bat
call delivery\h3-hsdp-offload\03-run-build.cmd
```

这会在同一分支创建明确带标记的空提交，再次触发构建，不依赖网页 `Run workflow` 按钮，不需要修改 main。不要反复运行来重复排队。

成功后下载本次的 `image-evidence-h3-hsdp-offload-<完整提交>-<attempt>` artifact ZIP，导入：

```bat
call delivery\h3-hsdp-offload\04-import-image-evidence.cmd "C:\Users\admin\Downloads\image-evidence-h3-hsdp-offload.zip"
```

导入器验证当前提交、封存配方、父镜像、最终镜像 ID、发布回执及证据哈希。部署引用取 `deployment-image.txt` 中的完整 `repo@sha256:...`，不能用 artifact ZIP 的哈希代替镜像 digest。

候选 tag 格式：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm:omni031rc1-gateway-hsdp-offload-<提交前12位>
```

## 构建检查与证据

1. 从父镜像实际安装目录导出 `sequential_backend.py`、`distributed/hsdp.py`、H3 pipeline 以及 PyTorch FSDP 实现。核对父镜像 ID、版本及三个 Omni 文件的已知 SHA256，先保存 `parent-source/` 和 `parent-source.json` 再修改。
2. 不访问 PyPI、Hugging Face 或 Docker Hub 前端；只拉现有 Harbor 的固定父 digest。Dockerfile 不使用额外 `# syntax=docker/dockerfile` 前端。
3. 镜像内补丁要求固定源码和支持的 PyTorch 实现。遇到来源不匹配会停止，不能把检查改成忽略，也不能直接替换 DTensor 内部 local tensor 绕过错误。
4. 镜像最终以原 `dynamo` 用户、无网络运行 CPU 验证。CI 再用最终镜像 ID 重跑；保留 Python 包的完整 METADATA/direct_url 和 dpkg 清单，保持镜像层、入口、命令、工作目录及环境配置。
5. 只发布实际测试的本地镜像 ID，发布后按返回的 digest 重新拉取并核对同一镜像 ID。artifact 即使失败也保存已有证据。

全部构建与运行验证脚本位于新目录 `container/h3-hsdp-offload`；现有 C14/lite 源文件保持原样。共享提交工具及既有测试来源锁到父提交。日后修改配方后，应先审查 diff，再显式运行 `python delivery\h3-hsdp-offload\ops.py seal` 更新封存；提交脚本不会擅自重新封存。

**旧 C14 专用诊断入口有意保留原身份锁。** `/opt/seedvr2-c14/sr_only.py` 会校验完整旧 Omni 源码及 RECORD，因此在本候选镜像中会拒绝新增的 HSDP 源码。不要改旧校验以强行继续。已有 C14 诊断实验仍使用原 `873650…` 镜像和旧 Run；常规 SeedVR2 框架/API 的源码与默认关闭的诊断行为保留，但这不等于所有旧诊断脚本都可直接在新镜像运行。本次 H3 实验运行器需显式更新镜像 digest 及源码锁。

## GPU 验收顺序

新镜像就绪后，使用新 Pod 与新 RUN；旧失败 Run、日志、state 与下载包保持不变。当前运行控制器在首次失败后 `generation_blocked=true`，`/health=200` 或 Pod Ready 不能证明生成可继续。不要修改旧 state 文件或复用已阻断的进程。

新实验包必须保留父层 `H3_EXPERIMENTAL_SOURCE_COMMIT` / `H3_C14_SOURCE_COMMIT`，并核对新增 `H3_HSDP_SOURCE_COMMIT` / `H3_HSDP_PARENT_IMAGE`，同时检查实际加载的 `sequential_backend.py` 和 `hsdp_migration.py`。旧控制器的来源清单未包含这两个文件，不能只替换旧 YAML 的 image。具体值和来源见 [新实验身份清单](BASE-AND-SKILL-REVIEW-20261027.md#3-新实验包必须采用的身份与来源检查)。

1. 在相同 PyTorch/GPU 环境先做真实多卡参数往返、forward、reshard、再次卸载验证。必须核对 Parameter 身份、FSDP 持有引用、global/local device、全 rank 完成，不能只凭 mock 或 CPU 测试。
2. 新 Run 先执行固定输入的 C02 cold，再 warm、连续重复。保持 8×H100、U8、HSDP8、TE TP8、BF16、Base50 与 Cache-DiT，SR 首先仍关闭。原权重只读挂载不改。
3. 记录各阶段峰值显存、CPU 内存、卸载/恢复耗时、完整生成与可解码音视频。确认文本编码前 DiT 在 CPU、去噪阶段 DiT 一直驻 GPU；不要把逐层搬运冒充该策略。
4. 同一新镜像跑关闭卸载的同输入 baseline，才能比较节省多少显存及增加多少延迟。失败 HTTP 请求的耗时不能当生成性能数据。
5. C02 重复生成和异常后状态测试通过，再提交 Ref7；SR 与 H3 共卡的联动另行验收。由实验会话更新其镜像身份与源码锁，不能只手改冻结 Pod YAML。

第一项真实多卡迁移验收的镜像内命令如下。在独立的新八卡 Pod 内执行，确认其中没有已经加载的 H3/SR 进程，每次使用全新的输出目录：

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  /opt/h3-hsdp-offload/gpu_roundtrip.py \
  --output-dir /tmp/h3-hsdp-gpu-gate --cycles 3
```

该工具先校验镜像实际安装源码和原生 Torch 方法，再检查真实 HSDP 参数迁移。将全部 `rank*.json`、`summary.json` 和标准输出归档到持久证据目录后，再结束 Pod。它不下载权重、不启动 H3，不能代替第二项真实 C02 生成。Windows 提交脚本和 CPU CI 不会自动执行此命令。

本交付没有启动、停止、删除集群 Pod，也没有自动清理 Runner 或构建缓存。

## 已知风险与定位

| 现象 | 处理 |
|---|---|
| HSDP 实际 Torch 内部结构与已审核的实现不同 | 保留 `parent-source/`，停止构建/启用，重新审查；不能通过升级或跳过检查掩盖差异。 |
| GPU 卸载后没有明显显存下降 | 检查通信缓存、未释放的全量参数及 CUDA allocator，区分 allocated/reserved；不代表 CPU/mock 成功即有收益。 |
| CPU 内存峰值或 pinned memory 超额 | 用八卡真实峰值校准 Pod 内存，预留转换/搬运瞬时副本和文本编码内存；不要把 GPU 显存削减视为总内存下降。 |
| 冷启动/每次请求变慢 | 记录 CPU↔GPU 传输、同步、通信等待；该策略用延迟换取峰值显存空间。 |
| `less than 100GiB available after GC` | Runner 启动钩子拒绝任务，交由 Runner 管理员处理磁盘与并发；不能绕过容量检查。 |
| 找不到 Actions 新记录 | 查看提交是否实际发生新 push、是否包含 `[h3-hsdp-offload]`，以及该完整提交号；相同配方已上传不会产生新记录。 |
| `uv` 无权限写入缓存 | 本镜像不重装包；检查验证工具是否仍使用无缓存或可写的临时目录，不能更改生产缓存权限来掩盖检查问题。 |
| 新源文件通过本地检查，但模型路径仍失败 | 本地检查不覆盖 CUDA/FSDP 状态与媒体结果，保留完整首错日志和所有 rank 状态，按阶段拒绝继续。 |

证据基础：用户交接包 `Ref2VA8_HSDP_Offload_Failure_20261027.zip` 及固定父镜像发布 artifact。交接包中的 GPU 失败是已观察事实；候选补丁的 GPU 有效性必须由上述新实验验证。
