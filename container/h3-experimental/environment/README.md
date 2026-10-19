# H3 实验依赖安装与复验

结论：依赖在同事的 Gateway 框架镜像构建阶段一次安装完整，通过依赖与 CPU 媒体检查后再运行框架原有许可证检查。四适配器镜像只增加权重和 Ref8 源码补丁，并离线复验；Pod 启动不安装任何包。

基础源码固定为 `DingoRouter-video-gateway@e8c9195672f761ec09344a641730844c9e6c9e99`，保留 `ai-dingo==1.3.0`、`ai-dingo-runtime==1.3.0` 和服务代码。vLLM 升级至 0.31.0，Omni 升级至固定 SHA256 的 0.31.0rc1 wheel。这里不再使用已放弃的 main 1.6.0/aisimulate 依赖方案。

## Linux 构建阶段的调用

在 Dockerfile 的 root 阶段、安装本仓库 ai_dingo/GMS wheels 后调用：

```bash
export DYNAMO_GATEWAY_SOURCE_SHA=e8c9195672f761ec09344a641730844c9e6c9e99
export DYNAMO_VLLM_BASE_IMAGE=vllm/vllm-openai:v0.31.0-ubuntu2404@sha256:3fb023cc987cc7c2bca7ca14c1e0aff5ae06bd9431bf07773731ce7e8a682460
bash /opt/h3-experimental/environment/install_environment.sh
```

目录可整体 COPY 或作为只读 BuildKit bind mount。安装器不要求 Omni 已存在。必须先安装 ai-dingo/runtime、GMS，并继承上述官方 vLLM 镜像。这个调用取代旧的只安装 Omni 的步骤，不能重复调用旧安装器。

最终 `USER dynamo` 下再执行：

```bash
python3 /opt/h3-experimental/environment/audit_environment.py --output-dir /opt/h3-experimental/build-info
python3 /opt/h3-experimental/environment/media_smoke.py --output-dir /opt/h3-experimental/build-info
python3 /opt/h3-experimental/environment/service_smoke.py --output-dir /opt/h3-experimental/build-info
```

构建端可用 BuildKit secrets `pip-index-url`、`uv-default-index` 传包索引。脚本读取后不打印凭据；这些设置不会修复 BuildKit 的 FROM 镜像拉取。没有新增代理或关闭 TLS。

## 依赖边界

- 完整求解并审计 ai-dingo 默认依赖和 **video-gateway extra**、ai-dingo-runtime、GMS、vLLM、Omni 默认依赖，以及递归要求的 extras。kubernetes、zstandard、pydantic 不会因 `--no-deps` 留下缺口。
- 同一次求解显式加入底图中所有已安装 `opentelemetry-*` 包名，不固定旧版本。`uv pip install` 不会自动修复未进入本次依赖图的遗留包；例如 common0.66b0 要求 SDK1.45.x，却可能在旧安装流程后与 SDK1.40.0 并存。保留监测组件，联合选择兼容版本，最终全局审计仍检查所有已安装依赖。
- 保留同事 `av>=14,<17` 约束，选择 `av==16.1.0`，也满足 Omni 的 `av>=14`。不因旧 main 配方使用 AV18 而放宽同事约束。Linux CPython3.12 wheel 已确认可用；实际编码兼容性由构建时真实媒体检查验证。
- Transformers5.14.1/tokenizers0.22.2 联合求解。冻结实际 GPU 依赖和所有已安装 nvidia-* 版本。
- 不整体激活 `ai-dingo[vllm]`：其 NIXL 固定版本不是本次核验的官方 vLLM0.31 native 组合。必要的 uvloop/blake3/soundfile/librosa/ray 辅助依赖单独验证。pyproject 的 vLLM 版本同步为0.31，但没有凭空替换 NIXL 固定版本。
- 唯一例外是官方 Torch2.13.0+cu130 对 NCCL2.29.7 的元数据声明与官方实际2.30.7不一致；同时验证官方镜像 digest、overrides 文件及安装前快照。**PyNvVideoCodec 保持官方2.0.4，没有旧 main 配方2.2.3例外。**
- KVBM、ModelExpress 维持关闭。框架许可证检查继续执行；依赖审计不能代替许可证检查。

两个 OpenCV 上游默认依赖分别要求 normal/headless，两者保持4.13.0.92，最后重装 headless 并验证 cv2 RECORD 和实际 import。不能在 Pod 中随意卸载其中一个。

## 输出与限制

`environment-check.json`、`dependency-audit.json`、`uv-pip-check.txt` 记录真实安装结果；`installed-environment.json`、`omni-wheel-origin.json` 记录版本、元数据路径和 wheel hash。

`telemetry-solve-inputs.json` 与 `inherited-telemetry-requirements.txt` 记录安装前监测包及参与联合求解的包名。实际最终版本以安装后清单为准；未使用全量 `--upgrade`，GPU 保护约束不变。

`prepare_omni_wheel.py` 先下载固定官方 wheel 并实算 SHA256，将约8MiB文件保留在 `/opt/dynamo/build-info/omni-wheels/`，写入 `omni-wheel-download.json`，再把本地 wheel 交给同一次联合安装。已实际复现 uv0.8.22 的 `direct_url.json` 中 `archive_info` 为空，因此不能单靠该字段判断 wheel 身份；离线审计同时核对原始官方 URL、锁定哈希、实际保留文件、大小、路径和安装器 file URI。缺哈希的远端来源或被修改的本地文件仍拒绝。不要删除该保留文件，否则后续离线复验会失败。

GPU 栈保护、Omni 来源、OpenCV 提供者分别记录状态。一项失败不再把另一项误记为 FAIL；`uv pip check` 失败时同时打印完整输出。

`media-smoke.json` 验证真实 CPU H.264/AAC 复用、PyAV/OpenCV/TorchCodec 解码，也执行同事 `VideoEncoder` 到 Omni0.31 的实际编码调用。FFmpeg/ffprobe 继承同事 runtime 的上游完整媒体栈，不重复 apt 安装或替换二进制。

`service-smoke.json` 验证安装包的 Gateway、Omni worker、监控及适配器模块真实导入，以及 Gateway `--help`。检查禁用出站 socket，且不启动 NATS、ETCD、Gateway 监听、Omni engine 或模型。
同时导入 OpenTelemetry SDK 与 gRPC/HTTP OTLP trace exporter；只检查可导入性，不实例化 exporter，也不连接监测后端。

`dependency-license-inventory.json` 仅记录 Python 声明许可证和安装前后差异，状态为 `METADATA_INVENTORY_ONLY_NOT_LEGAL_CLEARANCE`；框架 policy gate 独立决定构建是否允许。

Windows 单元回归只验证拒绝逻辑，不证明 Linux 构建、Harbor、完整 Gateway 服务、H100 或模型生成。四蒸馏模式当前使用实验 profiles；同事公共 Gateway 仍拒绝请求级 lora 且固定12/3，不能因为适配器进入镜像就声称 Gateway 接通全部蒸馏模式。
