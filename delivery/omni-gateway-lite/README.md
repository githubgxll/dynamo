# Omni 0.31 + 最新 Gateway + SeedVR2 修复：轻量版

**结论：使用已经构建成功、尚未加入蒸馏权重的框架镜像，保留完整 vLLM-Omni 0.31.0rc1 默认运行环境，只加入最新 Gateway 运行代码和 SeedVR2 RMSNorm 修复。没有安装、升级或删除 Python/CUDA 依赖，不携带四份适配器，不应用 Ref8 12:3 专用补丁。**

这是一份独立构建配方。沿用 `Dingorouter-H3-vllm-omni-jirx` 分支、原 Harbor 仓库与 Runner。准备脚本本身不代表新镜像已构建或发布，也不修改运行中的 Pod。

## 保留和省略的内容

| 项目 | 轻量版行为 |
|---|---|
| vLLM / vLLM-Omni | 保留 `0.31.0` / `0.31.0rc1`，完整默认依赖与模型代码 |
| SeedVR2 | 只回补 [PR #8767](https://github.com/vllm-project/vllm-omni/pull/8767)，RMSNorm 中间计算升为 FP32，恢复原输出精度 |
| video-gateway | 对齐 2026-10-21 实查的 `9ccee2f56d0122be94d7436ae2f5fec96d15ea9b`；保留此前成功的六个 Omni 0.31 后端适配文件 |
| Torch/CUDA/NIXL、FFmpeg/ffprobe、媒体依赖 | 全部继承，防止再次引入 ABI 或依赖冲突 |
| 监测、Dynamo、原生运行库 | 保留现有可用内容，不再额外开发或裁剪 |
| 四份少步蒸馏权重 | 不打入；没有 `/opt/h3-adapters` |
| Ref8 12:3 修复及六种实验 profile | 不加入；不能拿本镜像替代依赖该修复的 Ref8 部署 |
| H3 Base、SeedVR2 原始/FP16 权重 | 都不打入，继续按需要只读挂载，不重新下载或转换 |
| KVBM、ModelExpress | 沿用框架父镜像的关闭状态 |

“完整 Omni”指该固定发布 wheel 的全部包内容和默认依赖，没有删模型目录或用 `--no-deps` 省略依赖。它不代表所有可选 extra 和所有模型均经过 GPU 验收。

轻量化来自跳过原先约 **5.64 GB 的适配器文件**和蒸馏实验层，不是从最终镜像执行删除。后者无法减掉父层体积。CUDA/Torch 主体仍然较大，最终镜像大小与压缩传输量以 CI 的 `inheritance.json` 和 Harbor 为准。

## 固定的父镜像与更新范围

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:898cdea0a39ba5a673ca03ba10fbd15bfeebaa90ec6fffe6953e89408b5c9a1e
```

这是成功构建提交 `db65d1ec869b6deee02debe7a58e108de7c927ef` 的 framework 镜像。其身份从此前成功最终镜像 `112e3437…` 的已核验 config 和 `image-inputs.json` 恢复，原始证据在本目录 `parent-evidence/`。CI 必须实际拉取完整 digest 并重新检查，不通过浮动 tag 猜测来源。如果 Harbor 已回收这个 digest，构建会停止；应由仓库管理员恢复或重新审查框架构建，不能静默改用其他镜像。

同事最新分支仍采用 vLLM/Omni 0.29；所以本配方**并非直接 FROM 同事的最新 0.29 成品镜像**。它复用成功的 0.31 框架，并同步同事最新 Gateway 源码的有效变更。这样可同时满足新框架与服务适配，避免对旧原生栈直接覆盖新版 wheel。

与旧 Gateway `e8c919…` 相比，最新分支只有两个提交、12 个变更文件：CephFS 配额祖先目录统计，以及 Grafana 部分抓取/空区间修复。运行时代码仅 `dingo/video_gateway/artifact_store.py` 改变；依赖声明、原生库和构建模板无变化。图表、文档和测试的原文已封存在 `container/omni-gateway-lite/gateway/review/`，交给同事按监控发布流程应用；镜像不会自动更新集群 Grafana。

保留此前六个 `dingo/vllm/omni` 对接改动，不能用同事的 0.29 `dingo/` 整包覆盖。新旧源码和包 RECORD 都校验，版本号相同也不能掩盖源码不同。

## Windows CMD 操作

先检查：

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\omni-gateway-lite\01-check.cmd
```

检查通过后，提交并普通推送到原分支，自动启动轻量版构建：

```bat
call delivery\omni-gateway-lite\02-submit-build.cmd
```

这条命令创建普通提交，**推送后只触发 `omni-gateway-lite`，不触发旧补丁版或完整框架版**。不新建分支、不强推、不自动 rebase 同事分支。原有工作区无关改动或远端分叉会被明确拦截。若同一个提交已经上传，脚本不会反复制造构建；会输出提交 SHA 供查找对应记录。

然后到 [GitHub Actions 原工作流](https://github.com/githubgxll/dynamo/actions/workflows/dingo-router-ci.yml) 查看新运行记录。标题包含 `omni-gateway-lite`、分支和完整提交 SHA；每次运行有独立的 run ID，不会被同名记录覆盖。

**2026-10-21 触发方式修正：** 默认分支 `main@049eece…` 没有 `dingo-router-ci.yml`。GitHub 网页的手动触发要求工作流存在于默认分支，因此不能依赖 `Run workflow` 按钮。旧版脚本提交 `62d534c49945` 又带 `[skip ci]`，所以当时只有代码上传，没有构建记录。这是交付脚本的触发设计遗漏，与 Harbor 名称、镜像 tag 重复无关。当前已去掉跳过标记，并让分支推送直接运行轻量版；不用修改同事的 `main`。

仅在已提交的同一份配方需要再次构建时，运行：

```bat
call delivery\omni-gateway-lite\03-run-build.cmd
```

该脚本先检查工作区干净且本地/远端提交一致，再创建一个空提交并普通推送原分支。配方文件不变，但 Git 提交、Actions run 和最终 tag 都会有新身份。每运行一次就是明确请求再构建一次，首次提交后不用接着执行。无需 `gh` 或额外 token。继续复用本机 GitHub 的命令级代理，不把代理写入镜像或 Pod。

构建成功的新标签：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm:omni031rc1-gateway-lite-srnormfix-<Git前12位>
```

部署用 Actions 产出的完整 `repo@sha256:...`，不要使用 ZIP 的哈希代替镜像 digest。

下载 `image-evidence-omni-gateway-lite-<完整提交>-<attempt>` 制品后，可校验导入：

```bat
call delivery\omni-gateway-lite\04-import-image-evidence.cmd "C:\Users\admin\Downloads\image-evidence.zip"
```

## 检查覆盖

本地检查源码封存、原始父镜像证据、补丁字节、错误输入拒绝、提交范围、配置一致性、CI 发布条件以及 Python/Shell 语法。固定源码和模拟包元数据的测试不能冒充真实 Linux 依赖测试。

Linux 构建和最终镜像会检查：

1. 完整父层继承；Python 和 dpkg 清单、原依赖元数据保持不变。
2. Omni 包内所有文件与留存的官方 wheel 一致，唯一允许差异为 RMSNorm；H3 原生 `lora.py` 保持未打 Ref8 补丁的版本。
3. Gateway 全量源码身份、实际导入位置、唯一更新文件及 wheel RECORD；此前六处 Omni 0.31 对接保留。
4. RMSNorm FP16/BF16/FP32 × 有/无 affine 六组数值检查、原样上游六项回归、FP16 全 300 输入的旧零值故障对照。
5. 完整依赖闭包、非 root uv 检查、OpenCV 实际提供者、CPU 媒体和真实 Gateway/Omni 服务导入。
6. 从最终候选镜像以 UID1000、断网再次验证；只发布已经通过的同一个 image ID，随后按完整 digest 回拉核对。

新报告放在 `/opt/omni-gateway-lite/build-info`；父 `/opt/dynamo/build-info` 和 `/legal` 原样保留。没有安装新包，因此本轮保留原依赖清单和法律证据，并附源码修改 NOTICE；不把这说成新的法律审批。

## 已有经验必须继续保留

- 不重新安装 Torch、NCCL、NIXL、LMCache、OpenTelemetry 或媒体组件；原有完整依赖审计仍执行，仅允许既有精确 Torch/NCCL 元数据例外。
- 不启用 `ai-dingo[vllm]` 全量 extra 来“补齐”，其旧 NIXL 固定版本不适合继承的 vLLM 0.31 原生栈。
- 保留 OpenCV 两种同版本发行包的元数据和已验证 headless 实际提供者，不能随便卸载其中之一。
- `uv pip check` 使用可写临时目录与隔离缓存，避免 UID1000 再次访问 root 的 `/opt/uv/cache`。
- 不使用 Docker Hub 的外部 Dockerfile frontend；构建层断网，只需 Runner 能拉取 Harbor 父镜像。
- 冷 Runner 仍需存放完整框架镜像；`/etc/dingo-gc` 的磁盘准入由管理员处理，不清理同事缓存或降低阈值。
- 旧完整版的源码封存是历史快照。继续复跑原版请重跑其原提交对应的 Actions；本次使用新的 `delivery/omni-gateway-lite` 入口，不混用旧提交脚本。

后续更新 Gateway 时先比较相对本次 `9ccee2…` 的代码和依赖变化，再更新锁和测试；不要把“最新”写成自动追踪的 `main` 或 `latest`。这版不需要为追随同事 0.29 镜像而降级 Omni。

构建成功后还需独立验证 Gateway 的 NATS/ETCD/存储链路、H100 推理和同原片 SeedVR2 1088p/1440p raw/final 对照。FP32 中间计算可能改变峰值显存。新实验必须更新镜像和源码锁；不要沿用要求四适配器、Ref8 12:3 或旧 nadit 哈希的部署预检。
