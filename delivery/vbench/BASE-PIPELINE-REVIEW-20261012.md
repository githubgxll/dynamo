# VBench基础镜像流程复查与修复（2026-10-12）

本次从源码版本、基础镜像身份、CI执行顺序、版本锁和失败证据五处重新检查。已修复的共同问题是：构建本来要使用固定digest，却仍在每次prepare时依赖Buildx客户端在线解析tag；这个前置请求失败后，daemon预拉取无法执行，而且真正错误被隐藏。当前采用经过独立核验、纳入版本控制的基础镜像pin，后续仍必须实际拉取并核验镜像。没有把本机可访问误作Runner网络已经恢复。

## 1. 这份日志能确定什么

用户提供的错误发生在Linux Runner执行 `.github/scripts/vbench_ci.py resolve-base` 时，Dockerfile尚未开始。`docker buildx imagetools inspect --raw nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` 返回1；原代码使用 `check=True, capture_output=True`，异常只显示命令和退出码，未打印stderr。

所以这份日志不能判定是权限、网络、限流、标签消失或Buildx配置故障。也不能把它算成decord、PyTorch或CUDA运行错误。

Git历史确认 `resolve_base()` 自2026-09-30首个VBench提交 `4500a8e5f02ac62b8850648b7d68dc4767d9cea8` 起至本次修复前函数源码相同。最近的decord提交 `dc84fbd164008f2112cb1ed00e56edc69ebd6852` 只修改10个媒体、环境、测试和文档文件，没有修改该helper、工作流或镜像矩阵。这并不免除流程问题：之前新增daemon拉取时，没有同时处理这个在线解析前置依赖，本轮一并纠正。

## 2. 基础镜像和上游版本实查

10月12日本机通过既有 `http://127.0.0.1:17890` 代理访问Docker官方registry，tag索引、选中的amd64清单及其config均返回HTTP200，内容哈希与描述符一致：

| 字段 | 已核实的值 |
| --- | --- |
| 官方tag | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| 目标平台 | `linux/amd64` |
| tag多平台索引digest | `sha256:f4d8e1264366940438f0353da6f289c7bef069d993d111f8106086ccd18c4a30` |
| 本次实际固定的amd64 digest | `sha256:810756cab1c28ce693499a5c2ebb66f6d10a61d026998c8606bad449643a4c49` |
| image config digest | `sha256:02f0c5f1a54bd88a5242a21ef690ab6826c7d36eb2b8134b32860a258427d97e` |

这与此前已经构建到依赖安装层时使用的amd64镜像相同，没有换成猜测的新digest。本次官方检查只读取元数据，没有下载全部镜像层，也没有检查Runner出口。

| 来源 | 本次观察到的提交 | 对本故障的影响 |
| --- | --- | --- |
| VBench官方master/HEAD | `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490` | 与本项目固定提交相同；没有更近的默认分支提交可用来修复此次拉取错误 |
| 本项目DingoRouter-base共同基线 | `77a1716960867bf1aba0af53c1993aa352506e9e` | 保留作为本轮打包基线 |
| 最新DingoRouter-base | `badd16ffb033209893892af4fa0b9e3ed381f139` | 比基线新增10个提交、共10文件改变，未增加VBench或registry故障修复 |
| 最新main | `049eece3f32f8d38dc1d8c52d4b793cb30e394bb` | Dynamo为1.6.0，但没有组内dingo-router工作流、镜像矩阵和VBench helper |

最新base仍为Dynamo1.3.0；工作流和matrix脚本与共同基线一致。其构建配置变化只把默认构建切换到SGLang，版本从 `glm-5.3-flash` 改为 `v0.5.21`；其余修改是Anthropic请求兼容和SGLang统计等实现、测试与文档。直接套用该配置会改变当前专用VBench矩阵。因此本轮不rebase，不迁移到main；这是基于完整10提交差异作出的范围选择，不是忽略已有必需修复。

官方VBench当前仍声明 `numpy<2.0.0`、`timm>=0.9,<=1.0.12` 等旧栈边界，源码安装配置列出的Torch CUDA范围至12.1。本项目采用经过单独审查的六维依赖集合：pyIQA0.1.13要求Transformers4.37.2，与官方全部指标requirements中的4.33.2不同，不能宣称直接安装了原版全量requirements。完整组合仍需CPU模型和H100评分验收。10月9日各直接依赖的新旧版本及互相约束见 [依赖复查](DEPENDENCY-AUDIT-20261009.md)；升级整栈需要另立镜像协议并让baseline和优化视频在同一新环境重测。

## 3. 当前基础镜像流程

| 环节 | 当前行为与判定 |
| --- | --- |
| 版本控制输入 | `container/vbench/base-image.json` 记录已核验的source_tag、amd64 digest、平台和来源证据；自动纳入构建输入指纹 |
| prepare基础锁 | `prepare-base` 纯离线校验该pin并生成base.lock、GITHUB_OUTPUT和准备证据；不调用Docker，不解析tag。`PASS_PIN_VALIDATION_ONLY` 不表示镜像已拉取 |
| 实际获取 | `pull-base` 默认由Docker daemon按同一digest执行 `docker pull --platform linux/amd64`，完成后校验OS、架构和RepoDigests |
| 可选诊断 | 只有维护者显式传入 `--probe-client`，才额外运行Buildx客户端对比；常规构建不依赖此路径，`client_probe.category=not_requested` 是正常状态 |
| Docker构建 | 预拉取和身份检查成功才进入原VBench构建；继续传递固定digest、使用 `pull:false`。这允许复用本daemon内容，不保证BuildKit绝不访问registry |
| 锁的使用 | 本机check、准备制品导入、切换publish及CI发布检查均核对base.lock与当前pin一致；改pin或其他输入后须重新prepare |
| 失败证据 | 输出失败命令的脱敏正文和退出码，保存完整脱敏文件；workflow记录基础准备/锁校验/拉取的独立outcome，失败时仍尝试收集已有文件 |

daemon已有的有限重试保持：仅EOF、超时、连接重置、限流及部分服务临时错误最多3次，每次最多600秒，间隔10秒。明确拒绝、镜像不存在、证书错误或其他不属于临时错误的失败不循环重试。重试不能修复固定出口问题。

没有新增代理、关闭TLS、自动替换为Harbor源或删除依赖检查。若以后要改基础镜像，必须重新核实来源、平台、兼容性和digest，提交新的pin，再重新生成所有准备证据；不能仅编辑导入的base.lock。

## 4. 操作位置和下一次日志

在本机 **Windows CMD** 检查：

```cmd
cd /d D:\AI\dynamo-vbench
call delivery\vbench\01-check.cmd
```

看到 `PASS_LOCAL_ONLY` 后再执行：

```cmd
call delivery\vbench\02-submit-prepare.cmd
```

02会在自己的VBench分支提交、推送并触发组内Linux Runner。当前仍为prepare，不上传Harbor；无需登录K8s、SSH或申请GPU。必须触发含新代码的运行，重跑旧提交仍会执行旧 `resolve-base`。

新日志应先出现 `Using reviewed base pin: ...; no tag lookup`，然后出现 `Pulling locked base through Docker daemon`。基础拉取通过时会报告 `Base pull PASS: verified_locked_image`。若下一步仍失败，先按实际失败步骤定位，不把前一步PASS扩展为完整镜像已通过。

准备制品 `vbench-prepare-<SHA>-<attempt>` 中重点保留：

- `workflow-status.json`：包括 `BASE_PREPARE_OUTCOME`、`BASE_LOCKS_OUTCOME` 和 `BASE_PULL_OUTCOME`。prepare时发布锁校验为skipped属正常。
- `base-prepare/base-prepare-summary.json`、`base-prepare/base-definition.json`：pin校验、输入定义和是否使用网络。
- `base-pull/base-pull-summary.json`、`base-pull/daemon-pull-*.stderr.txt` 及对应stdout：实际拉取尝试和脱敏错误。
- `base-pull/daemon-image-identity.json`：只有执行到身份检查才存在；核对平台和匹配digest。

失败制品可能不完整，不能导入作为发布依据。若Runner被终止或制品上传也失败，Actions中输出的脱敏正文仍是必须保存的证据。

## 5. 何时找同事，提供什么

共同提供：失败job链接、源码SHA、实际Runner标识、最先失败的步骤、精确镜像digest、上述summary和脱敏错误正文。不要提供密码、令牌、Docker配置全文或带签名的URL。

| 证据 | 需要同事确认的内容 |
| --- | --- |
| pin离线校验失败 | 先核查提交中的base-image.json和锁；属于项目配置问题，无需先改集群代理 |
| daemon持续EOF/超时 | 这个Runner Docker服务实际出口、代理/NO_PROXY、DNS，以及registry、认证端点和实际镜像层CDN连接；本机17890代理不能代替该证据 |
| unauthorized/denied/401/403 | 针对报错中的Docker Hub或代理确认认证/授权及失效的已有凭据；Harbor管理员身份不等于Docker Hub身份 |
| 429/toomanyrequests | Runner所用Docker Hub访问身份和限流状态；由维护者决定已有账号或已批准缓存的使用方式 |
| x509/证书错误 | Docker服务信任链和代理证书配置；不通过关闭TLS验证绕过 |
| 拉取和身份通过，FROM失败 | BuildKit所用daemon/driver、解析和本地内容可见性；不要立即更换CUDA版本 |
| FROM通过，pip或资产仍失败 | BuildKit RUN内的实际出口、代理/NO_PROXY及对应索引或资产域名；与daemon拉镜像是不同网络路径 |

若维护者提出内部基础镜像缓存，请其给出确切registry/repository、可核验的上游来源与digest、可用认证方式及证书要求，再安排受控变更。当前没有擅自推断Nexus仓库或建立Harbor镜像。

## 6. 验证范围和仍有的问题

本次已完成官方清单/config哈希和平台核验、Dynamo最新分支完整差异审查、VBench官方默认分支比对。最终本地完整回归为 **114 passed、2 skipped、83 subtests passed**；两项跳过为Windows真实符号链接权限相关测试。本机检查证据为 `D:\AI\_workspace\VBench_Image_20260930\local-check-20261012T034846315045Z.json`。

此外，在本机WSL Docker的Ubuntu22.04/Python3.10隔离容器内，以 `--network none` 实际执行新的 `prepare-base` CLI；容器未挂载Docker socket，该步骤未调用Docker，结果为 `PASS_PIN_VALIDATION_ONLY`、`network_used=false`、`runtime_pull_verified=false`。证据为 `D:\AI\_workspace\VBench_Image_20260930\diagnostics-20261012-base\offline-prepare-cli\base-prepare\base-prepare-summary.json`。这验证了基础锁准备本身可以离线完成，**并非Runner已经拉取镜像**。

前一次已完成的Linux CPU decord解码验证继续有效，其范围仍是16包CPU媒体链路，不能证明完整106包CUDA环境或全部六维模型正常。

尚待实际验收及维护问题：

1. Runner实际按digest拉取与后续BuildKit网络仍未在本次修复后验收；固定pin消除多余解析路径，不会修复出网连接。
2. 当前基础镜像config记录创建于2023-11-10；旧CUDA、系统库和解码器存在维护与漏洞风险，尚未完成漏洞扫描。固定digest是追溯手段，不是安全更新。后续应单独评估维护中的完整基础栈并做同协议回归。
3. 七个checkpoint的许可仍为UNKNOWN，AMT/pyIQA等适用授权尚未记录；发布策略会继续阻止未完成审查的发布，Harbor管理员权限不会改变许可结论。
4. 完整依赖检查、六维CPU禁网加载、Harbor发布/拉取和H100原始MP4评分仍是独立检查点；正式36段对比尚未完成。
5. APT只保存安装版本记录，未固定历史软件源快照；整个镜像不承诺逐字节重建。

本次诊断证据目录为 `D:\AI\_workspace\VBench_Image_20260930\diagnostics-20261012-base`：`cuda121-official-registry-evidence.json`、`upstream-review.json`、`base-compare.json`、`upstream-files.json` 和 `modernization-review.json`。版本与网络观察均带本次时点；后续复用前应按对应风险重新核验。
