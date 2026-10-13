# MiniMax-H3 distilled：Omni 0.30 离线适配器镜像

本包从已经成功发布的 H3 0.30 镜像派生，仅加入三份 LightX2V Turbo 和一份 FastH3 Dense 适配器。包内提供构建、发布、证据保存脚本。准备检查通过不代表已经构建、发布或完成 GPU 验收。

用户已确认：“同目录”指 Harbor 中与 H3 Base 使用同一个仓库。适配器在容器内使用 `/opt/h3-adapters`，因此无需调整现有 `/userdata/mengsz` Base 只读挂载，也不会被该挂载遮住。

## 1. 固定输入和发布目标

| 项目 | 本包配置 |
| --- | --- |
| 本地独立仓库 | `D:\AI\dynamo-h3-distilled` |
| 本地分支 | `h3/distilled-omni030` |
| 成功构建配方基线 | `958c2ce95cd361fd5a6770e3023eac4de968f674` |
| vLLM / vLLM-Omni | 都为 `0.30.0`，不重新安装 |
| 平台 | `linux/amd64` |
| Harbor 仓库 | `registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm` |
| 镜像标签 | `v0.30.0-ubuntu2404-distilled-<源码SHA前12位>` |
| 推送触发分支 | `DingoRouter-h3-distilled-20261013` |
| 构建 Runner | `[self-hosted, linux, x64, dingo]` |

固定底图：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:aace31a2806c2c18535cb0bc6a81cc940e0901b4f02f380d2d880c1f34f965c2
```

这是已完成依赖修复、媒体检查和软件许可扫描的运行镜像。新镜像不重新执行 pip/apt/dnf，不编译 native builder，不更改 KVBM/ModelExpress、NCCL、FFmpeg 或旧 overlay 配置。原 `DYNAMO_COMMIT_SHA` 仍表示安装的框架源码；新打包提交另记为 `H3_DISTILLED_SOURCE_COMMIT`。

## 2. 四份权重如何进入镜像

文件的源地址、revision、精确字节数、SHA256 与 safetensors 文件头合同在 [assets.lock.json](../../container/h3-distilled/assets.lock.json)。四份合计 5,636,659,576 字节，约 5.64GB；实际镜像压缩层大小由构建结果决定。

```text
/opt/h3-adapters/
├── lightx2v/3ec17a324ced54151364f24f8b5fb6bf7e26414f/
│   ├── minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
│   ├── minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors
│   ├── minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors
│   ├── README.md
│   └── Apache-2.0.txt
├── fasth3/f509e629374cac104e7f62daecce6d1488a3041d/
│   ├── dense-datafree/adapter_model.safetensors
│   ├── README.md
│   ├── adapter_manifest.json
│   └── LICENSE
├── legal/MiniMax-H3-LICENSE
└── NOTICE
```

实际下载在 Linux Runner 的构建阶段完成。下载器固定官方 HTTPS 来源，保留 TLS 校验，不静默切换镜像站。只有实际传入构建容器的代理环境才生效；Runner shell 的 HTTP_PROXY/HTTPS_PROXY 不会自动传入 Docker RUN。本包不加入此前失败的 hd04 代理，也不把本机17890代理传给 Runner。下载失败时重试次数有限；支持校验响应范围后的续传，缓存命中也重新校验完整 SHA256。新镜像只复制最终验过的文件，不复制缓存、临时文件或认证配置。

下载阶段允许外网；最终校验阶段使用 `RUN --network=none`。K8s 只需拉取 Harbor 镜像，不需要下载适配器，不需要你写节点或 Base 目录。H3 Base 完整权重、视频、结果和凭据均不打包。

## 3. 本机 Windows CMD：先检查

以下都在本机 Windows CMD 执行；即使窗口由 TSM 打开，`D:\AI\...>` 也不是 Pod 内部。

```cmd
cd /d D:\AI\dynamo-h3-distilled
call delivery\h3-distilled\01-check.cmd
```

预期 `PASS_LOCAL_PREPARATION_ONLY`。检查包括成功配方保持不变、53 个官方 Omni 源码哈希、资产与许可文档清单、下载恢复与错误拒绝测试、专用工作流路由和 Shell 语法、发布证据工具。详细记录写入：

```text
D:\AI\_workspace\H3_Distilled_Image_20261013\checks\<run-id>\
```

本机不要求 Docker Desktop、WSL、GPU 或 Linux 镜像环境。复用此前已存在的工具 Python；脚本不会临时安装环境。

## 4. 本机 Windows CMD：触发构建和发布

确认检查通过后，由你执行：

```cmd
cd /d D:\AI\dynamo-h3-distilled
git diff --stat
git status --short
call delivery\h3-distilled\02-submit-build.cmd
```

**02 会实际 commit 和 push，成功推送即触发组内 Runner 构建，并在校验通过后发布 Harbor。**

脚本限定本地分支、origin、远端分支和允许提交的文件，拒绝权重等大文件进入 Git；先检查再提交，提交后再检查工作树，最后普通 push。不会 force push，不改 main/base 分支。推送失败可在解决账户权限或网络后重跑，不必手工重复 commit。

检查会使用独立临时Git索引模拟提交，覆盖尚未跟踪的新文件，并核对Git中七份上游文档的原始SHA256；不会改动你的实际暂存区。两份上游MiniMax许可证自带行尾空格，因此只对这两份原文的行尾空格设置Git格式例外，原文和哈希均不改动，其余文件仍严格检查。若旧脚本在`git diff --cached --check`处因此停止，直接重跑修复后的02脚本即可，无需reset或删除文件。`CRLF will be replaced by LF`是换行提示，本身不是失败。

本机 Git 继续使用命令级 `http://127.0.0.1:17890` 代理。它不写入镜像或 Runner 配置。沿用仓库已有 `REGISTRY_USERNAME` / `REGISTRY_PASSWORD` secrets，无需在聊天或脚本中填写真实凭据。

打开 [GitHub Actions](https://github.com/githubgxll/dynamo/actions)，选择新分支、新提交对应的 `Dingo runtime images`。

这次增加的是同一个工作流内的 `h3-distilled` 专用任务。蒸馏分支不会同时启动原框架完整编译任务；如手工运行工作流，明确选择该分支和 `h3-distilled`。不要在其他任务中选择 all 代替这个目标。

## 5. Linux Runner 自动完成什么

1. 校验 Harbor 目标、固定底图与标签，使用现有登录和 Docker driver。
2. 直接拉取已成功发布的 0.30 镜像，作为下载阶段和最终阶段底图。
3. 在构建阶段获取四份适配器，验证完整 SHA256、文件头、BF16 类型和文件布局；缓存供失败重试复用。
4. 最终镜像继承原环境，仅加入 `/opt/h3-adapters` 和少量核验脚本。
5. 以原 `dynamo` 非 root 用户，在禁网条件下检查文件可读、完整资产、原生源码及环境未变化。
6. 先导出并核对三个成功报告，再推送 runtime；两次构建共用本地 BuildKit 缓存。
7. 保存发布镜像 digest 和完整轻量证据，上传 `image-evidence-h3-distilled-<SHA>-<attempt>` 制品。

最终阶段保留原用户、入口、环境和 Base 挂载约定。权重只在 assets 阶段设置权限后复制，避免最终层再次 chmod 导致约5.64GB内容重复写入镜像层。没有外部 Dockerfile syntax frontend 镜像依赖；这里仍需要 Harbor 和权重下载域名的实际连通性。

三个门禁含义分别是：

| 标记 | 证明范围 |
| --- | --- |
| `PASS_ASSETS_ONLY` | 完整权重和随附文档匹配固定清单 |
| `PASS_INSTALLED_CODE_ONLY` | 实际安装版本、53个相关源码和Omni直接默认依赖元数据通过 |
| `PASS_DISTILLED_CPU_ONLY` | 上述检查通过，打包前后Python包清单和关键环境一致 |

它们都不表示 GPU 推理或 HSDP 已验收。原底图的软件 NOTICE/SBOM 保留；新增模型许可文件单独随资产分发，不将模型误标为全部 Apache 许可。

## 6. 成功后下载并保存发布证据

在这次 Actions 的 Artifacts 下载 ZIP，替换下面的实际文件名：

```cmd
cd /d D:\AI\dynamo-h3-distilled
call delivery\h3-distilled\03-import-image-evidence.cmd "C:\Users\admin\Downloads\image-evidence-h3-distilled-实际文件名.zip"
```

脚本检查 ZIP 安全性、锁文件、构建报告、文件哈希、Git 提交、发布仓库及镜像 digest，保存到：

```text
D:\AI\_workspace\H3_Distilled_Image_20261013\published\<commit>\
```

部署使用该目录 `deployment-image.txt` 的 `repo@sha256:...`，不要拿 ZIP 校验值当镜像 digest。该步骤不访问 GPU，不部署，也不等同于独立 Harbor 拉取验收。

## 7. 可选：本机提前下载完整权重

默认流程不要求这一步。只有需要本地备份或实算一次权重 SHA256 时执行：

```cmd
cd /d D:\AI\dynamo-h3-distilled
call delivery\h3-distilled\00-fetch-assets.cmd
```

下载使用本机17890代理，文件写到本机交付目录的 `downloads`。完成后权重缓存和已安装副本合计约11.3GB，应预留空间。

**本机下载不会随 Git push 上传到 Runner。** 当前 CI 从锁定来源下载；本机缓存仅是本机备份和校验。若 Runner 的源站下载失败，需要在构建端定位实际出口，或另行接入经过确认的内部资产源；不要将权重临时提交到 Git，也不要重试 Pod 外网代理。

## 8. 后续部署交接

镜像内 [profiles.json](../../container/h3-distilled/profiles.json) 提供六个参数合同和精确权重路径，但不是已经集成进现有 `h3.cmd` 的新 CLI。

| Profile | Base | 加载 | 步数 | shift |
| --- | --- | --- | ---: | --- |
| base-fl50 | FL2VA | 不传LoRA | 50 | 12/3 |
| base-ref50 | Ref2VA | 不传LoRA | 50 | 12/3 |
| turbo-fl8 | FL2VA | 预加载 + 请求级LoRA | 8 | 6/3 |
| turbo-ref8 | Ref2VA | 预加载 + 请求级LoRA | 8 | 6/3 |
| turbo-fl4 | FL2VA | 预加载 + 请求级LoRA | 4 | 6/3 |
| fasth3-dense4 | FL2VA | 启动时融合；请求禁用LoRA | 4 | 12/3 |

所有组使用同一个新镜像，Base组在独立未加载适配器的进程测试。Turbo请求path必须与预加载文件一致，scale为1；FastH3仅T2VA，不支持首帧/首尾帧/Ref2VA，不用通用PEFT自行合并。Base源目录不改写。

当前TP1配置下，Turbo的LoRA缓冲在各rank保留完整副本，不会自动按HSDP四卡分片；不能将适配器文件大小除以4来估算每卡显存。FastH3启动融合也有临时内存开销，均需在后续四卡检查中记录。

现有部署包锁定旧镜像和50步，不能只替换字符串便宣称完成适配。下一阶段需在独立部署包中更新镜像合同、加入profile-aware校验和请求构造，保持TP1/U4/HSDP4、encoder TP4、VAE patch4、BF16/eager、无Cache-DiT/量化/offload及单worker串行。

目前FastH3要求的Base revision `9bfb6693f2cf6de171db46d1aa586f67d773a1da` 尚未与实际挂载文件核实。打包适配器不等于解决这项验收。0.30大图片条件的多卡VAE问题也保留原状态，首轮必须覆盖首帧和参考图片；若复现再定向修复并同步重跑Base对照。

本轮不创建或修改K8s资源。未来新Pod仍使用 `jirx-`，镜像标签不使用个人缩写，固定context和节点调度边界照旧。临时结果先下载再清理。

## 9. 首错定位

| 首个失败位置 | 应检查的地方 |
| --- | --- |
| 本机Git403 | GitHub写权限；与Harbor管理员权限分开 |
| 拉取固定Harbor底图 | Runner的registry登录、证书和内部网络 |
| Acquiring adapter | Runner构建阶段到HF及重定向存储域名的出口；查看适配器ID和HTTP/TLS错误 |
| SHA/header检查 | 实际下载字节、缓存是否损坏及锁定来源；不得跳过检查 |
| 源码/环境检查 | 是否拉错底图、原环境是否被修改；不通过pip升级绕过 |
| Harbor push失败 | 仓库现有推送凭据、配额、存储空间 |
| K8s拉取或模型运行 | 属下一阶段；与构建成功分开记录 |

当前交付没有触发构建或发布，不承诺首次长下载一定成功。最终以Runner的完整资产报告、新镜像digest和后续四卡媒体结果为准。
