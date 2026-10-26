# SeedVR2 C14：轻量诊断镜像构筑交付

构筑配方复核日期：2026-10-26。实验依据仍为 2026-10-23 的 C14 指南，原片、权重及已完成的实验保持原身份。

**结论：直接继承已成功发布的 Omni/Gateway 轻量镜像，保留完整 vLLM-Omni 0.31.0rc1、Gateway 和 RMSNorm 修复，仅增加默认关闭的 C14 抓取与重放能力。没有依赖升级、权重下载、少步蒸馏适配器或 Ref8 补丁。** 本次是定位并验证修复的镜像，不预先宣称 LAB、VAE 或编码已被证明是根因。

镜像发布后，仍需按照两份诊断指南进行 A0/A1/A2/A3。CPU 构筑检查不替代 SP8、真实 VAE 重放与人工画质验收。完整 SR-only 入口和启动说明见 [OPERATIONS.md](OPERATIONS.md)。

## 1. 固定依据

| 项目 | 本次取值 |
|---|---|
| 唯一提交分支 | `Dingorouter-H3-vllm-omni-jirx`，不新建分支、不强推 |
| 仓库 | `https://github.com/githubgxll/dynamo.git` |
| 上版成功轻量源码 | `2362a3241c03a286b98e51b8003a51f494fab4f1` |
| 轻量父镜像 digest | `sha256:9382949d2789b23cdfe4ea71d68c7973592de28f209082bd7907c1d09fe008c9` |
| Gateway 固定源码 | `9ccee2f56d0122be94d7436ae2f5fec96d15ea9b` |
| Omni 固定源码 | `6dd0d1f9310f7598b773c796b434c8000c2816ec` |
| 保留的 RMSNorm 修复 | PR #8767 / `0708f05e2e27f16fca8b48e7c5a085277d1cafda` |
| 框架 | vLLM 0.31.0、Omni 0.31.0rc1、Torch 2.13.0+cu130、Cache-DiT 1.5.0 |
| 新 tag 格式 | `omni031rc1-gateway-lite-seedvr2-c14-<构筑commit前12位>` |
| Harbor 仓库 | `registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm` |

父镜像完整引用：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:9382949d2789b23cdfe4ea71d68c7973592de28f209082bd7907c1d09fe008c9
```

已归档的上版发布证明保存在 [parent-evidence/image-release-evidence.json](parent-evidence/image-release-evidence.json)，对应 [Actions 39583937193](https://github.com/githubgxll/dynamo/actions/runs/39583937193)。这份文件是历史发布记录摘要，CI 会重新拉取完整 digest、检查实际镜像配置和安装内容，不能把摘要当作本轮拉取成功。

为使 C14 只改变诊断能力，不自动追踪同事分支最新提交。本次保留上版已验证的 Gateway 版本；后续升级 Gateway 应单独 review。完整 H3 Base 和 SeedVR2 权重仍只读挂载；不打入新镜像、不在启动时重新下载或转换。

## 2. Windows CMD：检查、提交、观察

在 CMD 复制执行：

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\seedvr2-c14\01-check.cmd
```

检查源码锁、精确 Git 文件封存、父镜像历史证明、原有依赖审计/轻量框架/RMSNorm 回归、新诊断/重放/提交/发布回归与 Linux shell 语法。详细日志进入 `.h3-experimental-evidence\c14\local\checks-*`，正常控制台只汇总结果。

检查通过后，提交并启动本次构筑：

```bat
call delivery\seedvr2-c14\02-submit-build.cmd
```

该入口会再次检查，仅暂存已封存的 C14 文件及工作流修改，正常提交并 push 到既有分支。提交信息携带 `[seedvr2-c14]`，只启动 C14 job，避免同时重建完整框架或下载四份蒸馏权重。不会使用 `[skip ci]`。

到 [Dingo 镜像 Actions](https://github.com/githubgxll/dynamo/actions/workflows/dingo-router-ci.yml) 找名称为 `seedvr2-c14 | Dingorouter-H3-vllm-omni-jirx | <完整commit>` 的记录。记录内旧 job 显示跳过属于预期。

如果同一源码已经上传，重复运行 02 不会创建重复构筑。只有明确需要重试时执行：

```bat
call delivery\seedvr2-c14\03-run-build.cmd
```

03 要求工作区干净、本地与远端一致；只新增带 C14 标记的空提交并 push，源码树不变。不依赖 `gh`、网页 `Run workflow` 按钮，也不修改默认 main。此前工作流不在 main 上，网页没有手动按钮并不代表权限不足。

脚本默认复用已经成功使用的本地 Python。若路径变化，可先指定已有工具环境：

```bat
set "H3_EXPERIMENT_PY=D:\你的工具环境\Scripts\python.exe"
```

该环境需有 PyYAML、Jinja2、packaging；本机不必安装 Torch。已有 Git 代理只用于本地命令，不写入镜像、Runner 或 Pod。

## 3. CI 实际做什么

1. 核对本轮完整 commit、branch、文件封存与固定 parent digest。
2. 使用既有 Harbor 凭据和持续缓存的 Docker builder；拉取成功轻量父镜像。
3. 在关闭网络的 Docker RUN 中应用最小抓点补丁，检查整个安装源码、RMSNorm 与继承依赖；不执行 pip/apt 安装，不去 Docker Hub 下载额外 frontend。只允许源码锁列出的修改，并同步更新安装包 RECORD；旧轻量配方的校验器保留原样。
4. 先 build/load 本地候选，不发布。再次以 UID1000、无网络启动最终候选，实际导入已安装模块、执行 CPU 数值及诊断测试；验证依赖/系统包、入口、运行用户、完整父层保持不变。
5. 只发布已测试的同一个 image ID。从本次 push 回执取 digest，按 digest 重新拉取并核对 image ID；任何一步失败都不生成成功发布记录。
6. 上传独立证据，即使失败也保留日志。GPU、同输入 C14 重放、完整视频与人工画质检查仍标明未运行。

失败后的关键制品：`inputs.json`、`build/`、`final/`、`candidate-verified.json`、`package-preservation.json`、`inheritance.json`。build 和 final 都必须有新的诊断 CPU 回归、Gateway 回归、完整依赖、媒体和服务导入报告，缺少或跳过不会放行。成功额外生成 `image.json`、`deployment-image.txt`、完整 recipe 快照及所有证据哈希。

## 4. 导入发布结果

成功记录底部下载 `image-evidence-seedvr2-c14-<完整commit>-<attempt>` ZIP。回到相同提交的本地工作区执行：

```bat
call delivery\seedvr2-c14\04-import-image-evidence.cmd "C:\Users\admin\Downloads\image-evidence-seedvr2-c14-完整commit-1.zip"
```

导入器先检查 ZIP 路径、重复文件、体积、哈希，再核对提交、镜像 digest、已测试 image ID、CPU 报告和 recipe 快照。导入到新目录，禁止覆盖已有证据。部署必须取 `deployment-image.txt` 的 `repo@sha256:...`；**ZIP 的 SHA256 不是镜像 digest**。

本轮部署包使用新的 image/source 锁和独立 run ID；不要把新镜像硬塞进旧 Ref7 包，不修改旧保护、旧 manifest 或旧 C14 产物。

## 5. 保持固定的 C14 诊断条件

只用已有完整 362 帧、24FPS、1280×768 的 Base 原片。1080 档内部 1840×1104、最终 1800×1080；3B FP16、SP8、eager、一步 Euler、CFG1、seed7723、17帧窗/4重叠/13步长、原 LAB 亮度0.8、VAE tiling和高度分片。指定抓取窗口起点130/143/156/169/182，不能裁6–8秒后重新推理。

原 LAB/原 chunk2/原编码链先自对照，再比较候选。旧 raw1104 MP4 已经编码，不能代替校色前浮点 RGB。相同 seed 不能代替实际 noise/latent 证据。只支持受控本地张量制品，不加载不可信 pickle，不新增任意远程张量 HTTP 接口。

## 6. 已知坑与处理

| 现象 | 处理 |
|---|---|
| 本地检查通过但 Linux 镜像失败 | 本地无 Torch/无镜像，证据级别不同；检查本轮 artifact，不跳过最终安装模块检查 |
| 依赖、媒体及102项CPU测试通过，最后报 `Validation changed inherited environment or installed sources` | 旧C14检查在导入框架前后枚举同一进程；父镜像的setuptools80.10.2导入会把自带vendor目录加入sys.path，造成包发现范围变化。现已改为同UID/同环境的干净子进程快照，仍严格比较完整METADATA/direct_url、dpkg和继承文件；不忽略依赖。若仍有真实差异，控制台输出类别和前10项名称，完整差异保存为 `environment-diff-*.json`，另保存 `inventory-context.json` 与前后源码证明。原失败日志未保留分项差异，最终原因以重跑证据确认 |
| `Inputs differ from their reviewed seal` | 有文件变动；先 review 根因和完整 diff，再由维护者重新封存。不要把自动 seal 加进02来掩盖变动 |
| 工作区有其他修改或远端已前进 | 保留文件；脚本拒绝无关暂存和强推，先合并/核对既有分支 |
| `Run workflow` 不可见 | 用02新提交或03明确重试；不需要改main或换分支 |
| Docker Hub EOF、pip超时 | 本配方不重拉外部frontend、不装包；若发生，核对是否误启动旧full配方。Harbor拉取仍需Runner网络与权限 |
| UID1000的 uv cache Permission denied | 继承已修复隔离缓存逻辑，CPU验收独立临时目录；不把检查改成root掩盖权限问题 |
| Runner GC后不足100GiB | Runner准入问题，联系维护同事排期/扩容；不删其他任务缓存、不取消阈值 |
| 本地上游nadit.py未修复 | 不整目录覆盖；保留已修复哈希`03fd3a0c...1a6f01`，六组精度回归必须通过 |
| 抓点耗尽磁盘或rank失联 | 留存失败证据并停止所属任务；不静默丢张量或改变SP/dtype/window继续 |
| A0没有复现或原路重放不等价 | 标记无结论，先查输入/实际配置/RNG/调度/抓点影响；不能宣布C14修好 |

只有完成受控GPU试验、全片技术媒体检查、正常速度观看和回归后，才能把候选改动称作“已验证修复”。关闭抓点之后另做性能验收。
