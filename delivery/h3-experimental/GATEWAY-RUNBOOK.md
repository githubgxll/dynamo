# H3 实验镜像：Video Gateway 服务基线 + Omni 0.31

## 结论先行

以 **2026-10-16 实际查询的 `DingoRouter-video-gateway@e8c9195672f761ec09344a641730844c9e6c9e99`** 完整服务源码为基础，保留同事的 Gateway、监测与组内构建流程，再升级 **vLLM 0.31.0 + vLLM-Omni 0.31.0rc1**。不再以 main 的 `ai-dynamo 1.6.0` 配方为底座。

这不是把推理框架退回 0.29：服务基线使用 **ai-dingo / ai-dingo-runtime 1.3.0**，推理引擎升级到 0.31。main 只作为对比观察点记录，其后更新不会直接改变本次输入。

四份适配器仍打入 `/opt/h3-adapters`，Base 大权重仍使用已有只读挂载。指定 Ref8 适配器修复为 **8 步、视频 shift 12 / 音频 shift 3**。本次不加入超分实现。

**沿用 `Dingorouter-H3-vllm-omni-jirx`，不会新建 GitHub 分支。** 在已有 `6bada3e91b143da8551178103da15c88d356fb30` 后普通提交、普通 push，保留此前 main 方案的历史，不改写同事分支。

```text
Harbor 仓库：registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm
最终标签：omni031rc1-gateway-distilled-ref8shift12-<源码SHA前12位>
框架标签：omni031rc1-gateway-framework-<源码SHA前12位>
```

镜像名不带个人缩写。最终部署用记录中的 `repo@sha256:...`。
交付状态为**已准备并本地检查的构建配方**；CI、Harbor、GPU 和完整服务验收仍是独立门禁。

## 1. Windows 本机：检查与提交

以下在本机 **Windows CMD** 执行，不在 K8s Pod 内；本机无需 Docker/WSL2/GPU。

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\h3-experimental\01-check.cmd
```

看到 `PASS_LOCAL_IMPLEMENTATION_CHECKS_ONLY` 与 `PASS_LOCAL_RECIPE_ONLY` 后：

```bat
call delivery\h3-experimental\02-submit-build.cmd
```

第二条会真正提交、push，并触发 Linux Runner 构建及发布。脚本会：

1. 校验精确服务基线、输入指纹、资产锁、Ref8 补丁、工作流和本地回归。
2. 通过命令级本机 Git 代理 `http://127.0.0.1:17890` 查询 GitHub，确认 `DingoRouter-video-gateway` 没有在核验后漂移。
3. 确认远端既有实验分支历史包含在本地。分支不存在、无权限、网络失败、历史分叉都会停止，不创建替代分支、不强推。
4. 备份暂存区、写入已验证的完整 Git tree，保留已跟踪但被忽略的路径、Linux 文件模式和符号链接；创建 DCO sign-off 提交，普通 push。

同事服务分支有新提交时，先审查新变化并更新锁。GitHub 403 要核对账号写权限，不要把密码/令牌写入聊天或脚本。不要手工 `git add -f .`，不要执行旧目录中的提交脚本。

**必须推送本次新提交后再看 Actions**。旧 run 的 Re-run jobs 只会重跑旧源码，不会包含本次修复。

## 2. Linux Runner：自动构建

查看 [dynamo Actions](https://github.com/githubgxll/dynamo/actions) 中 **Dingo H3 gateway experimental Omni 0.31**。执行位置是 `[self-hosted, linux, x64, dingo]`，沿用现有 Harbor secrets 和 Docker daemon；不是本机，也不是 K8s GPU Pod。

| 阶段 | 自动工作 | 通过后的证据 |
|---|---|---|
| 输入检查 | 服务 commit、官方底图 digest、资产和目标仓库 | 配方身份 |
| native builder | 固定 manylinux、华为 Alma/EPEL、编译工具链 | reusable builder |
| 完整服务框架 | ai-dingo、Gateway/监测；联合求解完整 Omni、服务默认依赖及 Gateway extra | 实际 Linux 依赖、导入、CPU 媒体检查 |
| 合规与框架发布 | 保留 NOTICE/SBOM/许可证门禁 | 框架中间 digest |
| 蒸馏扩展 | 从真实框架 digest 加四份资产、Ref8 补丁和六模式脚本 | 资产/补丁来源记录 |
| 离线门禁与发布 | 非 root、断网重查资产、合同、依赖、服务导入及媒体 | 最终 digest 和 artifact |

完整依赖在框架层、合规检查之前求解。最终蒸馏层不再 apt/pip 安装，避免改变已核验的服务运行栈。构建时不启动模型、Gateway 或 NATS/etcd 服务。

继续使用 reusable builder、sccache 和 `dingo-h3-distilled-assets-v1` 下载缓存。同分支不保证所有缓存命中；底图/模板变化时生成新指纹，不硬填旧 builder tag。

## 3. Windows 本机：导入发布证据

成功后下载该次 run 的 `h3-experimental-omni031-<完整SHA>-<attempt>` artifact ZIP，保存完整日志，再执行（替换 ZIP 路径）：

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\h3-experimental\03-import-image-evidence.cmd "C:\Users\admin\Downloads\h3-experimental-omni031-artifact.zip"
```

脚本校验源码、服务基线、parent/final digest、报告状态与哈希，保存到 `.h3-experimental-evidence\local\import-*`。`deployment-image.txt` 必须指向最终四权重镜像。框架中间 digest、最终 digest、ZIP SHA256 不能混用。

## 4. K8s：后续单独验收

本轮脚本不操作 K8s，不为构建占用 H100。发布后在固定 hd04 context、新建 `jirx-` 实验 Pod 验收，不修改同事 Pod、节点和调度策略。

| 模式 | Base 只读分区 | 步数 | 视频/音频 shift | 适配方式 |
|---|---|---:|---|---|
| base-fl50 | FL2VA | 50 | 12/3 | 无 |
| base-ref50 | Ref2VA | 50 | 12/3 | 无 |
| turbo-fl8 | FL2VA | 8 | 6/3 | 请求 LoRA |
| turbo-ref8 | Ref2VA | 8 | 12/3 | 请求 LoRA + 定点补丁 |
| turbo-fl4 | FL2VA | 4 | 6/3 | 请求 LoRA |
| fasth3-dense4 | FL2VA | 4 | 12/3 | 启动融合，仅 T2VA |

Base 根目录 `/userdata/mengsz/MiniMax-H3`；不要用挂载遮住镜像内 `/opt/h3-adapters`。

**入口要区分**：镜像保留同事 Gateway/监测组件，六种实验 profile 仍调用原生 `vllm-omni serve ... --omni`。现有公共 Gateway 拒绝任意 LoRA 输入，并有自己的路由与参数合同；镜像里有四份权重不代表公共 Gateway 自动开放四种模式。生产接入需要同事配置受控 preset/backend 路由及监测，单独验收；本次未扩大公共 API 的 LoRA 权限。

容器内 Linux 示例，只打印命令、不启动服务：

```bash
python3 /opt/h3-experimental/profiles/launch.py --profile turbo-ref8
```

确认挂载和四卡配置后才使用 `--start`。FastH3 Dense4 会核验指定 Base revision 的 81 个文件（约 144GB）；不匹配时不能覆盖共享 Base。

验收顺序：独立拉取 → GPU 可见 → Base/适配器身份 → 原生 FL8/Ref8 小样例与 12/3 → Gateway 提交/轮询/下载/错误响应/监测 → 性能与人工音视频评审。构建期 CPU 检查不能替代上述门禁。emptyDir Pod 删除前先下载结果。

## 5. 历史问题与本次处理

| 历史坑 | 本次做法 |
|---|---|
| main 默认 Alma mirrorlist 再次 TLS 失败 | 继承服务分支固定 manylinux、华为 Alma/EPEL，覆盖新增 Synergy；保留重试，不仅修 Extras |
| 混淆 Windows/Runner/Pod 代理 | 本机代理只用于 Git；不硬编码曾失败的 hd04 8080 |
| `--no-deps` 导致 Pydantic/Kubernetes/zstandard 缺口 | 按真实 ai-dingo/runtime/vLLM 元数据联合 Omni 与 Gateway extra 求解 |
| 可选 KVBM/NIXL、ModelExpress/Protobuf 冲突 | H3 实验继续关闭两个组件，保留官方 NIXL/GPU 栈；不安装整个 ai-dingo[vllm] extra |
| main 专有依赖混入服务分支 | 移除 aisimulate/main 约束及 PyNvVideoCodec 2.2.3 覆盖，恢复服务分支身份 |
| PyAV18 与 Gateway `av<17` 冲突 | 使用同时满足 Gateway/Omni 的 PyAV16.1.0；保留上游 FFmpeg/ffprobe/RGB 探针 |
| 官方 Torch/NCCL 元数据不一致 | 只保留固定底图、实际版本和覆盖文件共同证明的精确例外 |
| OpenCV 两发行包共享 cv2 | 同版本 metadata 满足完整依赖，headless 最后安装并验证实际 RECORD |
| 旧源码补丁直接套新引擎 | 旧0.27专用补丁按版本门禁跳过；Ref8用0.31独立补丁及前后哈希 |
| 0.31 输入输出及内部导入变动 | 定点适配服务桥接代码，保留旧布局兼容；仍需真实服务 E2E |
| soxr LGPL 门禁 | 复用精确 soxr1.1.0/vllm-runtime 实验例外；保留检查、NOTICE/SBOM，不扩大未知包白名单 |
| Git ignored 文件导致提交失败 | 临时 index 验证后提交精确 tree，备份 index、保留文件模式 |

## 6. 仍需验证的风险

- 软件源曾成功不能证明今天所有外网可达。Registry、GitHub、PyPI、资产、RPM 文件是不同链路，按完整日志的首次失败阶段处理；EOF/TLS 不是 GitHub 写权限或版本不存在的证据。
- 本机无 Linux Docker/H100。依赖回归使用模拟 metadata；CI 才执行真实 resolver、安装后检查、服务导入与 CPU 媒体探针。
- 服务源码保留不等于生产兼容通过。队列、调度、取消、输出协议、监测和所有 backend 需在同事环境跑完整服务验收。
- RC 的 GPU kernels、Torch/Triton 编译、分片、LoRA 激活、Base 身份及感知质量需逐模式检查；一次 FL8 成功不能代表 Ref8/FastH3。
- 服务分支还继承了 `UCX v1.20.x` 移动分支和 native builder 的 Git TLS 校验关闭设置。本次没有新加代理或扩大此设置，也不能因此声称所有 native 源已严格固定、全链路 TLS 已校验；后续应与 Runner 负责人单独核对 CA 与 native 来源锁。
- 旧多阶段/分离部署 YAML 与 0.31 的 `deploy_config` 不是同一合同。新增兼容代码不代表旧 `stage_args` YAML 可直接运行；本次 H3 主路径为聚合 Omni worker，多阶段配置要迁移并单独验收。
- 新传递依赖若触发未知许可证，审查具体包而非全局跳过。报告与模型许可均保留；正式生产分发由同事按公司规则审核。

来源：[服务基线](https://github.com/githubgxll/dynamo/tree/e8c9195672f761ec09344a641730844c9e6c9e99)、[0.30 成功修复](https://github.com/githubgxll/dynamo/tree/958c2ce95cd361fd5a6770e3023eac4de968f674)、[四权重成功提交](https://github.com/githubgxll/dynamo/tree/36db79a59362bdac8e4fefcdceff994b46e49a62)。精确身份以源码锁、环境 policy、资产锁与 input-seal 为准。
