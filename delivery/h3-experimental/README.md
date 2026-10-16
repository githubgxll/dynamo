# H3 实验镜像：main + Omni 0.31.0rc1 + 四种蒸馏

## 结论与当前状态

本目录准备的是**可提交构建的实验镜像源码**。以 2026-10-16 实际读取的 `githubgxll/dynamo/main@049eece3f32f8d38dc1d8c52d4b793cb30e394bb` 为基础，使用 vLLM **0.31.0**、vLLM-Omni **0.31.0rc1**，包含四份固定版本适配器，并修复指定 Ref8 适配器的 **8 步、视频 shift 12 / 音频 shift 3**。

**Base 权重仍使用现有只读挂载，不打进镜像。** 镜像支持 Base 和四种蒸馏模式，适配器目录仍为 `/opt/h3-adapters`，四份共 5,636,659,576 字节。

本机准备目录：`D:\AI\dynamo-h3-experimental031`。按 2026-10-16 的最新约定，本次及后续本机 MiniMax-H3 实验镜像统一推送到 **`Dingorouter-H3-vllm-omni-jirx`**（大小写按此字符串）。远端尚不存在时，首次执行提交脚本会创建它；之后持续复用，不再按日期、框架版本或权重种类另开发布分支。旧 `DingoRouter-h3-distilled-20261013` 保留历史。Harbor 仍是：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm
```

最终标签格式：`omni031rc1-distilled-ref8shift12-<本次源码SHA前12位>`。框架中间镜像另用 `omni031rc1-main-framework-<SHA前12位>`；部署须用最终镜像的 `repo@sha256:...`。

交付时尚未提交、推送、触发 Actions、发布 Harbor 或操作 K8s。本地检查的结果只证明源码、配置、补丁和测试通过；镜像构建、真实 Linux 依赖求解、Harbor 拉取和 GPU 生成分别验收。

## 1. 你在本机 Windows CMD 执行

以下均在**本机 Windows CMD** 执行，不在 K8s Pod 中执行；本机无需 Docker、WSL2 或 GPU。

先检查：

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\h3-experimental\01-check.cmd
```

检查通过后，由你执行这一条提交并触发构建：

```bat
call delivery\h3-experimental\02-submit-build.cmd
```

此脚本会：

1. 校验 main 源码快照与本次 H3 修改的 Git 内容指纹，运行本地回归检查。
2. 通过本机专用代理 `http://127.0.0.1:17890` 获取 main 并查询指定实验分支；代理仅用于这次 Git 命令，不写进 Runner 或 Pod。只有成功查询且结果为空，才按本次授权创建这个精确命名的分支；网络或认证错误不会被当作“分支不存在”。
3. 若 main 已更新、配置被修改或远端实验分支的提交尚未包含在本地历史中，则停止，先整合并重查；不强推、不覆盖他人历史。
4. 备份当前暂存区，检查是否含有独立的暂存修改；重新确认工作文件与已校验的 Git tree 一致后，直接把这个精确 tree 写入暂存区（不改工作文件），保留 main 已跟踪但被忽略的路径、Linux 可执行位与符号链接。随后按仓库要求创建带 DCO sign-off 的提交，普通 push 到 `Dingorouter-H3-vllm-omni-jirx`，并设置为本地分支的 upstream。
5. 已提交但 push 失败可以修复网络/账号后重跑；不会重复创建空提交。已成功推送同一份内容则提示查看已有 Actions。

Python 复用已有工具环境：`D:\AI\_workspace\H3_Omni030_Image_20260929\tools-venv\Scripts\python.exe`。如果迁移本地目录，可指定另一个已安装 PyYAML、Jinja2、packaging 的 Python：

```bat
set "H3_EXPERIMENT_PY=C:\your-tools\Scripts\python.exe"
call delivery\h3-experimental\01-check.cmd
```

如果 Git 提示作者信息未配置，在**当前仓库**设置自己的 `user.name` 与 `user.email` 后重跑；这不是 Harbor 登录。GitHub 返回 403 则检查当前 GitHub 账号的仓库写权限；Harbor 管理员权限不会自动赋予 GitHub 写权限。不要把密码或令牌贴入聊天或写入脚本。

若此前提交在 `git add` 报 `CLAUDE.md`、`profiling_results`、`tensorrt-llm` 或 `server` 被忽略，它发生在提交前，可能已留下部分暂存内容。更新后的脚本可直接重跑，不需要清空目录或修改 `.gitignore`。它使用已校验 tree，且会保留 `.h3-experimental-evidence\local\index-backups\...` 下的原暂存区备份。若提示 `Staged-only content`，说明暂存区有既不同于 HEAD 又不同于本次目标的内容；应先保存并核对，不能强制覆盖。

不要在旧的 `dynamo-h3-distilled` 或 `dynamo-h3-omni031` 目录执行旧提交脚本；它们保留了旧候选，目标与本次不同。

## 2. GitHub Actions 在 Linux Runner 自动执行

打开 [dynamo Actions](https://github.com/githubgxll/dynamo/actions)，查看 **Dingo H3 experimental Omni 0.31**。

执行位置是组内 `[self-hosted, linux, x64, dingo]` Runner。不是本机，也不是你的 GPU Pod。流程沿用仓库 Harbor secrets 和持久 Docker daemon，不需要你 SSH、sudo 或申请 H100 构建。

| 顺序 | 自动做什么 | 通过后能说明什么 |
|---|---|---|
| 1 | 校验固定 main、底图 digest、四资产锁、目标仓库 | 本次输入一致 |
| 2 | 检查/构建与 main 输入对应的 reusable builder | native 工具链可用 |
| 3 | 用 main 模板构建框架 runtime，执行 main 自带合规流程，发布框架 digest | main 框架镜像构建并发布 |
| 4 | 从这个真实 digest 派生 H3 层；补齐音视频依赖，下载四份适配器并校验 SHA256 | 依赖和资产已装入候选层 |
| 5 | 应用 0.31 专用 Ref8 补丁，非 root、断网运行资产/合同/依赖/CPU H.264+AAC 检查 | CPU 门禁通过，不代表 GPU |
| 6 | 导出并核对上述证据，再发布最终 distilled runtime，记录真实 digest | 本次最终镜像发布记录完整 |

框架和最终层共享源码提交身份，但 parent 通过 **digest** 传递，不通过可变 tag 猜测。失败时保存可用的证据；如果框架成功、最终层失败，不能把框架 digest 当成四权重镜像。

构建 PyPI 配置沿用最新 main 的 BuildKit secret 注入接口。当前公开索引为 `https://pypi.org/simple`。之前的 `127.0.0.1:17890` 是 Windows 本机代理，`10.201.136.61:8080` 也没有作为通用修复重新硬编码。若 Runner 出网再次失败，请先区分 Docker registry、构建阶段软件源和资产下载三个环节，并由 Runner 负责人核对网络。

## 3. 构建完成后下载证据

下载该次 Actions 的 `h3-experimental-omni031-<完整源码SHA>-<attempt>` artifact ZIP。然后在**本机 CMD**执行，例如：

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\h3-experimental\03-import-image-evidence.cmd "C:\Users\admin\Downloads\h3-experimental-omni031-artifact.zip"
```

替换为你实际下载的 ZIP 文件名。脚本检查源码、main、框架 parent、最终 digest、各报告状态与文件哈希，输出最终部署地址。结果保存在仓库内被 Git 忽略的 `.h3-experimental-evidence\local\import-*`。保留 ZIP 和导入目录。

必须保留的证据包括 `final/image.json`、`deployment-image.txt`、源码锁、最终环境包清单、依赖检查、音视频检查、四资产校验、Ref8 补丁/合同、main 合规报告与最终环境许可证差异清单。

**ZIP 自身 SHA256、框架 digest 和最终镜像 digest 是三种不同标识。** 导入成功仍不代表 K8s 已经拉取镜像。

## 4. 六种模式与 Pod 使用边界

| profile | Base 只读目录后缀 | 请求步数 | 视频/音频 shift | 适配器方式 |
|---|---|---:|---|---|
| `base-fl50` | `FL2VA` | 50 | 12/3 | 无 LoRA |
| `base-ref50` | `Ref2VA` | 50 | 12/3 | 无 LoRA |
| `turbo-fl8` | `FL2VA` | 8 | 6/3 | 每请求选择对应 LoRA，scale=1 |
| `turbo-ref8` | `Ref2VA` | 8 | **12/3** | 每请求选择对应 LoRA，scale=1；已定点修复 |
| `turbo-fl4` | `FL2VA` | 4 | 6/3 | 每请求选择对应 LoRA，scale=1 |
| `fasth3-dense4` | `FL2VA` | 4 | 12/3 | 启动时融合；仅 T2VA |

Base 默认根目录仍为 `/userdata/mengsz/MiniMax-H3`。不修改宿主机内容。每个 worker 只选择一个 profile；一个镜像支持多种模式，不等于一个进程可以任意切换所有模式。

这次不创建 Pod。后续在新的四卡 H100 实验 Pod 中，以最终 digest 配置 image，Base 只读挂载，`/work` 和共享内存使用可写 emptyDir。新 Pod 名用 `jirx-` 前缀，沿用 hd04 context；节点通过 `nodeSelector` 调度，不改同事 Pod、污点或节点策略。

下面命令只在**新的 Linux GPU Pod 容器内**执行，或写进它的启动参数，不在 Windows CMD 直接运行：

```sh
# 仅显示完整参数和请求合同，不启动模型
python3 -B /opt/h3-experimental/profiles/launch.py --profile turbo-ref8

# 真正检查运行环境、Base挂载并启动当前profile
python3 -B /opt/h3-experimental/profiles/launch.py --profile turbo-ref8 --start
```

实际使用已安装的原生 `vllm-omni serve ... --omni` 入口，固定四卡参数；不使用旧 `python -m dingo.vllm` Gateway 命令。main 的 Python 包已变成 `ai-dynamo`/`dynamo`，旧网关协议与生产监测逻辑需要同事另行集成。

详细参数、FastH3 Base 全量验证和请求合同见 [profiles/README.md](../../container/h3-experimental/profiles/README.md)。首次分别做 Base FL、Base Ref、FL8、Ref8、FL4、FastH3 的单样例验收，保存请求、适配器绑定证据、日志和生成 MP4。健康接口返回成功不代表生成成功。

## 5. 必须提前知道的变化与风险

- **main 是本次核验日期的固定快照，不是浮动的“永远最新”。** 提交脚本会重新检查是否漂移；以后升级重新审查并更新锁。此次从旧实验提交继承历史，应用 main 完整快照再加 H3 覆盖层，然后发布到固定实验分支；后续通过普通快进提交迭代，不改写历史。固定发布分支不表示永远沿用旧依赖。
- **首次升级构建可能较久。** 同分支有助于沿用原工作流，但 Docker 缓存由父镜像、输入和 Runner daemon 决定。main/native/CUDA 栈变化会使旧 builder 失效；保留 reusable builder、sccache 与原 `dingo-h3-distilled-assets-v1` 下载缓存，不伪造旧指纹。
- **main 的无关工作流也要核对。** 已修正 NVSkills reusable workflow 的相对路径写法，并仅排除本实验分支的 push，避免升级整个快照后误触发与镜像无关的任务；其它分支行为保留。
- **main 默认媒体镜像不能直接等同于 H3 MP4 音频镜像。** main 使用受限媒体组件并移除部分 Python 包。本次参考 main 自带 `examples/backends/vllm/omni/video_audio.Dockerfile`，用独立实验扩展层恢复真实 H.264/AAC、PyAV、soxr、torchcodec 等所需依赖，同时保留 main native 库。
- **不再套用 0.30 的旧冲突结论。** main 的 ModelExpress 已为 0.7.0，KVBM/NIXL 约束也已更新。本实验仍关闭可选 KVBM 和 ModelExpress，理由是 H3 实验范围；不是声称旧 Protobuf/NIXL 冲突仍然存在。
- **完整 Omni 默认依赖保留。** 严格检查 vLLM、Omni 及 Dynamo 默认依赖，冻结实际继承的 native 栈。没有为了过检查删掉必需包；没有把未启用的 `ai-dynamo[vllm]` 可选 extra 伪称全部满足。
- **仅有来源明确的精确元数据例外。** 官方底图 Torch/NCCL 覆盖与 main 的 PyNvVideoCodec 2.2.3 覆盖需要源码、实际版本和报告一致。其它依赖错误仍使构建失败，不使用 `pip check || true` 掩盖问题。
- **OpenCV 有两个上游发行包要求。** vLLM 需要 headless，Omni 的 guardrail 需要普通包；两者固定同版本，并最后安装 headless 为实际 `cv2` 提供者，逐文件核查 RECORD。不要在运行 Pod 中卸载任一包；最终 CPU 解码测试必须通过。
- **Ref8 是定点补丁。** 只改固定文件 `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` 对应的 12/3 校验；校验补丁前后源码哈希。8 步请求填 8，对应 9 个 sigma 边界，不照搬旧说明填 9。
- **FastH3 Dense4 有独立限制。** 本次是 Preview v1 LoRA，不是 V2 全量模型。只能 T2VA，启动融合且不能通过省略请求 LoRA 切回 Base；固定 schedule，禁止 offload。启动前完整核验指定 Base revision 的 81 个文件（约 144GB）；不匹配时请存储负责人提供正确只读快照，不能覆盖同事 Base。
- **最终音视频扩展改变许可范围。** main 自带合规报告保留；最终额外记录 Python 许可证元数据、包差异、媒体文件与 dpkg 清单。它们不是对最终实验镜像的法律批准，也不能沿用 main 受限编解码镜像的全部许可结论。交生产镜像时把这些差异交同事审查。
- **没有本机 Docker/H100 验收结果。** 真实 Linux 依赖求解、系统 FFmpeg/PyAV 兼容性、H100 kernels、四卡分片、Base 权重身份、适配器激活和输出质量仍须按上述门禁逐层验收。Pod 删除前下载 `/work`，emptyDir 会丢失。

## 6. 主要文件与来源

| 文件 | 作用 |
|---|---|
| `source-lock.json` / `input-seal.json` | 上游身份和相对 main 的精确修改指纹 |
| `.github/workflows/dingo-router-ci.yml` | `Dingorouter-H3-vllm-omni-jirx` 的唯一实验构建入口 |
| `.github/dingo-images.json` / `prepare_dingo_image_matrix.py` | vLLM 单项矩阵、Harbor 与 builder 指纹 |
| `container/context.yaml`、模板、`render.py` | main 构建配置与组内 reusable builder 接口 |
| `container/h3-experimental/environment/` | 完整依赖恢复、保护、审计和真实 CPU 媒体检查 |
| `container/h3-experimental/assets/` | 四份适配器的固定 revision、SHA256、下载与许可原文 |
| `container/h3-experimental/ref8/` | 独立 0.31 源码定点修复与合同验证 |
| `container/h3-experimental/profiles/` | 六模式启动、请求合同、FastH3 Base 校验 |
| `01-check.cmd` / `02-submit-build.cmd` / `03-import-image-evidence.cmd` | Windows 本机检查、提交、证据导入 |

依据：[本次 main](https://github.com/githubgxll/dynamo/tree/049eece3f32f8d38dc1d8c52d4b793cb30e394bb)、[维护 base 观察点](https://github.com/githubgxll/dynamo/tree/adca349852f88c035201ab95eff6bc0ced9ad93c)、[原实验分支基线](https://github.com/githubgxll/dynamo/tree/36db79a59362bdac8e4fefcdceff994b46e49a62)。完整 wheel、模型和补丁身份以锁文件为准。
