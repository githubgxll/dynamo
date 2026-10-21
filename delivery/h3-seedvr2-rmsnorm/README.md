# SeedVR2 RMSNorm 补丁镜像构建交付

日期：2026-10-21。结论：继承已经成功的 H3 成品镜像，只回补 vLLM-Omni PR #8767；不升级框架、不重装依赖、不下载或转换权重。使用原分支 `Dingorouter-H3-vllm-omni-jirx`。

## 1. 本轮固定输入

父镜像（必须使用完整 digest）：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:112e34379489f05c737079212abb9fa587676fd82650dd1c1af96b32f46dc9a7
```

| 项目 | 固定值 |
|---|---|
| 父镜像构建提交 | `db65d1ec869b6deee02debe7a58e108de7c927ef` |
| vLLM / Omni / Cache-DiT | `0.31.0` / `0.31.0rc1` / `1.5.0` |
| 上游修复 | [PR #8767](https://github.com/vllm-project/vllm-omni/pull/8767)，commit `0708f05e2e27f16fca8b48e7c5a085277d1cafda` |
| 修前 nadit.py SHA256 | `3c782ad3588f7f544a9cb14fe8258a9eb4d921881deee8501460ceaf0bf1ad63` |
| 修后 nadit.py SHA256 | `03fd3a0c199305ce3004643f8d1510183b06731099a72bc447c2bd589c1a6f01` |
| 新标签 | `omni031rc1-gateway-distilled-ref8shift12-srnormfix-<Git前12位>` |
| 仓库 / Runner | 原 Harbor `openclaw/ai-dingo-vllm`；`self-hosted, linux, x64, dingo` |

此次不追随 main 或视频网关分支更新。运行环境由完整父镜像 digest 固定；Git 分支只是提交载体。下一次框架升级才重新审查维护分支和依赖组合。

## 2. Windows CMD：检查和提交

在现有目录操作，禁止强推，不创建其他分支：

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\h3-seedvr2-rmsnorm\01-check.cmd
```

检查成功后提交。下列命令会检查、创建签署提交、普通推送原分支，并触发 Runner 构建和发布：

```bat
call delivery\h3-seedvr2-rmsnorm\02-submit-build.cmd
```

本次不要运行旧的 `delivery\h3-experimental\02-submit-build.cmd`。旧输入封存属于完整框架配方；新补丁有独立 `source-lock.json` 和 `input-seal.json`，以成功父提交为基线封存本轮所有 Git blob/模式。

若本地有无关改动、独立暂存内容或远端分叉，提交会停止并保留现有工作，不自动覆盖。若封存失败，应先审查 diff；不要为消除报错直接改哈希。维护者确认改动后才重新封存：

```bat
call delivery\h3-seedvr2-rmsnorm\env.cmd
"%H3_EXPERIMENT_PY%" -B -X utf8 delivery\h3-seedvr2-rmsnorm\ops.py seal
call delivery\h3-seedvr2-rmsnorm\01-check.cmd
```

本机 GitHub 使用既有的命令级代理 `127.0.0.1:17890`；不写入镜像、Runner 或 Pod。无需在本机登录 Harbor。CI 复用既有 `REGISTRY_USERNAME` / `REGISTRY_PASSWORD` secrets。

## 3. GitHub Actions 会执行什么

现有 `dingo-router-ci.yml` 已增加 `build_recipe` 选择：正常推送和默认手动运行采用 `seedvr2-rmsnorm`；只有显式选择 `full` 才触发旧的完整框架构建。

1. 校验提交范围、完整输入封存和父镜像身份。
2. 从 Harbor 拉固定父镜像；复用 docker driver，不拉 Docker Hub 的 Dockerfile 前端，不进入 dnf/pip/uv 安装流程。
3. 在离线构建层检查实际安装的旧源码 SHA，只修改 RMSNorm；更新对应 wheel RECORD，保留原 RECORD 和原始 wheel 安装来源。
4. 以 UID1000 运行原样上游六组测试和实际导入的独立数值探针；全 300 的 FP16 输入必须从旧式全 0 恢复为接近全 1。
5. 重新执行继承的依赖、CPU 媒体、Gateway 导入、Ref8 12:3、六种 profile 和四份适配器哈希检查。
6. 加载候选镜像，独立检查父层完整继承、镜像配置、Python/dpkg 版本清单；再次从最终镜像真实导入运行测试。
7. 只推送已通过测试的同一个本地 image ID；拉取发布后的完整 digest，确认仍是同一个 image ID，再生成发布记录。

中途失败不发布修复标签，也不会回退到无补丁镜像冒充成功。若推送成功但发布后的回拉或证据上传失败，标签可能已经存在；仍不视为完整交付，应检查日志并重试该工作流，不能自行认定验收完成。

本机检查覆盖补丁精确字节、错误源码/RECORD 拒绝、Git 暂存与封存、CI 路由/发布条件、Python/Shell 语法及原有回归。本机工具 Python 未安装 Torch；真实六组数值测试和实际模块导入必须由 Linux 镜像门禁执行，不能把本地 PASS 当作其结果。

## 4. 证据位置与导入

Actions 制品名：`image-evidence-h3-seedvr2-rmsnorm-<完整提交SHA>-<attempt>`。失败时也保存已有证据。

| 文件/目录 | 含义 |
|---|---|
| `image.json`、`deployment-image.txt` | 只有发布并回拉验证完成才生成；记录新完整镜像 digest |
| `build/patch-application.json` | 父来源、上游 commit、实际修前/修后源码及 RECORD 哈希 |
| `build/preservation.json` | 未修改其他 Omni Python 源码、继承 H3/法律文件和依赖元数据 |
| `build/retained/` | 本轮新执行的原有能力检查，不是复制旧 PASS |
| `final/runtime-validation.json`、`final/upstream-pytest.xml` | 最终镜像实际导入路径、五文件锁、六组数值和全300对照 |
| `inheritance.json`、`package-preservation.json` | 父层/配置继承和包版本零变化 |
| `recipe/` | 本轮全部封存输入，便于离线复核 |

下载 Actions 制品 ZIP 后：

```bat
call delivery\h3-seedvr2-rmsnorm\03-import-image-evidence.cmd "C:\Users\admin\Downloads\image-evidence.zip"
```

把占位文件名替换为实际下载的 ZIP。导入会拒绝路径穿越、缺失/篡改报告、错误镜像来源，以及与当前提交不符的配方；保留 ZIP 的 SHA256 和解压后证据。**ZIP SHA256 不等于镜像 digest。**

镜像内旧 `/opt/dynamo/build-info`、`/opt/h3-experimental/build-info` 保持原样。新报告在 `/opt/h3-seedvr2-rmsnorm/build-info`。`H3_EXPERIMENTAL_SOURCE_COMMIT` 更新为本轮 Git commit；`H3_RMSNORM_PARENT_IMAGE` 单独记录成品父镜像。旧 `H3_EXPERIMENTAL_BASE_IMAGE` 仍表示原框架父来源，Gateway/DYNAMO 源码标记保持继承值。

## 5. 这次重点避免的历史问题

- 不 rebase/重装，所以不重新触发 AlmaLinux、PyPI、NIXL、Protobuf、遥测版本求解或适配器下载。
- 不安装 pytest：父环境已有 `9.1.1`。测试禁用自动插件和工作目录配置，防止导入另一份源码。
- 不裸跑 `uv pip check`：复用已修复的非 root、独立临时缓存、离线检查封装，避免 `/opt/uv/cache` 权限错误。
- 依赖检查重新执行；许可证政策保持不变。逐文件核对父 `/legal`、检查包清单不变，新增源码修改 NOTICE 和原上游 Apache-2.0 许可。此次不重新生成依赖 SBOM，也不宣称新增法律审批。
- 原源码/五文件锁任一不匹配就停止；不模糊替换，不篡改原 wheel 的 direct_url 来伪装未修改。
- 构建只发送专用上下文白名单，避免再漏 COPY 必需文件或带入本机模型/证据。
- Runner 的 `/etc/dingo-gc` 磁盘准入仍由管理员控制。补丁层虽小，冷节点拉取完整父镜像仍需要足够磁盘；不删除组内缓存或降低准入阈值。
- 继续保留原镜像和原运行 Pod；本脚本不执行 kubectl，不替换任何部署。

## 6. 交回实验会话后的动作

将新 `repo@sha256:...`、构建提交、Actions URL、完整证据 ZIP 交回。新部署必须使用独立目录和 run ID，更新 `workflow.py`、`resident.py`、`resident_media.py` 的镜像与真实源码锁。预计五个 SeedVR2 文件中只有 `nadit.py` 改变，但仍以最终报告复核。

现有部署包还需实现显式旧 Base 导入/SR-only 接口，分别记录原片来源镜像和本轮执行镜像；不能改写旧 manifest，也不能通过重新生成 H3 视频替代同原片对照。此处仅交付镜像配方，不修改部署包。

GPU 阶段保持同一原片 SHA、seed7723、window17/overlap4、Lab、FP16一步、SP8/VAE height-shard8 和 H3 常驻条件，对照 1088p/1440p 的 raw 与 final。记录斑块、闪烁、时长、音画同步和峰值显存；FP32中间张量可能影响峰值。CPU回归不能证明所有伪影已消失。
