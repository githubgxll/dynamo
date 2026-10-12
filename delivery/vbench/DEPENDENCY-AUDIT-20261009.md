# VBench 依赖与构建复核记录（2026-10-09）

本次确定的直接故障是 **decord 官方 wheel 的文件名与包内平台标签不一致**。日志中的两套依赖检查均报告 decord 平台不兼容；同一日志里的 assets 为 `CANCELED`，表示被并行任务失败连带取消，不能据此宣称资产下载通过或再次失败。

本记录区分已确认事实、已做的局部修复和仍待验收的阶段。完整镜像、全部六维模型离线加载和 H100 评分尚不能因为本地检查通过就视为成功。

## 1. decord 的问题与修复边界

实际下载并校验的官方文件为：

```text
decord-0.6.0-py3-none-manylinux2010_x86_64.whl
```

文件名声明通用 Python 3 ABI；但其 `decord-0.6.0.dist-info/WHEEL` 内部却写为：

```text
Tag: cp36-cp36m-manylinux2010_x86_64
```

目标环境为 Linux x86_64 / CPython 3.10。因此安装器可依据文件名选中它，安装后 `pip check` 和 `uv pip check` 读取包内标签又将它判为 Python 3.6 制品。两种检查结果同时失败，与这一不一致吻合。PyPI 当前最新稳定版仍是 **0.6.0**，没有一个可直接升级的新版本解决此问题。

修复只针对经哈希确认的这一个官方制品：

- 将内部标签改为与官方文件名一致的 `py3-none-manylinux2010_x86_64`。
- 将发行版身份改为 `0.6.0+vbench1`，同步调整 dist-info 目录并重建 RECORD。
- 保留 Python 文件、18 个原生 ELF 库、其他载荷和随包许可字节；没有替换解码器、重编译 FFmpeg、移除依赖或绕过依赖检查。
- 导出原/新哈希、METADATA/WHEEL 差异、逐文件载荷哈希和 RECORD，发布锁使用修复后的明确身份。

| 制品 | SHA256 |
|---|---|
| 官方原始 wheel | `51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976` |
| `0.6.0+vbench1` wheel（最终版本，Windows/Linux一致） | `626dfe70549e12c3443c20a2530e6ee64be4e27ac6ce83bfef9725182d3f5772` |

这是有实际制品证据的标签修复，不是“把任意 Python 3.6 二进制包改标签”。已检查本包采用 ctypes/C API 路径、原生库为 Linux x86_64；这些静态证据仍不能替代目标 Linux 环境的加载和真实 MP4 解码测试。若载荷或原哈希变化，修复应立即停止重新审查。

本机真实 wheel 重打包两次得到相同哈希；55 个保留载荷文件（含18个原生 ELF）的字节不变，58条 RECORD 已校验。证据位于本机 `D:\AI\_workspace\VBench_Image_20260930\diagnostics-20261009\real-decord-repack-check.json` 和同目录的 `decord-wheel-audit.json`、`decord-repack-evidence`。本地二次重打包可复现不等于整个 Docker 镜像逐字节可复现。

2026-10-12补查时固定了ZIP的 `create_system=3`、时间戳、权限和成员顺序。第一次跨平台重打包还发现Windows的zlib-ng与Linux的zlib产生不同压缩字节，虽然所有成员内容相同，整包哈希仍不同。最终改用ZIP_STORED保存成员，消除压缩器差异；Docker镜像层仍会压缩。随后实际Windows/Linux重打包得到上表相同哈希；代码、原生库仍逐字节保持。只有这个最终版本用于本次提交。

## 2. 当前依赖是否较新

通过本机显式代理 `http://127.0.0.1:17890` 查询了26个直接包的官方 PyPI JSON。每项证据记录 URL、读取时间、HTTP状态、响应哈希、发布文件日期、Python要求及依赖声明；全部查询成功。以下“最新”是2026-10-09查询时的稳定版，不表示已经适配本镜像，也不表示清华镜像源已同步或 Runner 已能下载。

| 包 | 当前评测/构建版本 | 查询到的最新稳定版 | 处理判断 |
|---|---|---|---|
| torch | 2.3.1+cu121 | 2.14.1 | 与 torchvision/CUDA 成套评估，当前保留 |
| torchvision | 0.18.1+cu121 | 0.29.1 | 当前明确要求 torch==2.3.1，当前保留 |
| numpy | 1.26.4 | 2.5.3 | 最新要求Python≥3.12；3.10可匹配最高2.2.6，仍跨ABI |
| timm | 1.0.12 | 1.0.30 | 模型构造/预处理须回归，当前保留 |
| pyiqa | 0.1.13 | 0.1.16 | 最新要求Transformers≥5.0，不作局部升级 |
| transformers | 4.37.2 | 5.19.0 | 当前pyiqa精确要求4.37.2 |
| accelerate | 0.28.0 | 1.15.0 | 跨主版本，需旧API调用验证 |
| bitsandbytes | 0.43.1 | 0.50.2 | 最新要求Torch≥2.4，与当前pin冲突 |
| huggingface-hub | 0.23.5 | 2.2.0 | 当前Transformers要求小于1.0 |
| tokenizers | 0.15.2 | 0.23.3 | 当前Transformers要求小于0.19 |
| safetensors | 0.4.3 | 0.8.0 | 可另行评估，非本次根因 |
| opencv-python-headless | 4.10.0.84 | 5.0.0.93 | 最新在Python≥3.9要求NumPy≥2 |
| Pillow | 10.4.0 | 12.3.0 | 图像预处理变化须比较分数 |
| decord | 0.6.0 → 0.6.0+vbench1 | 0.6.0 | 本次仅修复已核实的发行元数据 |
| openai-clip | 1.0.1 | 1.0.1 | 无更新版本，保留 |
| omegaconf | 2.3.0 | 2.3.1 | 可列入独立小更新 |
| easydict | 1.13 | 1.13 | 无更新版本，保留 |
| imageio | 2.35.1 | 2.38.1 | 最新需Python≥3.11；3.10最高2.38.0 |
| packaging | 24.2 | 26.3 | 改变元数据/标签解析，独立验证后更新 |
| PyYAML | 6.0.2 | 6.0.3 | 可列入独立小更新 |
| tomli | 2.2.1 | 2.5.0 | 可列入独立小更新 |
| pip | 24.3.1 | 26.2.1 | 构建工具候选更新，非decord修复 |
| setuptools | 75.8.0 | 84.0.0 | 需验证旧sdist和构建API，当前保留 |
| wheel | 0.45.1 | 0.48.0 | 与构建工具一起单独验收 |
| uv | 0.8.22 | 0.12.24 | 解析/安装行为须回归，不能用于掩盖错误标签 |
| facexlib | 0.3.0+vbench1 | 0.3.0 | 原有headless依赖元数据修复保持不变 |

详细发布日期及 `Requires-Python` 不在此逐条重复，见本机 `D:\AI\_workspace\VBench_Image_20260930\diagnostics-20261009\dependency-versions.json`。带 `+cu121` 的 Torch 制品来自官方 CUDA 12.1 包源；本次版本审计查询的是 PyPI 基础版本元数据，不能当作CUDA wheel或GPU的实测结果。

当前直接pin在Linux/Python3.10适用条件下的已声明约束未发现冲突：pyiqa0.1.13要求Transformers4.37.2，后者要求hub≥0.19.3且<1.0、tokenizers≥0.14且<0.19，现有pin满足；torchvision0.18.1要求Torch2.3.1，现有版本也满足。

所以本次保持评分依赖组合，先修复确定的错误制品。整栈升级会同时改变CUDA、NumPy ABI、模型API和预处理路径，可能使新旧分数不再可直接比较。若后续升级，应另立镜像版本与完整锁，baseline和优化视频在同一新环境重测。版本审计也不替代漏洞扫描；旧版本的维护和安全风险仍须单独评估。

## 3. Dynamo分支和配置检查

本地VBench工作基于 `DingoRouter-base@77a1716960867bf1aba0af53c1993aa352506e9e`。本次查询到的远端分支头是：

| 分支 | 本次观察到的提交 |
|---|---|
| DingoRouter-base | `d64c67c0194b423cf7bfe0379082f2cf5ab88c10` |
| main | `049eece3f32f8d38dc1d8c52d4b793cb30e394bb` |

因此当前基线**不是远端最新分支头**。但本次比较的4个关键CI/context文件与旧base内容相同（忽略文件末尾换行差异），没有证据表明此次decord错误来自这部分Dynamo基线滞后。未完成整个远端分支差异审查；不宣称所有文件一致，也不为了消除“非最新”标记而把无关改动合并进来。

本镜像使用独立VBench环境，不复用H3的vLLM/Omni依赖。以下约束保持：

- VBench源码固定为 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。
- 构建目标为Linux amd64 / Python3.10 / CUDA12.1，评测依赖固定；基础镜像锁定digest。
- 普通Python包使用清华HTTPS源，Torch使用官方CUDA12.1源；不关闭TLS、不写入凭据。
- 当前是prepare阶段，沿用组内Dingo工作流和Runner，尚不发布Harbor。
- 镜像目标为hd04的 `openclaw/ai-dingo-testing/vench`，新Pod才使用 `jirx-` 前缀；镜像名不加个人缩写。

## 4. 本轮验证结果与尚待完成的验收

本轮完整本地测试结果为 **101 passed、2 skipped、75 subtests passed**。两个跳过项为Windows真实符号链接权限相关测试；模拟链接拒绝测试仍保留。真实官方wheel的下载、哈希、元数据差异、载荷保持与确定性重打包另外通过。

**2026-10-12已完成真实Linux CPU验证。** 在本机WSL Docker的隔离容器中使用Ubuntu22.04.5、Python3.10.12、Torch2.3.1+cpu、NumPy1.26.4、OpenCV4.10.0.84、pip24.3.1、uv0.8.22。整个验证容器为 `--network none`；解码脚本另启Python离线审计，未使用用户视频。

| 验证 | 实际结果 |
|---|---|
| 安装原始官方decord0.6.0 | pip/uv均退出1，逐字复现用户日志中的两条平台错误 |
| Linux运行最终修复脚本 | 新wheel与Windows重打包SHA256相同 |
| 安装修复后decord0.6.0+vbench1 | pip/uv均退出0，16个测试环境包的元数据检查通过 |
| 合成H.264/yuv420p MP4 | 96×64、8帧；FFmpeg、OpenCV、Decord实际打开并解码成功 |
| 非顺序及重复取帧 | Decord批次 `[0,3,7,3]` 的数量、尺寸、顺序、FPS检查通过 |
| 像素和Torch桥 | OpenCV/Decord相对FFmpeg的最大RGB差均为0；Decord→CPU Torch张量逐字节一致 |

完整结果为 `PASS_LOCAL_LINUX_CPU_DECORD_VALIDATION`，文件保存在 `D:\AI\_workspace\VBench_Image_20260930\diagnostics-20261009\linux-media-smoke\result-20261012-02`：`result.json`、`original-platform-check.json`、`corrected-platform-check.json`、`media-smoke.json`、网络审计、安装清单、系统包清单和重打包证据。第一次跨平台压缩差异的诊断保留在同级 `result-20261012`，不作为最终通过证据。

这只验证媒体链路所需的16包CPU环境；**没有完成目标CUDA镜像的106包完整环境、全部模型加载、真实H3视频评分或H100验证**。正式Dockerfile已加入相同媒体检查；Runner还必须重新完成完整依赖检查、全部模型的CPU禁网加载和准备制品导出，再依次通过许可、Harbor发布/拉取、H100 GPU可用、单样例六维离线评分。

当前用户日志只证明：当次106个包的依赖检查报告了decord这一项。它没有验证原生库动态加载、视频解码、GPU内核、全部模型资产或评分协议。

## 5. 已知的后续阻断与风险

| 项目 | 已知状态与下一步 |
|---|---|
| 许可发布门槛 | 当前7个checkpoint明确登记为UNKNOWN，组内策略拒绝UNKNOWN。该配置若不补充依据，publish会被阻止；这不是Harbor账号权限问题。AMT代码CC-BY-NC-4.0、pyIQA内的CC-BY-NC-SA/S-Lab条款还需结合公司用途与分发方式确认。先取得完整报告及授权/许可依据，再按具体组件更新记录。不能使用H3的soxr例外替代。 |
| decord原生库许可 | wheel内实际含FFmpeg、x264等原生库。不能仅凭decord顶层许可证覆盖所有捆绑库；完整许可清单与适用义务需复核。标签修复没有移除、修改或重新授权这些库。 |
| 构建端TLS连接 | Hugging Face/CDN曾多次TLS EOF。有限重试和缓存只缓解临时断连，不修复Runner出口。持续失败时交同事核实BuildKit构建容器的代理/出口；本机Git代理成功不能证明该路径可用。 |
| 系统包锁 | 基础镜像digest固定；apt安装结果有dpkg清单，但软件源尚未锁定历史快照。因此不同日期重建可能取到不同系统包，不能承诺整个镜像逐字节一致。 |
| 完整Python锁 | requirements.in不是完整传递锁。prepare成功后必须导入该次哈希锁与指纹；升级工具或依赖后重新prepare，不能沿用过期证据。 |
| 原生解码与模型API | 已通过上述局部CPU合成视频解码；目标CUDA完整环境、pyIQA等模型API和实际H3视频仍待Runner/H100检查。小合成视频不能覆盖长视频、不同编码参数或实际资源压力。 |
| GPU和正式评测 | CPU构建不证明H100评分成功。正式比较前先单卡单样例跑通全部六维且无外网依赖，再安排四卡任务；原始MP4及baseline/candidate相同环境要求不变。 |
| 结果保留 | K8s仍使用emptyDir，导出结果后再删除或重建Pod；不修改同事资源。 |

需要同事确认的最少事项是：持续TLS失败时的构建端出口配置，以及许可负责人对具体组件/权重的用途与内部分发授权记录。不需要在聊天中提供账号密码或令牌，也不需要重新确认已经授权的Harbor目标。

## 6. 下一步操作位置与命令

以下在**本机Windows CMD**执行，目录为 `D:\AI\dynamo-vbench`。无需进入K8s命令行或Pod，也无需申请GPU。

```cmd
cd /d D:\AI\dynamo-vbench
call delivery\vbench\01-check.cmd
```

看到 `PASS_LOCAL_ONLY` 后提交新修复：

```cmd
call delivery\vbench\02-submit-prepare.cmd
```

第二个脚本提交并推送用户工作分支，由组内Linux Runner构建prepare目标。不要重跑旧提交期待含有本次修复。当前步骤不推Harbor、不部署K8s。

新任务重点保存 `pip-check.txt`、`uv-pip-check.txt`、`dependency-checks.json`、`decord-patch.json`、`media-smoke.json`、`cpu-check.json`及许可报告。两套依赖检查均应退出0，媒体检查应为 `PASS_CPU_MEDIA_DECODE_ONLY`；如出现其他错误，应保留首个失败段落，不能用删除检查、忽略返回码或放宽哈希来通过。若prepare成功，按同目录README下载并导入准备制品；发布前仍需使实际许可报告通过。
