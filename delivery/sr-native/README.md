# 原生超分参考镜像：SeedVR2 3B／7B 与 FlashVSR+ Base

2026-10-28。**本次新增两套独立的官方参考环境，先做 C14 画质归因和模型对照。使用 Dynamo 现有分支、Runner 与 Harbor 发布流程；不把不同 Torch 版本混入现有 Omni／Video Gateway 服务。** 两套镜像不含大权重，运行时使用只读权重挂载、独立可写结果目录。准备配方不代表已经发布或通过 H100 验收。

| 镜像 | 固定模型代码／环境 | 首轮拓扑和用途 |
|---|---|---|
| `sr-native-seedvr2-3b7b-*` | ByteDance-Seed/SeedVR `e4de8c24441a67e1b7df56abea10645059bb1185`；Python 3.10、Torch 2.3.0+cu121；匹配源码编译 Apex／FlashAttention | 3B 与标准 7B 共用镜像、分别启动；首轮 4 卡 SP4、17 帧；官方完整链对照 |
| `sr-native-flashvsr-plus-base-tc4-*` | OpenImagingLab/FlashVSR_Plus `b548a2cc2c929d3d4676f2973abe27876ec4dee7`；Python 3.10、Torch 2.7.1+cu126、源码编译 BSA | Base、BF16、TC4、官方流式入口；单卡原生路径 |

目标仓库统一为 **`registry.hd-04.alayanew.com:8443/openclaw/vllm-omni`**，按用户约定作为多模态模型镜像的共同仓库。两个 tag 前缀后附 Git 短提交、Actions run ID 和 attempt，避免同名误认：

```text
registry.hd-04.alayanew.com:8443/openclaw/vllm-omni:sr-native-seedvr2-3b7b-<commit12>-<run_id>-<attempt>
registry.hd-04.alayanew.com:8443/openclaw/vllm-omni:sr-native-flashvsr-plus-base-tc4-<commit12>-<run_id>-<attempt>
```

最终部署必须使用各自 `deployment-image.txt` 的完整 `repo@sha256:...`。CI的目标检查、推送、回拉和本地发布证据导入均绑定此仓库；两套原生环境仍各自使用已锁定的CUDA基础。

### Harbor仓库名称与已有镜像

`openclaw` 是 Harbor 项目，`vllm-omni` 是镜像仓库名，并非可随意改名的文件系统目录。标准Harbor的仓库更新接口用于修改描述，没有通用的原地重命名入口。若未来确需更名，使用Harbor的Copy功能将制品复制到新仓库，核验新地址，再更新构筑与部署引用；旧引用不会因此自动跳转。本次继续使用现有仓库即可，仓库名不限制里面镜像采用哪个推理框架。可在仓库Info页把描述改为“多模态模型镜像：vLLM-Omni、SeedVR2、FlashVSR+”。

用户提供的既有Omni0.31镜像：

```text
registry.hd-04.alayanew.com:8443/openclaw/vllm-omni@sha256:a3654da076b62c2bc46cea9ad560dcec9137140ecfed0fc6b240dd8451ee6f37
```

此引用作为既有镜像记录，未在本次重新拉取验收，也不作为两套原生SR镜像的父层。本次上传新tag，保留现有镜像与引用。

Registry主机、端口、Harbor项目不变，沿用Actions的`REGISTRY_USERNAME`／`REGISTRY_PASSWORD`。其实际CI账号需要对`openclaw/vllm-omni`拥有push/pull权限；本地管理员身份与Actions Secret中的账号须区分，本次没有读取或修改凭据。

官方依据：[Harbor仓库与描述](https://goharbor.io/docs/main/working-with-projects/working-with-images/repositories/)、[复制制品到其他仓库](https://goharbor.io/docs/2.10.0/working-with-projects/working-with-images/retagging-images/)、[Harbor v2 API](https://github.com/goharbor/harbor/blob/main/api/v2.0/swagger.yaml)。本次未操作Harbor端的仓库名、描述或已有制品。

## 直接提交构筑（Windows CMD）

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\sr-native\01-check.cmd
call delivery\sr-native\02-submit-build.cmd
```

`01` 只做本地检查；`02` 会重新检查、提交并推送到既有 **`Dingorouter-H3-vllm-omni-jirx`**，提交标记为 `[sr-native]`。它触发原 `dingo-router-ci.yml` 中的两项原生 SR 构筑，串行执行，避免首次编译同时挤占 Runner 磁盘／内存。其他 H3、C14、Lite 构筑不会被这个标记触发。不创建新分支，不修改 main，不强推。

进度查看：[Dynamo Actions](https://github.com/githubgxll/dynamo/actions/workflows/dingo-router-ci.yml)。运行名称含 `sr-native`，任务矩阵分别是 `seedvr2`、`flashvsr-plus`。不依赖网页出现 **Run workflow** 按钮。

若配方已上传且需要原样重跑：

```bat
call delivery\sr-native\03-run-build.cmd
```

`03` 以新空提交触发同一配方，要求本地已提交、与远端一致。构筑失败时先保留失败日志；代码改动应走 `02`，不把旧 tag 当作新结果。

### 首轮 APT 502 修复（2026-10-28）

Actions [#301](https://github.com/githubgxll/dynamo/actions/runs/40675154119) 的两项构筑均在系统包阶段失败：SeedVR2 获取 `jammy-security/universe` 索引失败，FlashVSR+ 下载多个 `.deb` 失败，HTTP 请求均出现 `502 Bad Gateway [IP: 10.201.136.68 1080]`。这发生在 Python 包安装／CUDA 扩展编译之前，不是模型依赖冲突，也没有 Harbor 权限失败的证据。

两份 Dockerfile 已共用 [install_system_packages.sh](../../container/sr-native/shared/install_system_packages.sh)：

- 官方 Ubuntu archive/security 源从 HTTP 改为 HTTPS，保留原域名、Jammy suite、所有组件及签名校验；NVIDIA 源不变。父镜像必须已有 CA 证书，缺失就停止。
- 索引更新使用 `APT::Update::Error-Mode=any`，任何索引缺失都不得继续使用旧索引安装。
- 索引与安装包下载分别最多尝试 3 轮，内部获取重试为 3、HTTP/HTTPS 连接及数据等待超时为 60 秒，轮间等待 5／10 秒。超时是连接／数据等待限制，不是整阶段总时限。
- 先 `--download-only` 下载完整安装包，再 `--no-download` 安装一次。下载重试期间不重新执行 update，避免容器清理钩子删掉已下载的包；不反复运行失败的安装脚本。
- Dockerfile 的 `BASE_IMAGE` 默认值与已锁定 digest 一致，消除 `InvalidDefaultArgInFrom` 警告。该警告不是此次 502 的原因。

本次不改变 Python／Torch／CUDA／上游模型代码或权重，不合并 main／Gateway，不修改代理账号。因为变更在最早的系统包层，后续层需要重新构筑；已有源码和 pip cache mount 仍可复用。**修改后的配方使用 `02-submit-build.cmd` 提交，不用 `03` 重跑旧配方。**

本地回归只验证脚本流程；本机没有 Linux Docker，不能证明 Runner 出口已经恢复。若 HTTPS 下仍持续返回 502，请将新 Actions 链接、失败域名及时间交给 Runner 管理者，排查构筑容器到 Ubuntu 官方仓库的代理路径；不要删除 security/universe、关闭 TLS／签名检查或忽略 APT 失败。

构筑成功后，分别下载两个 `image-evidence-sr-native-*` 制品，逐个导入：

```bat
call delivery\sr-native\04-import-evidence.cmd "D:\path\image-evidence-sr-native-seedvr2.zip"
call delivery\sr-native\04-import-evidence.cmd "D:\path\image-evidence-sr-native-flashvsr-plus.zip"
```

导入会验证文件哈希、Git 配方、实际镜像身份和 CPU 验收报告，写入 `.h3-experimental-evidence\sr-native\imports`。制品 ZIP 的哈希不是镜像 digest。

## 这次为什么不用升级现有 Dynamo／Omni

本轮只读核对远端（2026-10-28）：

- 实验分支：`977b1ca80a61305e1565ae0321925ba4767a8bc2`。
- `DingoRouter-video-gateway`：`9ccee2f56d0122be94d7436ae2f5fec96d15ea9b`。
- `main`：`049eece3f32f8d38dc1d8c52d4b793cb30e394bb`。

这两套镜像调用独立官方 SR 代码，不安装 `vllm`、`vllm-omni`、`ai-dingo`，因此没有必要为了本轮参考实验合并 main 或重构已成功的 Gateway 镜像。这里复用的是 Dynamo 构筑与发布设施。后续若把胜出模型接进 Gateway，需要另行实现任务协议、模型生命周期、输出与音轨交付；本镜像不宣称已实现该服务接入。

CUDA 父镜像已现场读取 Docker Registry 的 linux/amd64 manifest，并验证内容 SHA256：

| 用途 | tag 仅供辨识 | 实际构筑固定引用 |
|---|---|---|
| SeedVR | `12.1.1-cudnn8-devel-ubuntu22.04` | `nvidia/cuda@sha256:cc55d151af1e8e083f3210af753a5cfbcbc5455421531eb0459887026bb4699f` |
| Plus | `12.6.3-cudnn-devel-ubuntu22.04` | `nvidia/cuda@sha256:cb239b67719dfa32ec6b525b54c1b78559bebd51a47249dd702f6c5429372154` |

原始 manifest 在 [base-evidence](base-evidence/)。CI 拉取固定 digest，不每次重新解析可变 tag。Dockerfile 不声明需要联网拉取的 `docker/dockerfile` frontend；仍需 Runner 正常连通 Docker Hub、PyPI、PyTorch wheel 源和固定源码下载站点。

## 权重与运行入口

权重不进入 Git、不在 Actions 下载、不打入镜像。独立准备机器按锁下载或验证既有完整文件，然后将目录只读挂到实验 Pod。K8S 不需要联网 GitHub，也不需要对权重目录有写权限。官方 `.pth` 不先转 FP16，不复用 Omni 生成的 latent 作为参考链输入。

下载工具及资产锁：

```bat
call delivery\sr-native\env.cmd
"%H3_EXPERIMENT_PY%" container\sr-native\shared\acquire_assets.py --lock container\sr-native\seedvr2\assets.lock.json --output D:\AI\models\seedvr2-native
"%H3_EXPERIMENT_PY%" container\sr-native\shared\acquire_assets.py --lock container\sr-native\flashvsr-plus\assets.lock.json --output D:\AI\models\flashvsr-plus-native
```

以上是可选的独立权重准备步骤，会下载大文件，**本次准备没有执行**。已有文件用同一命令附加 `--verify-only` 检查。工具检查固定 revision、字节数、SHA256；保留断点并有限重试，既有文件不匹配时停止，不覆盖。选定单项可使用资产锁里的 ID：`--select ID1 ID2`。具体挂载文件布局、GPU 原语检查、完整链命令见：

- [SeedVR2 原生运行说明](../../container/sr-native/seedvr2/README.md)
- [FlashVSR+ 原生运行说明](../../container/sr-native/flashvsr-plus/README.md)
- [首轮对照与交接条件](ACCEPTANCE.md)

## 本次重点避开的坑

1. **SeedVR 的 Apex 版本混搭。** 官方 requirements 指定 Torch 2.3，README 提供的某个 Apex wheel 面向 Torch 2.4。这里按选定 Torch 编译 Apex 和 FlashAttention，保留原生 fused 算子；没有把 import 成功当成 GPU 正确。
2. **FlashVSR+ 的 SDPA 并非高分辨率保底。** 固定源码的密集 mask 有大小限制，目标画布可能超过上限。这里明确选 BSA；编译或执行失败即报告失败，不静默切换 SDPA 或旧 FlashVSR v1.1。
3. **“较新”依赖不等于官方参考。** SeedVR 选择官方 requirements 路线，Plus 选择官方允许的 Torch 范围；分别锁定依赖、保存安装清单。没有把 Omni 的 Torch 2.13 强塞进去，也没有延用 Torch/NCCL 元数据豁免。新环境 `pip check` 必须通过。
4. **补帧与裁切会破坏归因。** SeedVR 3B 的 20 heads 不支持直接照搬 SP8；SP4 的17帧符合时间约束。Plus 宽高需128整除，因此跨模型轨使用1920×1152，不能直接与旧1840×1104比画质后归因模型。
5. **官方 NaResize 的 `mode='area'` 是目标面积模式。** 重新核读源码后确认插值默认是 BICUBIC，不能误写成 PyTorch 的 area 插值；本配方调用官方原函数，不根据误解重写预处理。
6. **构筑网络错误不能靠换包版本掩盖。** 固定父镜像拉取有有限重试；源码／资产做完整哈希检查；包安装保留缓存与更长超时。持续 EOF、TLS 或出口故障仍需 Runner 管理者处理，不关闭 TLS 校验、不使用未知镜像站。
7. **首轮原生扩展编译较重。** 两套CUDA开发基础与算子编译首次耗时、占盘较大；没有GPU也可为H100编译，但运行正确性需真实GPU验证。Runner的100GiB剩余空间准入仍由管理员管理，脚本不自行清理共享Docker缓存。
8. **LICENSE／SBOM 与正式合规审批不同。** 保留上游许可证、Python METADATA/direct_url、系统包清单和包级CycloneDX清单；报告是来源与声明证据，不把新参考环境冒充已通过原生产镜像的许可白名单审批。

本机检查结果与限制记入 [VALIDATION.md](VALIDATION.md)。实际Linux构筑、CUDA算子、17帧完整推理、跨模型媒体与人工画质结果分开验收。
