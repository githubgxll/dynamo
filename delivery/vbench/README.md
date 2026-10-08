# VBench 镜像构建与验收操作手册

本包已准备具体脚本。当前配置为 **prepare**：首次由组内 Linux Runner 完成依赖解析、模型资产下载、CPU禁网加载和许可扫描，导出可复用版本锁，**不上传镜像**。导入并解决报告中的真实问题后，切换为 publish，才会构建并上传 Harbor。Windows 本机不需要 Docker、WSL 或 GPU。

## 镜像位置与执行环境

目标：`registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-testing/vench:v0.1.5-fd18b3d-cu121-r1-<提交前12位>`。

- `openclaw`：Harbor项目。用户已确认管理员权限，不再要求重复确认。
- `ai-dingo-testing/vench`：项目内带层级的repository名；Harbor显示方式由其界面决定，不需要手工创建操作系统文件夹。
- 保留用户指定的 `Vench` 拼写，因容器repository路径要求小写，写成 `vench`。
- 冒号后是版本tag；`0.1.5`对应所选VBench源码中的包版本，`fd18b3d`锁定提交，`cu121`对应CUDA12.1，`r1`为本打包协议版本；最终部署用完整digest。

| 位置 | 执行内容 |
| --- | --- |
| 本机 Windows CMD | 以下01–04脚本、Git提交与推送、下载制品、运行kubectl客户端 |
| 组内 Linux Runner | GitHub自动执行Docker构建和下载；无需登录构建机或SSH |
| K8s评价容器 | 发布后执行GPU检查与评分；由本机kubectl exec发起 |

如果窗口提示符为 `D:\AI\...>`，你操作的是Windows本机，即使窗口由TSM打开，也不是Pod内部。不要在Pod的Linux shell里执行 `cd /d`、`set` 或Git提交命令。

源码目录为 `D:\AI\dynamo-vbench`，本地分支 `vbench/six-dimensions`，基于 `DingoRouter-base@77a1716960867bf1aba0af53c1993aa352506e9e`；H3成功目录保持原状。用户触发的远端分支为 `DingoRouter-vbench-20260930`。脚本不用force push，不创建PR，不修改base/main。

## 第一步 在Windows检查

打开普通 **Windows CMD**，执行：

```cmd
cd /d D:\AI\dynamo-vbench
call delivery\vbench\01-check.cmd
```

此脚本解析代码和工作流、运行准备脚本回归检查、确认默认matrix只有VBench，并检查Git空白错误。使用已有 `D:\AI\_workspace\H3_Omni030_Image_20260929\tools-venv\Scripts\python.exe`；需要换解释器时先设置 `VBENCH_PY`，必须是带PyYAML和pytest的工具环境，而非目标GPU环境。

看到 `PASS_LOCAL_ONLY` 才继续。它不表示Docker构建、依赖安装、模型加载或GPU评分成功。记录保存到 `D:\AI\_workspace\VBench_Image_20260930`。

## 第二步 在Windows提交首次准备任务

先查看本次文件，再运行提交脚本：

```cmd
git diff --stat
git status --short
call delivery\vbench\02-submit-prepare.cmd
```

**02会真正执行Git add、commit和push，并触发GitHub任务。** 它只暂存本包约定的文件；检测到无关改动会停止。已有相同提交时不会重复commit。推送沿用本机命令级 `http://127.0.0.1:17890` 代理和Git已有凭据，不会索要或记录密码。

在 [GitHub Actions](https://github.com/githubgxll/dynamo/actions) 中打开该分支的最新 **Dingo runtime images**。本次应只出现VBench构建目标；不要手动选 `all`，因为那会请求其他框架。

Linux Runner自动执行：

1. 解析官方CUDA12.1基础镜像的linux/amd64 digest，再分别通过Buildx客户端与Docker daemon检查同一digest。daemon实际预拉取并核对平台和RepoDigests后才进入构建。
2. 为Python3.10生成完整传递依赖锁和哈希，安装后执行两套依赖检查。
3. 准备七个评价权重、固定VBench/DINO源码、配置和Torch hub缓存映射。
4. 在BuildKit `--network=none` 中检查文件和加载全部六维模型。这里没有视频评分，也没有GPU验收。
5. 复用仓库许可扫描器，补入第三方源码、模型资产及pyIQA真实许可记录。
6. 导出 `vbench-prepare-<完整SHA>-<attempt>` 制品。**不会登录Harbor或推送镜像。**

构建可在CPU机器进行，不需要申请H100。源码、wheel和权重下载均在准备/构建端，不在K8s运行期进行。

依赖采用独立Python3.10、PyTorch2.3.1/CUDA12.1环境，不继承H3的vLLM/Omni环境。保留pyIQA完整声明依赖；其中facexlib原声明普通OpenCV，而pyIQA要求headless版，两者会同时提供 `cv2`。脚本将facexlib固定wheel的这一条依赖改为headless并增加本地版本号，保留所有Python代码及许可，导出原/新wheel哈希、METADATA差异和文件清单。它没有修改六维评分算法，也没有用 `--no-deps` 隐去依赖冲突。

Python取包来源统一记录在 `container/vbench/package-sources.json`：普通包沿用组内SGLang已使用的清华HTTPS镜像，Torch/torchvision继续使用官方CUDA12.1索引。工具bootstrap、uv解析/安装和facexlib原始wheel获取读取同一配置。facexlib取包时的 `pip download --no-deps` 只下载一个带固定SHA256的原始wheel，不执行环境安装；最终uv步骤仍检查完整依赖。配置文件参与构建输入指纹，改变来源后必须重新prepare。

## 第三步 下载并导入准备制品

准备任务结束后，在运行详情页的 **Artifacts** 下载 `vbench-prepare-...` ZIP；不要下载误认为镜像的 `.dockerbuild` 文件。如果任务失败，制品可能只有workflow-status和base.lock；这不够继续发布，需要保留第一个失败步骤的日志。

在Windows CMD执行，ZIP路径替换为实际下载文件：

```cmd
cd /d D:\AI\dynamo-vbench
call delivery\vbench\03-import-preparation.cmd "C:\Users\admin\Downloads\vbench-prepare-实际下载文件名.zip"
```

脚本验证ZIP CRC、路径、源码身份和输入指纹后，将完整证据保存到本机交付目录，并导入五项：

- `base.lock.json`：基础镜像digest。
- `requirements.lock`：Linux环境实际解析的Python依赖与哈希。
- `assets.lock.json`：七个权重及源码/配置的实际SHA256与路径。
- `input-fingerprint.json`：本次构建脚本和策略的指纹。
- `preparation-license-status.json`：许可策略结果。

**导入不会commit、push或切换阶段。** 基础镜像、模型资产和Python锁确定后，评测程序的改动需要按新来源重新验证。apt系统包版本另存 `dpkg-packages.tsv`；本包不宣称完整OS软件源已做历史快照或镜像可逐字节重建。

重点查看制品中的 `cpu-check.json`、`pip-check.txt`、`uv-pip-check.txt`、`legal/license-status.json`。CPU检查应为 `PASS_MODEL_LOAD_ONLY` 且六项模型均已加载。许可报告与准备任务是否成功是两个独立结果。

## 当前已知的许可检查项

管理员权限解决Harbor推送授权。当前AMT/pyIQA及权重的许可适用性尚未记录，因此预计首次准备报告会包含待审项，发布入口会据仓库策略停止。

这是实际沿用的 [licenses.toml](../../container/compliance/policy/licenses.toml) 中 `unknown = "deny"` 等规则，以及已读取的AMT/pyIQA许可文件所致，不是另设Harbor权限确认。此包没有修改allow/deny、添加广泛例外或自动宣称已获许可。先取得具体组件、版本和完整报告，再按组内授权范围处理精确记录。

AMT代码为CC-BY-NC-4.0；pyIQA0.1.13包内另有CC-BY-NC-SA及S-Lab条款，与其Apache分类器不同。checkpoint专属适用条款暂记UNKNOWN。需要负责人确认公司研究测试与内部镜像分发是否覆盖，或提供已有授权记录。不能拿H3的soxr例外代替这些记录。

七个checkpoint目前明确登记为UNKNOWN；现有扫描器会先拒绝UNKNOWN，单纯在策略中添加例外不能解决它。取得来源许可或授权依据后，需更新 `container/vbench/scripts/license_report.py` 中对应资产的声明并保留依据，再按需要维护精确策略记录、重新prepare。完整代码与策略均参与输入指纹校验。

经审核修改策略或来源后，重新prepare以生成匹配的报告：

```cmd
call delivery\vbench\env.cmd
"%VBENCH_PY%" -B -X utf8 delivery\vbench\ops.py set-phase prepare
call delivery\vbench\02-submit-prepare.cmd
```

重新下载并导入对应新提交的制品；不要把旧报告的false手动改为true。即便手改，Docker发布阶段仍会重新扫描和校验。

## 第四步 在Windows触发正式构建与发布

**仅当导入的锁匹配、许可报告通过后执行：**

```cmd
cd /d D:\AI\dynamo-vbench
call delivery\vbench\04-submit-publish.cmd
```

04先检查锁和许可结果，再切换publish、运行本地检查、commit和push。Linux Runner读取已锁定基础镜像，按哈希安装依赖和资产，重做CPU禁网检查与许可校验，再使用既有 `REGISTRY_USERNAME/REGISTRY_PASSWORD` secrets推送目标。无需在聊天或代码中填写管理员密码。

成功后下载两类制品：

- `image-evidence-vbench-runtime-...`：`image.json`中的 `deployment_image` 是后续部署的完整 `repo@sha256:...`。
- `vbench-publish-...`：镜像内依赖、资产、CPU验证和许可证据。

制品ZIP的哈希不是镜像digest。Actions上传成功也不能代替K8s拉取和GPU评分验收。若镜像已推送但随后证据导出失败，先读取对应步骤和image-evidence，不要直接宣称整条链路通过。

## 第五步 在Windows生成K8s单卡验收清单

以下命令在 **Windows CMD** 执行。`kubectl`访问固定context，不改变当前默认context。首先从image.json复制真实digest引用；占位内容不能直接运行：

```cmd
cd /d D:\AI\dynamo-vbench
call delivery\vbench\env.cmd
set "VBENCH_IMAGE=这里粘贴image.json中的完整deployment_image"
set "VBENCH_PULL_SECRET=填写token-factory中现有拉取凭据的名称"
"%VBENCH_PY%" -B -X utf8 delivery\vbench\ops.py pod --image "%VBENCH_IMAGE%" --name "%VBENCH_POD%" --namespace "%VBENCH_NAMESPACE%" --gpus 1 --pull-secret "%VBENCH_PULL_SECRET%" --output "%VBENCH_DELIVERY%\smoke-pod.json"
```

只生成文件，不访问集群。如果已确认该namespace的默认ServiceAccount具备所需拉取凭据，可将 `--pull-secret ...` 替换为 `--use-serviceaccount-pullsecrets`；管理员推送权限不能直接替代Pod的拉取配置。脚本拒绝覆盖已有清单，下一次用新文件名/Pod名。

清单沿用已有H100自定义资源键、已用toleration和nodeSelector，不修改节点或同事资源。首次只申请1卡，资源请求为8CPU/32Gi内存，是否能分配需看集群实际状态；如果生成worker仍占满八卡，不要通过修改抢占/调度设置强行插入。

## 第六步 经部署安排确认后创建独立Pod并评分

仍在Windows CMD，通过kubectl执行：

```cmd
kubectl --context "%VBENCH_CONTEXT%" -n "%VBENCH_NAMESPACE%" create -f "%VBENCH_DELIVERY%\smoke-pod.json"
kubectl --context "%VBENCH_CONTEXT%" -n "%VBENCH_NAMESPACE%" get pod "%VBENCH_POD%" -o wide
kubectl --context "%VBENCH_CONTEXT%" -n "%VBENCH_NAMESPACE%" wait --for=condition=Ready "pod/%VBENCH_POD%" --timeout=300s
kubectl --context "%VBENCH_CONTEXT%" -n "%VBENCH_NAMESPACE%" get pod "%VBENCH_POD%" -o json > "%VBENCH_DELIVERY%\smoke-pod-observed.json"
kubectl --context "%VBENCH_CONTEXT%" -n "%VBENCH_NAMESPACE%" exec "%VBENCH_POD%" -- nvidia-smi
kubectl --context "%VBENCH_CONTEXT%" -n "%VBENCH_NAMESPACE%" exec "%VBENCH_POD%" -- /opt/venv/bin/python /opt/h3-eval/verify_runtime.py --device cuda --output /data/gpu-check.json
```

`Ready`只表示容器就绪。GPU模型检查通过后，选择一个实际原始MP4上传，原字节和SHA不变：

```cmd
"%VBENCH_PY%" -B -X utf8 delivery\vbench\ops.py upload-smoke --pod "%VBENCH_POD%" --namespace "%VBENCH_NAMESPACE%" --file "D:\实际路径\样例.mp4"
kubectl --context "%VBENCH_CONTEXT%" -n "%VBENCH_NAMESPACE%" exec "%VBENCH_POD%" -- /opt/venv/bin/python /opt/h3-eval/score.py --videos /data/input/sample.mp4 --output /data/smoke --gpu-devices 0
```

`exec --`后面的程序在 **Linux评价容器** 中运行；不要把 `/opt/...` 当成本机Windows路径。无需进入容器交互shell，不执行pip/git/wget。首次全部六维在一张GPU顺序完成，日志和评分保留在 `/data/smoke`；必须看到 `run-manifest.json` 中 `PASS_GPU_ALL_SIX_DIMENSIONS`。

如果断连或失败，不要直接重复执行同一个输出目录。先查看原日志和进程，保存失败证据，再选择新输出目录；GPU模型检查也拒绝覆盖已有JSON或审计日志，重试时使用 `--output /data/gpu-check-02.json` 等新文件名。本烟测入口不提供生产任务的断点调度。网络审计覆盖Python外联与子进程下载；CPU构建阶段还有BuildKit真正禁网。它不等价于修改K8s NetworkPolicy或隔离整个Pod。

## 第七步 下载证据再安排正式评测

```cmd
"%VBENCH_PY%" -B -X utf8 delivery\vbench\ops.py export-smoke --pod "%VBENCH_POD%" --namespace "%VBENCH_NAMESPACE%" --output "%VBENCH_DELIVERY%\smoke-results.zip"
```

导出包含评分、GPU模型检查及其网络审计、依赖/资产版本和许可报告，校验整包SHA256与ZIP CRC；不会删除Pod。每次导出使用新的容器内归档名；下载中断后保留本机 `.part`，改用新输出文件名重试即可。数据在emptyDir，删除或重建后会丢失。此小样例导出不替代旧正式任务的大文件分块续传。这里导出默认 `/data/smoke`；若评分改用其他目录，应另行导出该目录。

正式阶段可用 `--gpus 4` 生成另一只明确命名的评价Pod，并以 `--gpu-devices 0 1 2 3` 执行指标队列；不是TP，不需要八卡构建。需另做最长样例的内存/临时磁盘验证。

`score.py`提供单视频或平铺MP4目录的六维原始分数，**没有绕过或替代现有baseline/candidate配对锁**。业务FL11、Ref7和专项FL6、Ref6要分组，先由既有配对规则核对输入。封存PRIVATE.zip中的旧adapter仍需下一步接入新镜像入口；不得直接运行旧bootstrap，也不得将only-baseline伪造成candidate对比。当前交付重点是镜像与单例验收，不宣称四组正式对比已经完成。

## 失败时收集什么

| 最先失败的位置 | 应提供的非敏感信息 |
| --- | --- |
| 本机Git push | 完整报错、当前分支和HEAD；403看账号写权限，连接失败看本机17890代理 |
| Resolve base | 基础镜像tag和imagetools错误；这是Runner拉取层 |
| Check and pull locked VBench base with Docker daemon | 制品中base-pull目录；客户端检查、daemon拉取及本地镜像身份分别记录。无需自己SSH登录Runner |
| foundation中的Python工具安装 | 第一处ReadTimeout/SSL/HTTP错误及实际索引地址；索引超时后的No matching distribution不能证明版本不存在 |
| prepare_environment | uv依赖冲突正文、pip-check；不能靠删除默认依赖通过 |
| prepare_assets | 失败资产ID/host与HTTP、reason_type、errno或hash错误；不要发signed URL或token |
| CPU禁网加载 | cpu-check或首个异常，缺哪个文件/导入；不得转到Pod联网补装 |
| 许可检查 | legal/license-status.json和对应许可证；不发送账号密码 |
| K8s Pending/ImagePull | Pod describe中的调度或镜像拉取事件；不据此修改同事节点 |
| GPU维度失败 | 对应dimension.log、worker-evidence及run-manifest；保留原始输入与失败目录 |

## 基础镜像HEAD请求反复EOF：本次检查与操作

2026-09-30已对比H3成功任务（run181/job109755487555）与VBench重跑失败任务（run186/job109817114361）：实际Runner均为dingo-gxl-runner、同一Machine，Docker记录的代理均为10.201.136.68:1080。该值是日志中的观察记录，没有写入本包的代理配置。

两条流程的GitHub token都是contents:read，Buildx均使用docker driver。H3登录的是内部Harbor，日志显示复用Harbor builder，同时成功读取docker.io/vllm/vllm-openai的元数据。VBench prepare跳过Harbor登录，只拉公开nvidia/cuda；Harbor凭据不会授予Docker Hub权限。H3成功不证明另一个镜像路径在稍后的请求一定正常。

本机经127.0.0.1:17890匿名验证了报错中的精确CUDA digest，HEAD/GET均200且清单内容SHA匹配。因此镜像存在且公开可读取；Runner的EOF仍可能涉及代理/出网访问控制，但日志没有给出GitHub或Harbor权限不足的证据。本机检查不替代Runner检查。

本次改动为VBench增加 `Check and pull locked VBench base with Docker daemon` 步骤：

- 客户端读取同一digest，保留独立结果；不是只检查父tag。
- daemon按linux/amd64拉同一digest，只对EOF/超时等临时故障进行最多3次有限重试。每次拉取上限10分钟，间隔10秒；明确拒绝、镜像不存在或证书错误不循环重试。
- 拉取后核对OS、架构和RepoDigests。通过后VBench使用pull:false，允许使用同一个Docker daemon的本地内容；固定digest不变。pull:false并不保证BuildKit绝不访问registry。
- native框架原有流程保持不变，未更改账号、代理、TLS或GPU环境。失败报告经过脱敏，随prepare制品导出。

提交本次改动仍在 **Windows CMD**：

```cmd
cd /d D:\AI\dynamo-vbench
call delivery\vbench\01-check.cmd
call delivery\vbench\02-submit-prepare.cmd
```

这会生成新提交并触发新运行。GitHub里重跑旧run186仍使用旧提交4500a8e5f，不会带上这些新检查。当前仍是prepare，不发布Harbor。

如果新检查失败，下载 `vbench-prepare-...` 制品，查看 `base-pull/base-pull-summary.json` 和对应的客户端/daemon日志。失败制品不能用于03导入完整准备锁。

| 新证据 | 后续定位 |
| --- | --- |
| 客户端成功，daemon失败 | 同事检查Runner Docker服务的代理、DNS、出口和镜像仓库访问策略；不能只看Runner shell中的export |
| 客户端与daemon都失败 | 看错误类型核对目标仓库/清单、公共认证服务及共同出网链路 |
| 明确unauthorized/denied | 针对实际报错域名检查认证、失效的已有Docker Hub凭据或代理授权；不把Harbor管理员密码用于Docker Hub |
| 429/toomanyrequests | Docker Hub限流；由Runner维护者处理账号或镜像缓存策略 |
| daemon拉取和身份检查成功，随后FROM仍失败 | 单独检查BuildKit解析/缓存路径；已拉取不代表构建自动通过 |

交给同事的最少信息：对应job链接、失败阶段、base-pull-summary.json、脱敏错误和精确digest。请其确认这个Runner Docker实际使用的出口、Docker Hub相关域名（registry-1.docker.io/auth.docker.io及实际blob下载目标）是否可用，以及是否已有组内批准的Harbor基础镜像缓存。不要发送密码或令牌，不要随机替换基础版本或关闭证书校验。

## 2026-10-08：基础镜像之后的PyPI超时修正

用户贴出的run187日志已到foundation的工具安装，失败的是容器内访问 `https://pypi.org/simple/pip/`，这一层约99秒；不能把约20分钟的整场构建全部算成这一层。此前基础镜像拉取与系统依赖安装已经推进，但还没有验证VBench依赖或GPU。

旧脚本在bootstrap、uv compile/install及facexlib的PyPI JSON查询中分别使用默认或写死的PyPI来源。当前统一为版本控制的清华HTTPS索引配置，保留四个工具原版本及Torch官方cu121源。没有增加新的代理地址、关闭TLS或更换pip版本。bootstrap/单wheel获取使用15秒单请求超时、一次重试，尽早结束不通的请求；后续实际Runner连通性仍需CI确认。

本机通过既有127.0.0.1:17890代理已查到四个工具版本，并下载facexlib原wheel验证SHA256。证据位于 `D:\AI\_workspace\VBench_Image_20260930\diagnostics-20261008\tuna-mirror-probe.json`，它只证明本机测试结果，不代表Runner已通过。

执行位置仍是 **Windows CMD**，先检查，通过后再提交：

```cmd
cd /d D:\AI\dynamo-vbench
call delivery\vbench\01-check.cmd
```

```cmd
call delivery\vbench\02-submit-prepare.cmd
```

02提交新代码并触发Linux Runner，无需在Windows安装目标Torch、进入Pod或申请GPU。不要重跑旧run187期待加载本次修改。当前phase仍是prepare，不上传Harbor。修正放在apt层之后；同一Runner的缓存仍保留时，可复用此前系统包层，不保证缓存未被清理。

如果下次实际请求已变成清华镜像但仍超时，请提供该步骤首个错误和job链接，由同事核对 **BuildKit构建容器** 到 `pypi.tuna.tsinghua.edu.cn` 的出口、代理/NO_PROXY和访问策略。Docker daemon能拉镜像与RUN里的pip能访问索引是两条不同路径。Nexus只有在同事给出已确认的PyPI仓库地址及用法后再接入，当前不编造内部路径。

## 2026-10-08：LAION权重下载失败修正

最新日志的首个失败是 `laion-linear-head` 访问 `raw.githubusercontent.com` 时出现 `URLError`。旧诊断只保留外层异常类型，不能据此确定是DNS、TLS、代理还是连接中断，也没有权限拒绝的证据。并行的environment步骤标记为CANCELED，是assets失败导致构建终止，不能据此判断Torch依赖失败或已经安装完成。

前面的VBench/DINO源码已通过 `codeload.github.com` 下载。本次将LAION改为从官方同一提交 `6d122adad522ab246644d9dc1c6d7a3810ee255f` 的codeload ZIP提取精确的 `sa_0_4_vit_l_14_linear.pth`。这是明确记录的下载路径变更，没有自动选择不明镜像。只提取该权重，不复制归档中的其他模型、图片或数据集；评分路径与六维协议保持不变。

本机通过既有代理分别下载raw文件和归档，确认目标权重逐字节相同、不是LFS指针；未反序列化模型：

- 归档：1,822,818字节，SHA256 `0326ed1d15965dc82bde9dfc7799858785e1114915ffff2dd737566743ee4783`。
- 提取权重：4,071字节，SHA256 `2cd4e60f4f24ae3bcd57b847b13c1f3ba27edc28cc1a7f9ce74ee9f421243cba`。
- 目录中的 `sha256` 校验下载归档，`checkpoint_sha256` 校验提取权重；prepare/publish仍分别记录、检查下载和最终文件。若官方重打包导致ZIP字节改变，会停止并要求重新核验，不会静默接受。
- 本机证据：`D:\AI\_workspace\VBench_Image_20260930\diagnostics-20261008\laion-source-probe.json` 与 `laion-extraction-check.json`。本机验证不能替代Runner验证。

下载诊断现会保留底层异常类型和数字错误码，例如 `reason_type=gaierror` 或 `errno=...`，不打印可能带签名URL的异常原文。后续MUSIQ下载仍访问GitHub Releases及其CDN；本机HEAD已成功，但Runner实际下载尚未验证。

在 **Windows CMD** 先检查：

```cmd
cd /d D:\AI\dynamo-vbench
call delivery\vbench\01-check.cmd
```

看到 `PASS_LOCAL_ONLY` 后提交新代码：

```cmd
call delivery\vbench\02-submit-prepare.cmd
```

02会commit、push到自己的VBench分支并触发Linux Runner；不要重跑旧提交期待获得修复。本轮仍为prepare，不推送Harbor，不需要K8s命令或GPU。新日志应出现 `Acquiring laion-linear-head from codeload.github.com`。目录和脚本变更后需生成新的准备锁，不能沿用旧制品。

本次没有改动environment安装层；其uv下载缓存能否复用取决于Runner是否保留缓存。之前失败的assets层未完成，下一次会重新获取该层的资产。若新官方地址仍失败，提供第一个失败资产、reason_type/errno/HTTP状态及job链接，让Runner维护者检查该构建容器到对应域名的出口；无需到K8s重复代理测试。
