# H3／SeedVR2 生产共用镜像构筑交付

**本轮只生成一张 FL／Ref 共用的生产运行镜像，继承已验收的 c88 父镜像。** 两档区别由冻结 profile 表达，不分别重建镜像。保留父镜像的 Python、Torch、CUDA、Omni 与 Gateway 依赖；构筑仅复制、应用、检查本轮生产代码。模型大权重继续只读挂载。

## 固定身份与边界

- 父镜像：`registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:c88bef388ed25359c78c48ae841736049100d7447c572678992699ae6291e25d`。
- 父 imageID：`sha256:cd87bc74b19402c536fc775ea2ebbc15f31aad0013756e8d8c448afe1c895a6b`。
- 配方基础：`b42388a1a65067ed92dfe64f9364c4b044a8d666`；仅推送既有分支 `Dingorouter-H3-vllm-omni-jirx`，不改 main、不建新远端分支、不强推。
- 发布仓库：`registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm`；tag 为 `h3-sr-production-<实际提交前12位>`。最终部署使用 `deployment-image.txt` 中的完整 digest。
- 专用工作流：[h3-sr-production.yml](../../.github/workflows/h3-sr-production.yml)，唯一提交标记 `[h3-sr-production]`。旧 Omni／Router 的 push 路由明确排除此标记。
- 本轮不安装或升级依赖、不构筑／发布 builder、不打包大权重。CPU 验收不能替代 H100、两档生成、媒体与压力验收。

## 本地检查与最终冻结

在 Windows CMD 中执行，工作树可以是本轮隔离的 detached worktree：

```bat
cd /d D:\AI\dynamo-h3-sr-production-20261105
call delivery\h3-sr-production\01-check.cmd
```

该命令只做本地回归与语法检查，日志在 `.h3-experimental-evidence\h3-sr-production\local`。零测试、失败或跳过不会被算作通过。可用 `set "H3_PRODUCTION_PY=完整Python路径"` 指定已有 Python，不会自动安装环境。

真实ASGI回归需要 FastAPI／Starlette／Pydantic 等Web依赖。若全局Python缺少它们，可复用本机既有独立target，避免全局重装：

```bat
set "H3_SR_CHECK_DEPS=D:\AI\_scratch\h3sr-web-cpu-01363"
set "PYTHONPATH=D:\AI\_scratch\h3sr-gateway-cpu-20261105"
call delivery\h3-sr-production\01-check.cmd
```

第一目录提供真实Web栈，第二目录提供本机独立的Pillow／PyAV；缺少媒体依赖而skip不算通过。目录只用于本地测试子进程，不进入镜像，也不修改镜像依赖。其他机器应指向其已有、版本核对过的独立环境，或使用包含这些依赖的已有Python。本机Python3.14媒体wheel与镜像Python3.12版本不同，已明确记录；不能把本地通过视为Linux实际环境验收。Bash必须能够启动，shell语法未实际检查时不会报告PASS。

所有相关实现、复核及遗漏修正完成后，由唯一提交者冻结：

```bat
python -B -X utf8 delivery\h3-sr-production\ops.py seal
```

`input-seal.json` 记录成功父 digest、相对基础提交的逐文件哈希／删除项、`recipe_sha256` 与 `review_tree`；review tree 排除 seal 和 approval 自身以避免循环哈希。最终 `freeze-approval.json` 需包含同一 `recipe_sha256`、`review_tree`、`parent_image`、`status=READY_FOR_BUILD`，以及 root／K8S 的 PASS 复核。这里只是已有授权任务的内部复核，不是替用户扩大线上权限。代码变更后重新运行受影响检查，再重新冻结；不能先上传阶段产物再补最终审阅。

## 明确提交构筑

```bat
call delivery\h3-sr-production\02-submit-build.cmd
```

**此命令会进行 Git commit／push；push 本身就是构筑请求。** 它检查封存、最终复核、真实远端分支基线和上轮结果，使用隔离临时索引构造并审查实际导出的 Git tree；正式暂存前备份原索引。不提交无关工作、不用 `git add .`，不在全局设置 safe.directory。

提交前的本地检查按 recipe 哈希复用；字节改变时重新检查。没有新配方时不制造空提交。本轮首次提交不要求历史不存在的 run／Harbor观察；有上一轮请求时才执行对应结果确认。

## CI实际行为

1. 核验分支／事件／实际提交、唯一标记、完整源码封存与最终复核。Actions attempt 可为任意正整数，不设置 attempt=1 门禁。
2. 查询稳定 tag：明确不存在才允许构筑；TLS／EOF／认证／限流与含糊404都记 UNKNOWN并停止。已有 tag 则回拉并取得不可变 digest，跳过 build，验收后复用原制品。
3. 固定拉取 c88 父 digest，核对平台与父 imageID。只有一个无 push 的 build，并 load 到当前 Docker daemon，不生成远端中间镜像。
4. 核对完整父层、原生环境、服务默认配置与子提交身份。以实际 imageID、`--network none --user 1000:0` 执行最终实际模块／源码与 CPU 检查；写入本次独占的 `/tmp/h3-sr-production-evidence` 挂载目录，避免写父层只读位置或共享 uv cache。
5. 仅推送这一个实际验收 imageID，随后按 digest 回拉并再次比对 imageID。若已有合格制品则 push数为0。
6. 在可能的回拉SSL失败前先保存 `publication.json` 的 `PUBLISHED_PULL_PENDING` 与真实 digest；末尾另存实际 Harbor 状态及完整证据。末尾失败不能当作镜像没有上传。

工作流复用 `dingo-runtime-images-${{ github.ref }}` 互斥组、`cancel-in-progress:false`；不取消其他既有构筑。单独的旧工作流可能显示 skipped，判断实际任务需看 **H3 SR production common runtime** 的 run与job。

## 续查、成功复用及失败后顺序重提

本地 journal 在 `.h3-experimental-evidence\h3-sr-production\local\request-journal.json`，逐次请求与失败历史保留。观察原run：

```bat
python -B -X utf8 delivery\h3-sr-production\ops.py observe --run-id 实际ActionsRunID
```

观察覆盖该run的全部attempt；早期attempt已发布的事实不会被后续失败／skipped覆盖。排队、运行、网络失联、结果UNKNOWN均先续查原请求，不追加push、rerun、dispatch或空提交。

成功后，下载当前专用工作流的 artifact并解压，导入其 `publication.json`：

```bat
python -B -X utf8 delivery\h3-sr-production\ops.py record-publication --receipt "D:\实际制品目录\publication.json"
```

只接受本轮提交对应的已推送且按 digest 回拉、CPU验收通过的回执。后续部署、调试和验收都引用这个完整 digest，不为同一结果重建。

明确失败后，先从该run最新attempt的artifact导入实际Harbor观察：

```bat
python -B -X utf8 delivery\h3-sr-production\ops.py record-outcome --receipt "D:\实际制品目录\publication-observation.json"
```

只有原run明确完成且失败、全部attempt未发现已成功发布，并有最新attempt／同commit的 `ABSENT_CONFIRMED` 实际注册表观察，才允许修复后在原授权范围内顺序重提。`EXISTS` 先核对并复用已有digest；`UNKNOWN` 继续定位，不把认证／SSL失败伪装成未发布。若构筑机hook在checkout前失败而没有artifact，需由有权限的操作者核对实际Harbor，不能通过删journal绕过。

完成失败定位、修正、受影响检查和最终复核冻结后，再使用 `02-submit-build.cmd`。不清空历史、不重置attempt，不主动无限重试。发布本身成功但回拉失败时，应验收已有稳定tag／digest并复用，不新发一个源码提交制造第二份相同修复镜像。

## 构筑端与运行端职责

最终同Pod入口及必填配置见 [runtime-contract.json](runtime-contract.json)。启动器按H3→SR→Dingo worker顺序启动同一容器内三个受管进程；`--gateway-command-file /etc/h3-sr/gateway-command.json`为必填参数，文件是Dingo参数JSON数组，不是shell命令。双模型health通过后仍须匹配本次epoch、profile SHA和Gateway进程出生身份的注册回执，8125 `/ready`才成功。管理接口只监听127.0.0.1，使用容器内exec探针。完整启动上限为22950秒，含Ref身份请求与Gateway注册预算。

部署方提供Dingo地址、发现/请求/事件连接配置、鉴权和模型池绑定；显式启用`adapter_options.enable_sr=true`。`stages`还须注入新发布镜像的完整`H3SR_IMAGE_DIGEST=仓库@sha256:...`，不能用父digest代替。worker除了Base与转换后的FP16目录，还须只读挂载`/models/SeedVR2-3B`供verify-only核验。不要将ConfigMap目录挂到`/opt/h3-sr`遮住镜像代码。

Dingo部署参数须显式包含`--output-modalities video`；其上游默认是`text`，不能只按模型别名推断视频注册类型。镜像内的真实Dingo解析检查覆盖该参数。

当前Omni入口没有`--component`参数，组件固定为`backend`。部署通过`--namespace 你的命名空间 --endpoint generate`配置名称，注册目标为`命名空间.backend.generate`；`--endpoint`在该Omni入口传叶子名称，不能直接把通用帮助中的完整`dyn://...` URI代入。第一次Linux检查已拦截并纠正了错误的component样例，Harbor未产生该失败候选。

本轮另修复SeedVR2 W65输入的容器时长量化误拒：65帧Matroska可能记录2709毫秒，在24FPS下估算为65.016帧。时长预检仅容忍一个容器timebase tick；逐帧解码的65帧硬限制、空间／clip预算及W33／W65配置保持不变。不能用此修复为66帧窗口或超预算画布放行。

控制状态在失败、取消或未知结果后保留阻断；容器重启不会自动清空旧任务或重放。部署方确认旧任务终态后，采用明确的新Pod／epoch／工作卷恢复，并保留旧证据。

构筑证据包括父镜像、Git提交、recipe／review tree、完整有效源码清单、真实依赖／非root CPU测试和发布回拉身份。K8S交付另查同一生产profile、Base与Seed权重来源／只读挂载、FP16转换来源及完整性、八rank身份、GPU运行、输出与媒体验收。不要将构筑成功、Ready、生成成功、人工画质和负载验收混为一个PASS。

本地工具不自动访问K8S、不重启现有Pod、不删除共享Runner缓存。Runner job-started GC不足100GiB时，需恢复主机准入条件；仓库内首步清理或换仓库都来不及解除该hook。
