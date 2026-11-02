# H3＋SeedVR2：正式 Omni 0.32.0 升级配方

本配方继承已验收的 b315 镜像，联合升级正式 vLLM／Omni 0.32.0，并把 H3 卸载和 SR 优化／v2 指标补丁迁到新源文件。不重打旧工作树 Dingo wheel，不下载权重，不更改模型精度或实验拓扑。

## 先看什么

- `source-lock.json`：功能父镜像、源码、分支、上传仓库和正式 wheel 身份。
- `container/omni032-h3-sr/environment-policy.json`：准确升级项、原生字节保护和监测／默认 extras。
- `container/omni032-h3-sr/requirements.lock`：11 个新增 helper 的官方 URL＋SHA；不是整环境 transitive hash lock。core／Omni 两 wheel 单独下载和校验，厂商包从不可变父镜像保留。
- `expected-source-manifest.json`：正式 wheel＋9 项局部变换＋4 个 helper 的静态期望，实际安装必须独立匹配；不是已安装或 GPU 证据。
- `container/omni032-h3-sr/native-cli-argv.json`：完整现场 H3／SR 参数；K8S 共享 04 原字节快照保存在同目录 `native-cli-k8s-source.json`。额外 core profiler 仅解析探针，不加入现场启动。
- 工作区共享 `H3_SR_NextStage_20261102` 的 00／01／02／03／04 文档与机器合同：职责、接口、预构筑评审和后续单例验收。

## Windows CMD 本地检查

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\omni032-h3-sr\01-check.cmd
```

检查只运行本地源码／回归／JSON／shell 语法；没有 Linux 安装、Docker、上传或模型执行。需要已存在的 Python 工具环境；如位置不同，先设置 `H3_EXPERIMENT_PY`。受管自动化内的 MSYS 启动限制与真实 shell 语法失败要分别报告，不能删除检查。

镜像内部实际验收另包含：完整默认依赖和递归 extras、安装 origin／RECORD、实际原生和服务字节、全 Omni 源码、UID1000 禁网、六组 RMSNorm 数值、真实 CPU IPC／媒体及实际 0.32 parser 完整参数解析。模拟 CUDA 控制回归不代替 GPU 验收。

## 冻结与顺序提交

仅本轮唯一线上提交者执行。image／K8S／独立审计必须对同一最终配方给出 PASS，再按以下顺序操作；共享合同尚未完成时 freeze／submit 应被拒绝。

```bat
"%H3_EXPERIMENT_PY%" -B -X utf8 delivery\omni032-h3-sr\ops.py seal
rem 由各评审确认共享合同内的 freeze.tree 与 freeze.recipe_sha256
"%H3_EXPERIMENT_PY%" -B -X utf8 delivery\omni032-h3-sr\ops.py freeze
call delivery\omni032-h3-sr\02-submit-build.cmd
```

`freeze.tree` 排除 seal／approval 两个辅助文件，避免自引用；完整 prospective tree 单独记录。全部实际配方、共享输入和父 digest 仍由 recipe SHA 覆盖。任何实际修改使旧审核失效，必须先在本地重审。不要把上面的注释行当作已批准。

提交记录保存在 `one-build-journal.json` 和独占的 `requests/request-NNNN.json`，历史不删除。`request_count` 只计本地受控提交序号；GitHub `run_attempt` 与 Action 下载重试另记，均不固定为 1。新 workflow 接受正整数 attempt，不支持 dispatch、空提交或仅改审批文件的请求；旧 workflow 对本轮 marker 不构建。推送固定已审核完整 commit SHA，避免并行会话移动 HEAD。最终 tag 构建并 load，同一个 imageID 禁网验收后 push 一次，再按 digest 拉回核对。

取得上一轮确切结果前不追加请求。运行中、失联或发布结果 UNKNOWN 时先查原 run／Harbor；明确终态失败且没有发布时，可由唯一提交者在当前用户授权内修复、复核、重新冻结后顺序提交，不需要反复泛泛确认。成功后复用已有 digest；push 已成功而 pull 验证失败也不能当成没有发布。构建机准入、网络等基础条件持续失败且诊断无进展时报告原因，不无限请求。

再次提交前可用以下只读操作核实原 run 最新 attempt，保存结果并同步到共享合同。提交脚本还会在 push 前复查；若有人在期间手动重跑，状态变化会阻止新请求。

```bat
"%H3_EXPERIMENT_PY%" -B -X utf8 delivery\omni032-h3-sr\ops.py observe-previous --run-id 41388420551
```

CI 在 Harbor 登录后、构建前检查同 commit 的最终 tag：已存在就停止并复用／核验 digest；只有明确 manifest 不存在才继续，TLS、鉴权等异常按 UNKNOWN 停止。证据目录按 `run_id-run_attempt` 隔离，各阶段记录同一身份，不能复制旧 attempt 的 PASS。2026-11-02 首次 run 的 attempt1 在 Action 下载阶段 SSL 失败，attempt2 被旧条件跳过，均未进入 Docker；保留两条历史，不把三次 Action 下载误写成三次镜像构筑。

## 发布结果与后续功能验收

Actions artifact 包含 preparation／parent config、build 和同 imageID 的 final 验收、publication 和日志。只有 publication 的完整 `repo@sha256` 是部署镜像；源码 SHA、静态 source manifest、artifact ZIP SHA 与镜像 digest 分别记录。

新实际来源入口：`/opt/omni032-h3-sr/runtime.py verify`；八卡迁移入口：`/opt/omni032-h3-sr/gpu_roundtrip.py`。本轮标记为 `OMNI032_H3_SR_SOURCE_COMMIT`／`OMNI032_H3_SR_PARENT_IMAGE`，旧父层标记保持各自含义，不以 Pod env 伪造来源。

父层非 Python 许可证说明保留；`/legal/omni032-upgrade` 提供当前 Python NOTICE／SBOM，新增与版本变化包通过原许可证策略，不放宽 UNKNOWN 或 copyleft。旧 Python 清单属于父镜像历史。

发布后由 K8S 会话将真实 image／commit／cohort／源码清单／freeze 身份绑定到新包，只做一次新 Base C02，再从同一 Base 独立生成 1080 与 1440 两档 SR。该用例用于功能和媒体检查，不计算稳态性能增益，不恢复旧性能矩阵。
