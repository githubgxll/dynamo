# H3／SR 与 Video Gateway 共用镜像构筑

本轮在个人分支 `Dingorouter-H3-vllm-omni-jirx` 发布一份共用镜像。源码基线为 `2d58b79df39a9335e2f850d3a83bf938610cc611`，运行父镜像采用已验证的 c88 digest。构筑配方继续使用 `container/h3-sr-production/Dockerfile`，运行时原生依赖和源文件锁位于 `delivery/h3-sr-production`。

本目录维护本轮独立的 `release-lock.json`、`input-seal.json` 和 `freeze-approval.json`。本轮提交标记为 `[h3-sr-video-gateway]`，发布 tag 为 `h3-sr-video-gateway-<commit12>`。请求、观察和发布证据写入 `.h3-experimental-evidence/h3-sr-video-gateway/`，早期 production 发布记录保留在原目录。

## 本地检查与提交

Windows CMD 在仓库根目录执行。

```cmd
call delivery\h3-sr-video-gateway\01-check.cmd
call delivery\h3-sr-video-gateway\02-submit-build.cmd
```

提交前需要完成协同复核，并将冻结审批绑定到本轮精确源码 tree、recipe 哈希和父 digest。`02-submit-build.cmd` 将推送个人分支并触发本轮 Actions。唯一提交者继续观察同一请求，明确失败后核对 Harbor 再顺序修正，成功后复用已发布 digest。

## 实际 Linux 镜像检查

CI 先 build／load 共用镜像，再以该 imageID、UID1000 和离线网络执行原生运行时检查。随后只读挂载 Gateway 测试目录及标准 `pytest-asyncio` wheel，运行配置、H3 适配器、ContextIR、生产兼容、输出策略和 artifact store 七组测试。测试包仅包含 `tests`，生产 Dingo 从镜像实际安装目录导入。

JUnit 结果需至少155例，覆盖四个历史 Windows 媒体失败用例，并且失败、错误、跳过均为零。具体测试选择及 wheel 哈希保存在源码及 `test-deps/pytest-asyncio.lock.json`。这些测试资产挂载到验证进程，最终运行镜像的依赖保持受原生锁核验。

`test_artifact_store.py` 同时覆盖最新 Gateway 的 Ceph 配额行为，相关 Windows 基线问题列入目标 Linux 验证。发布证据逐组核对七份测试文件都产生实际执行记录。

通过后，CI 发布已验证的同一 imageID，并按完整 digest 回拉检查身份。发布回执包括运行时验证、Gateway JUnit、测试源码与依赖哈希。真实 GPU 模型加载、三条业务路径和媒体质量由配套启动测试记录验收。
