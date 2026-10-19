# H3 实验镜像交付入口

当前方案使用固定的 `DingoRouter-video-gateway@e8c9195672f7` 完整服务基线，升级到 vLLM 0.31.0 / Omni 0.31.0rc1，保留四份蒸馏适配器与 Ref8 12/3 修复。提交时验证固定 commit 仍在上游历史中；上游正常追加提交会提示，不自动合并，也不因其更新阻断这份已封存配方。

完整命令、执行位置、变更原因、缓存与验收边界见 [GATEWAY-RUNBOOK.md](GATEWAY-RUNBOOK.md)。

提交分支仍为 `Dingorouter-H3-vllm-omni-jirx`，本次不会新建远端分支。先在本机 CMD 执行 `01-check.cmd`，检查通过后由你执行 `02-submit-build.cmd` 触发构建。
