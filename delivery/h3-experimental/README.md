# H3 实验镜像交付入口

当前方案已切换为最新核验的 `DingoRouter-video-gateway` 完整服务基线，升级到 vLLM 0.31.0 / Omni 0.31.0rc1，保留四份蒸馏适配器与 Ref8 12/3 修复。

完整命令、执行位置、变更原因、缓存与验收边界见 [GATEWAY-RUNBOOK.md](GATEWAY-RUNBOOK.md)。

提交分支仍为 `Dingorouter-H3-vllm-omni-jirx`，本次不会新建远端分支。先在本机 CMD 执行 `01-check.cmd`，检查通过后由你执行 `02-submit-build.cmd` 触发构建。
