# 本机验证记录与剩余验收

日期：2026-10-28。

**状态：首轮远端构筑已执行，但两套镜像均因 APT 下载 502 失败；尚未成功发布。已准备共用下载修复，修后仍需重新提交远端构筑。** 本轮没有下载模型大权重、调用GPU、修改旧实验包或K8S资源。

已完成：

- 修后86项本地回归全部执行通过，无跳过：共享下载/解包及APT流程22项、SeedVR11项、FlashVSR+15项、提交/导入12项、CI发布26项。
- 20份Python文件按Python3.10语法检查；6份shell文件/工作流块通过Bash语法检查。
- Linux x86_64 / Python3.10依赖解析：SeedVR79项、Plus46项，记录精确版本与发行文件SHA256。解析不等于实际安装。
- SeedVR78份、Plus60份官方运行源码/配置按下载的固定上游内容校验；源码归档、Apex、FlashAttention及BSA/CUTLASS来源固定。
- 真实六份源码档案经过共享解包器校验；FlashAttention源码含CUTLASS，BSA子模块以精确gitlink单独获取。Plus prompt是实际Torch归档，不是LFS文本指针。
- 两个CUDA基础镜像现场读取并验证linux/amd64 manifest内容SHA256；原始记录随配方保存。
- 交叉审查修复了扩展导入前需先加载Torch、依赖编译缓存失效、逐帧像素hash格式不统一和导入证据完整性检查问题。
- 提交脚本使用临时Git索引审查封存内容，拒绝越界改动；初版已提交为 `d22d86dba3b2159dbf909e80eeab534cfb40d017`，对应 Actions #301。以下 APT 修复尚未提交或推送。
- 上传目标按用户最新约定改为`registry.hd-04.alayanew.com:8443/openclaw/vllm-omni`；两种tag生成及发布证据导入均使用该仓库。回归明确验证新地址，并拒绝旧仓库锁／旧仓库发布记录。CUDA基础与模型依赖保持原锁定组合；已重新封存。
- 新增12项APT回归用真实Bash／sed执行共享脚本，模拟`apt-get`和等待：检查官方源精准转换、配置保留、幂等、CA缺失、有限重试及耗尽、失败阻断、安装脚本不重复执行、参数透传和Dockerfile默认digest一致性。没有在本机执行真实APT，不将模拟成功表述为Runner出口恢复。

远端首轮记录：[Actions #301](https://github.com/githubgxll/dynamo/actions/runs/40675154119)。SeedVR2在安全索引下载失败；FlashVSR+在多个系统安装包下载失败，两者均为HTTP路径的502。共用修复见[README的APT说明](README.md#首轮-apt-502-修复2026-10-28)。修改后必须走`02-submit-build.cmd`上传新配方，不能用旧提交重跑代替。

本地完整检查输出按批次存入：

```text
D:\AI\dynamo-h3-experimental031\.h3-experimental-evidence\sr-native\local\checks-*\summary.json
```

`01-check.cmd`可重新生成报告。首次验证中Git Bash曾受本机受限进程环境阻止启动；随后在允许启动Git Bash的本地环境完成了真实`bash -n`，未以字符串匹配代替shell检查。

尚未完成，必须如实保持未验收：

1. Linux容器中的实际依赖安装和Apex／FlashAttention／BSA源码编译。
2. 构筑后最终镜像的UID1000离线import、媒体和pip检查，以及Harbor推送/回拉身份确认。这些已配置为CI门禁。
3. H100原生算子、SeedVR SP4通信、官方完整17帧3B/7B推理、FlashVSR+目标画布与容量验证。
4. 与现有Omni冻结结果的同输入/画布比较，原生RGB及成片的人工画质判断。
5. Gateway接入与压力验收，不属于这两个离线参考镜像的已实现能力。

当前没有依据保证官方3B消除C14，也不能保证7B或Plus在本业务胜出。两套镜像是为了获得可归因的真实对照证据。
