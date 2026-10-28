# 本机验证记录与剩余验收

日期：2026-10-28。

**状态：配方与本地回归通过；尚未实际构筑或发布镜像。** 本轮没有下载模型大权重、调用GPU、修改旧实验包或K8S资源。

已完成：

- 74项本地回归全部执行通过，无跳过：共享下载/解包10项、SeedVR11项、FlashVSR+15项、提交/导入12项、CI发布26项。
- 19份Python文件按Python3.10语法检查；5份shell文件/工作流块通过Bash语法检查。
- Linux x86_64 / Python3.10依赖解析：SeedVR79项、Plus46项，记录精确版本与发行文件SHA256。解析不等于实际安装。
- SeedVR78份、Plus60份官方运行源码/配置按下载的固定上游内容校验；源码归档、Apex、FlashAttention及BSA/CUTLASS来源固定。
- 真实六份源码档案经过共享解包器校验；FlashAttention源码含CUTLASS，BSA子模块以精确gitlink单独获取。Plus prompt是实际Torch归档，不是LFS文本指针。
- 两个CUDA基础镜像现场读取并验证linux/amd64 manifest内容SHA256；原始记录随配方保存。
- 交叉审查修复了扩展导入前需先加载Torch、依赖编译缓存失效、逐帧像素hash格式不统一和导入证据完整性检查问题。
- 提交脚本使用临时Git索引审查封存内容，拒绝越界改动；本轮未提交或推送。
- 上传目标按用户最新约定改为`registry.hd-04.alayanew.com:8443/openclaw/vllm-omni`；两种tag生成及发布证据导入均使用该仓库。回归明确验证新地址，并拒绝旧仓库锁／旧仓库发布记录。CUDA基础与模型依赖保持原锁定组合；已重新封存。

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
