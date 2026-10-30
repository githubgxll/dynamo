# SeedVR2 CPU后处理／真实D2H计时修复

本配方从已发布9547镜像派生，修正观测落点和完整性校验，**不升级框架、不改变模型数值、不增加权重、不改变复制／同步／SHM路径**。旧9547配方、旧06交接和旧RUN保留。

## 结论与根因

现场C08/S17/1080 single_encode媒体技术通过，80份模型rank记录完整，但10份API后处理回执因`get_device_properties(cpu)`失败。实际GPU→CPU早在SP8 rank0 Worker的IPC输出打包发生，API拿到CPU张量是正常协议。API中的原`.cpu().numpy()`表达式保持不变，它现在记录`cpu_video_numpy` CPU耗时，并将CUDA／allocator标null＋not_applicable；不能把这段叫D2H。

固定父镜像：`registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:9547a5e924383dcaba4cdb8cc4fc9c5b5b3b80c24a5290edbd64e4cf96992d69`。
分支仍为`Dingorouter-H3-vllm-omni-jirx`，Harbor仓库仍`registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm`，新tag前缀`omni031rc1-gateway-sr-metrics-repair-`。

## 新合同 sr-timing-memory/2

模型专用HTTP字段和legacy／single输出行为不变，模型pipeline／long_video／RMSNorm／VAE／H3代码字节均保留。只改三个安装文件：`sr_telemetry.py`、`long_video_support.py`、`diffusion/ipc.py`，维护完整RECORD与1830个父源文件清单。

| 收据 | 执行范围和含义 |
|---|---|
| windows/N/rank-0…7.json | 保留七个模型GPU阶段及窗口allocator；不代表这些rank都发生输出D2H |
| windows/N/transfer-rank-0.json | 真实发送rank0 Worker；阶段device_to_host_transport紧贴原copy调用；async使用传入side d2h_stream，复用原等待解析events，不插入额外sync/barrier |
| framework_output_pack | 上述transfer收据中的独立CPU墙钟包络，含原dict递归打包、pinned分配、必要等待、NumPy／SHM；不含IPC接收／排队 |
| video_tensor_pack_wall_ms | 更窄的目标video张量打包CPU包络，包含在framework_output_pack中；不与其相加 |
| windows/N/postprocess-PID.json | api_cpu_postprocess/cpu_video_numpy，仅CPUwall；rank/world/GPU和CUDA/allocator均null，明确local_d2h=not_executed_here并引用上游transfer |

必须按native请求、window、实际模型request、payload.video、发送进程/GPU及源码身份匹配。真实D2H、pack和CPU后处理分别完整；缺上游、错设备、重复producer、缺阶段、NaN或失败不能PASS，不能补0。重复回执作为错误保留，不覆盖旧producer。GET metrics新增scope_completeness，旧/1即使只改schema字符串也不能通过。

CPU pack墙钟与CUDA copy API stream区间是不同口径；events不是独立硬件DMA性能承诺。后台pack可能与下一窗forward重叠，**不重置allocator峰值**；观测到的进程历史峰值不能当D2H独占峰值。对同步传输若原有边界尚不能解析新event，则保留null＋METRICS_INCOMPLETE，不偷偷添加等待；本轮实际MP异步side-stream路径已有原同步完成点。

## 检查与提交

```cmd
cd /d D:\AI\dynamo-h3-experimental031
call delivery\seedvr2-metrics-repair\01-check.cmd
call delivery\seedvr2-metrics-repair\02-submit-build.cmd
```

同一tree已上传不重复提交；明确需要重复构筑时才用`03-run-build.cmd`。脚本仅提交既有分支；marker`[seedvr2-metrics-repair]`只触发新job。共享Runner不足100GiB的预作业门禁不能被本配方绕过，不清理他人缓存。

工具环境沿用`D:\AI\_workspace\SeedVR2_Optimization_20261030\cpu-venv`。本地无Torch时真实CPU张量测试显式未运行；Linux镜像设置`SR_METRICS_REQUIRE_RUNTIME=1`强制执行。严格FakeCUDA拒绝CPU设备，测试原stream/copy/sync调用次序、同步/async、无marker路径、失败传播、重复回执及实际producer→receiver合同。真实现场/1 fixture原样保留；用于/2测试的新D2H数据为明确的模拟数据，不是补造现场计时。

Linux最终容器以UID1000检查：实际CPU张量与原安装postprocess，CPU IPC往返，实际导入／源码／RECORD／完整包环境，继承C14/HSDP数值与媒体检查。旧媒体suite只有旧/1错误计数专用case被新/2完整性suite替代并明确记录；其余编码／音频／PTS／取消测试保留。

新H3 GPU门禁入口为`/opt/seedvr2-metrics-repair/gpu_roundtrip.py`，仍加载原HSDP gate全文并核对固定SHA，显式绑定完整修复子镜像来源验证器。旧入口会拒绝修复差异，不能跳过校验或伪造旧父身份。

## 交接与验收

独立交接目录：`D:\AI\H3_experiments_H100_HBM3_20261012\delivery\SeedVR2_Optimization_Ablation_20261030\08_Metrics_Repair_20261030`，旧06不覆盖。新包需新增IPC源码锁，更新3个源SHA／镜像身份、metrics_schema与launcher。metadata原`SEEDVR2_OPT_SOURCE_COMMIT`属于父bd3，不改写；本次标记为`SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT`。

首次新GPU验收：独立RUN固定同源C08/S17/1080 single_encode，确认10窗×8rank、10份唯一真实rank0transfer和10份CPU后处理完整；核验video/音轨和off/basic扰动，再开展同一新digest的正式矩阵。旧媒体成功可保留，但旧缺失D2H不能被事后推算填补，正式80矩阵统一重立新镜像基线。镜像构筑不操作Pod或旧RUN。

### 最终冻结身份补充

/2实际收据`actual_source_hashes`必须恰含sr_telemetry.py、pipeline_seedvr2.py、ipc.py三个64位SHA；请求source_hashes中各需唯一冻结匹配，空清单、缺项、无关项或多个后缀匹配均拒绝。rank→PID／Linux启动ticks／GPU UUID需跨窗口稳定。K8S新包保留相同跨窗检查。阶段commit a4d77的构筑用于阶段证据，最终部署以独立08中的最终digest和source_commit为准。
