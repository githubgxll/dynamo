# SeedVR2 阶段观测、单次交付编码与窗口消融镜像

本配方从已成功父层派生：**保留 vLLM-Omni0.31.0rc1、Gateway、RMSNorm局部FP32、C14诊断、H3 HSDP卸载／DiT入口／fastpath；仅增加SeedVR2 Python代码。**不安装／升级Python包，不下载权重，不重编Dynamo。GPU、媒体与画质验收另行进行。

## 固定身份与文件

- 父层：`registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:45b3ec519faec85ac4a40ff8766e033704bc92623f97fda16610efdc85e71e0f`。
- 父层构筑：[40827996676](https://github.com/githubgxll/dynamo/actions/runs/40827996676)，source commit `430c62be0e38179535b4a6a1aa38dc67e8367d00`。
- 分支沿用 `Dingorouter-H3-vllm-omni-jirx`；上传仓库 `registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm`；tag前缀 `omni031rc1-gateway-sr-optimization-`。
- [source-lock.json](source-lock.json) 固定父层和版本；[source-patches.lock.json](../../container/seedvr2-optimization/source-patches.lock.json) 固定两处前后SHA及两份新增helper。
- Dockerfile在 `container/seedvr2-optimization/Dockerfile`；不重新应用父层H3修复。父层整包源码、RECORD和身份先验证，再精确改动两文件并新增helper。

## Windows CMD入口

本机工具环境位于 `D:\AI\_workspace\SeedVR2_Optimization_20261030\cpu-venv`。它只供CPU合同检查，不属于镜像依赖。需PyYAML、Jinja2、packaging、PyAV、numpy、FastAPI、requests、regex、imageio-ffmpeg、python-multipart；可用 `H3_EXPERIMENT_PY` 指向具备这些包的Python。镜像实际PyAV16.1与Torch数值测试在Linux父运行环境中强制执行，不能由本机较新PyAV的通过替代。

```cmd
cd /d D:\AI\dynamo-h3-experimental031
call delivery\seedvr2-optimization\01-check.cmd
call delivery\seedvr2-optimization\02-submit-build.cmd
```

01仅本地检查。02会检查、保存原index、提交并推送上述既有分支，提交标记`[seedvr2-optimization]`只触发新配方。相同tree已上传时不伪造新上传；明确需要重跑才使用：

```cmd
call delivery\seedvr2-optimization\03-run-build.cmd
```

03创建空提交以使用当前有效push触发，不依赖main上是否存在Run workflow按钮。不改main，不开新分支，不强推。

开发者确实修改实现／锁后，先审核完整差异，再执行下面的重新封存；02不会自动接受未审核的文件漂移：

```cmd
call delivery\seedvr2-optimization\env.cmd
"%H3_EXPERIMENT_PY%" -B -X utf8 delivery\seedvr2-optimization\ops.py seal
```

## 运行接口

只扩展模型专用 `/v1/seedvr2/restore-long` 表单，通用`/v1/videos/sync`尺寸语义不变：

| 字段 | 规则 |
|---|---|
| `size` | 内部模型尺寸，继续16对齐与原clip/frame准入 |
| `encoding_mode` | `legacy`默认，或`single_encode` |
| `delivery_size` | single必填WIDTHxHEIGHT；偶数，短边1080或1440，与源和模型精确等比例；legacy不接收 |
| `window_frames` | 可选显式17／33／65；省略保留旧自动行为。显式大于容量直接400，不能自动降窗 |
| `measurement_mode` | off默认／basic／diagnostic |
| `measurement_context` | 小JSON：run_id/batch_id/case_id/action_id/phase/image_digest/source_hashes/model_manifest_hash/expected_world_size；不接收文件路径 |

5:3源片的1080档：`size=1840x1104,delivery_size=1800x1080`；1440档均`2400x1440`。单次模式在内部RGB融合后再Lanczos缩放，以libx264 medium/crf18/yuv420p完成唯一完整视频有损编码；不继承旧raw的zerolatency/bf0，记录实际B帧／profile／SAR与库版本。legacy保留veryfast/crf18/zerolatency/bf0原服务writer；外部控制器仍按旧链做medium交付编码。

窗口内部无损编码继续保留，不能宣传整个服务只执行一次编码。原生服务AAC回封继续原行为；single外层必须核验后video copy、Base原audio copy。外部remux、最终完整解码和音轨校验由新控制器实现，不能用少做验收制造提速。

`GET /v1/seedvr2/restore-long/{job_id}/manifest`返回已完成服务输出规格、writer与窗口映射。输出包含model_size与delivery_size两种几何，不再混用width/height。

## 指标与验证边界

API与所有SR worker须共享UID1000可写的绝对目录，显式传`SEEDVR2_METRICS_DIR`。镜像缺省不设置该目录、不启用观测。basic无逐阶段同步；在当前stream窗口结束解析events。diagnostic额外阶段等待单列，不进入正式basic性能分母。保持RNG、数值运算、dtype与模型参数；移除观测包装后pipeline AST与父层相同。

指标schema为`sr-timing-memory/1`。服务job UUID为sr_request_id；每窗独立model_request_id，模型实际request_id另存，8rank身份必须一致。每rank写独立原子JSON；D2H由实际后处理进程单独回执。allocator窗口峰值与D2H进程历史观测峰值分别命名，后者不冒充D2H阶段峰值。

`GET /v1/seedvr2/restore-long/{job_id}/metrics`返回请求索引和小JSON包：`files`按相对路径索引JSON或JSONL数组，另有metrics_status、missing_rank_records、inconsistent_rank_records。8rank阶段、events／allocator和D2H缺失均标METRICS_INCOMPLETE；不能补0或重新提交成功推理。off没有新增指标文件，metrics GET返回404。指标完整性与generation状态、媒体及人工质量分别判断。

NVML整卡与逐PID、CPU RSS/cgroup由外部控制器采样。NSpid仅记录可见信息，不保证宿主PID映射可用；不可读须标未知，不能把整卡used分摊到SR。阶段活动和、rank最大值、完整E2E不能相加。原生服务server_sr_e2e到服务输出发布，外部控制器另保存最终回封／校验／发布和完整动作E2E。

原生job清理TTL沿用1小时。新控制器须在原结果被清理前保存需要P6观看的首轮代表成片及所有小JSON；纯性能客户端仍不下载视频。

## 构筑与发布验收

Linux先拉完整digest并核对实际imageID、Gateway/HSDP身份及整个父Omni源码，再离线应用精确补丁。最终UID1000强制检查：整包源码／RECORD、实际导入路径、六组RMSNorm数值、继承C14／HSDP CPU测试、依赖／媒体／服务烟测及新合成媒体合同。包METADATA/direct_url、dpkg、父修复代码及legal/build证据保持一致。

CI重新运行最终容器验收后，仅发布同一个已测imageID，并按完整registry digest拉回核对；保存image.json、deployment-image.txt及build/final目录。只有成功发布回执中的repo@digest才是部署身份，artifact ZIP的SHA不是镜像digest。

Runner的`less than 100GiB available after GC`发生在仓库步骤前。本配方不绕过门槛、不prune共享缓存、不修改管理员脚本；若被阻塞，保留运行URL／日志，交管理员恢复容量。

本轮技术合同和机器交接位于 `D:\AI\H3_experiments_H100_HBM3_20261012\delivery\SeedVR2_Optimization_Ablation_20261030` 的00／02／03／04／06。GPU门禁、off/basic扰动、17→33→65容量阶梯、同源两档媒体与Ref7人工质量另行完成。本文的设计和本地测试不等于这些验收已通过。
