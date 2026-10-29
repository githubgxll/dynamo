# Ref2VA 八卡卸载耗时优化与细分计时交接（2026-10-29）

本轮已修改镜像运行代码、真实八卡门禁和本地计时汇总工具。**需要重新构筑镜像；当前尚未构筑、发布或测得加速结果。** 当前成功基线 `c053e525…`、既有RUN与结果不修改。新实验必须使用新镜像身份和新RUN。

## 1. 本次针对哪里

历史无卸载与当前卸载A的完整Ref7中，七例 H3 HTTP 累计 `624.622 → 857.865秒`，增加 `233.242秒`；其中 `diffuse` 增加 `223.728秒`，约占95.92%。`stage_0`、H3 HTTP和H3服务动作是逐层包含的区间，不能把三个增量相加。不同历史镜像之间仍有配置混杂，这些数值用于定位，不能作为本补丁的纯因果收益。

外层DiT入口已经每次请求加载一次并在50步间保持驻留；但内部forward hook仍反复遍历参数／模块图、执行无实际搬运的检查和显式同步。本轮优先去掉这一部分重复操作。**不承诺能消除全部约32秒/例的差额**：一次请求必要的H2D/D2H、reshard和安全校验仍保留，等待分项计时判断剩余成本。

历史对比文件：

`D:\AI\H3_experiments_H100_HBM3_20261012\reports\Ref7_Offload_H3_SR_20261029\Ref7_Offload前后_完整流程用时对比_不含下载_20261029.md`

## 2. 代码改动与不变配置

| 项目 | 本轮行为 |
|---|---|
| DiT作用域入口 | 完整检查、释放选中encoder、恢复DiT，完成后才允许快速路径 |
| 50步内forward | 已处于同一DiT驻留作用域时，跳过重复图遍历、无效搬运检查和显式同步 |
| 作用域退出 | 保留原有DiT卸载、FSDP reshard、原生 `_apply` 与参数身份验证 |
| 防误用 | 跨组件／跨线程、配置变化、通过本后端在作用域内直接驱逐与移除hook均拒绝；异常离开最外层DiT作用域时清除驻留证明并保留首错 |
| 对照开关 | `H3_OFFLOAD_FASTPATH=0` 为保留逐步检查的同镜像对照；`1` 为优化路径，默认1 |
| 细分计时 | `H3_OFFLOAD_TIMING=1` 启用；默认0，避免日常运行无关记录 |
| 输出位置 | 默认每个外层作用域一行 `H3_OFFLOAD_TIMING {JSON}`；可选 `H3_OFFLOAD_TIMING_DIR` 每rank写独立文件 |

保持8×H100 80GB、Ref2VA Base50/BF16、U8/HSDP8/TE TP8、VAE原参数、Cache-DiT原参数、offload A的DiT＋text_encoder选择不变；H3/SR计算继续串行。保留现有SR FP16、窗口17/overlap4、SP8与两档等比输出。

本轮没有改DTensor内部存储，也没有把原先阻塞迁移改为异步。日志中的 `dtensor_pin_skipped` 说明DTensor锁页路径仍未实现，不能将配置中的 `pin_memory=true` 理解为所有分片已经锁页。

同线程、同模块的内层上下文复用外层驻留证明；内层普通异常若被调用者捕获、没有离开最外层作用域，外层证明仍可继续使用。作用域期间不得由第三方代码绕过本后端直接调用 `module.to()`、改写参数或使用私有迁移入口；当前保护不声称拦截任意外部写操作。

## 3. 计时如何看

| 指标 | 含义／包含关系 |
|---|---|
| `component_body` | 已激活组件后的模型调用区间；DiT内含50步及其内部hook |
| `hook_pre_forward` | 完整慢路径hook：目标卸载、当前组件恢复、同步 |
| `hook_synchronize` | hook原本已有的同步等待 |
| `migration_total` | 一次组件迁移调用总区间；也包含no-op检查 |
| `migration_to_cuda` / `migration_to_cpu` | 按方向的迁移调用区间；不能直接称为纯传输时间 |
| `graph_preflight` / `fsdp_preflight` | 模块图／参数与FSDP状态检查 |
| `source_sync` / `reshard` | 迁移前源同步、原生FSDP重新分片 |
| `metadata_prepare` | 参数身份、别名和属性等快照准备 |
| `native_apply` | 原生转换；包含 `tensor_to` 和 `pin_memory` |
| `post_validate` | 迁移后设备、对象与FSDP引用验证 |
| `allocator_empty_cache` | 原有卸载后allocator清理 |
| `debug_memory_query` | 仅DEBUG日志启用时查询空闲显存，INFO模式不再计算 |
| `hook_fastpath` / `hook_slowpath` | 快／慢路径次数；结合实际GPU门禁解释 |

均为**CPU墙钟区间**，没有为计时额外同步CUDA，不等于纯GPU kernel耗时。父子指标不能求和，各rank时间也不能相加当E2E。应逐rank比较，关注最慢rank及差异；rank=-1表示未知，不冒充rank0。记录失败、硬终止或写入失败时缺失指标，不能填0。

`h2d_moves`／`d2h_moves` 是该外层作用域中所有实际组件迁移的累计次数。进入DiT作用域时也可能卸载encoder，因此不能把累计D2H次数直接解释为DiT卸载次数。每轮DiT一次H2D／D2H应以GPU门禁中按 `component` 标注的 `component_moves` 序列验证。

同一份累计h3日志可能进入多个case ZIP。汇总工具按scope_id去重；分析单个case时，按成功动作证据中的开始/结束时刻设置过滤，或使用隔离日志。工具不会仅凭ZIP所属case推断日志里的所有事件都属于该case。

## 4. 构筑与身份

沿用现有 `Dingorouter-H3-vllm-omni-jirx` 分支和Harbor发布链路。本次只从固定成功父层离线重放补丁，不重装框架、不运行原生SR矩阵或完整builder；通用新仓库方案暂缓。

**提交前先确认Runner容量准入已恢复。** `/etc/dingo-gc/manager.py` 的100GiB剩余空间检查发生在仓库步骤之前，即使这次是轻量派生也无法绕过。尚被该检查拦截时，先由有主机权限的维护者恢复空间／调整机器安排，再执行下面的提交命令；不改父镜像或依赖来掩盖主机容量问题。

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\h3-hsdp-offload\01-check.cmd
call delivery\h3-hsdp-offload\02-submit-build.cmd
```

第一条仅本地检查；第二条会提交、推送并触发Linux构筑／Harbor发布，由镜像操作者执行。本轮没有执行第二条。

仍从固定父层构筑：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:87365073cbdf95fa17ca352a24a93cb74fde144a47c96135e91011d1a03df72e
```

不要直接把FROM替换为已打补丁的 `c053…`。vLLM 0.31.0、Omni 0.31.0rc1、Torch 2.13.0+cu130、Cache-DiT 1.5.0不变。来源／输出哈希在 `delivery/h3-hsdp-offload/source-lock.json`、`container/h3-hsdp-offload/source-patches.lock.json`；封存清单在 `input-seal.json`。新digest由实际构筑返回，当前没有可填的新digest。

提交器也修正了历史变更的封存边界：配方累计基线仍为 `0b11a41…`；本轮工作树基线锁到已有 `f78ca9ee…`。该提交中已存在的其他SR配方保持原样，不混入本配方导出；本轮任何无关新增、修改、删除或回退仍拒绝，共享输入仍校验旧配方基线。这个修正不触发构筑，也不修改现有Git提交。

## 5. 新实验包必须接入的操作代码

现有 `resident.environment()` 主动清除所有 `H3_*`，因此在Windows CMD执行 `set H3_OFFLOAD_TIMING=1`，或仅在Pod添加env，**均不能保证模型收到开关**。

随本次交付提供 `integrate_experiment.py`，对指定旧包生成两文件的受检补丁：

- `workflow.py` 的prepare新增 `--offload-fastpath 0/1`、`--offload-timing 0/1`；显式选择写入冻结配置。未指定时不补值，兼容既有配置。
- `resident.py` 在清除环境后，从冻结配置恢复这两个开关，且只给H3子进程；SR环境不带这两个开关。

已对当前实验包生成并验证：[集成补丁](experiment-integration-local/experiment.patch)、[本地集成结果](experiment-integration-local/integration.json)。补丁随附7项真实实验代码的离线控制契约测试。它是待接入的新包补丁，不是可直接启动的部署目录。补丁为零上下文unified diff；在独立新副本核对 `integration.json` 中源文件SHA后，以 `git apply --unidiff-zero` 应用。

需要基于更新后的操作包重新生成时，输出目录必须不存在且位于旧包之外：

```bat
python -B -X utf8 delivery\h3-hsdp-offload\integrate_experiment.py --source-package "D:\AI\H3_experiments_H100_HBM3_20261012\experiments\ref2va8-offload-dit-entry-20261028" --output-package "D:\AI\_workspace\offload-integration-review-20261029"
```

工具不修改旧包、已有RUN或ConfigMap，不能代替发布后的镜像身份绑定。该步骤完成后还必须：

1. 用新digest、真实 `H3_HSDP_SOURCE_COMMIT`、新backend/helper哈希更新**独立新包**的 `image_binding.py` 与 `image-repair-lock.json`。
2. 更新recipe-snapshot、来源封存、payload与包哈希；保持父镜像标记原值，不伪造环境变量绕过校验。
3. 同步新GPU门禁及**新实验包的 `repair_gate.py` 验证器**，18阶段／每轮50步／至少三轮保持不变；按冻结配置显式传递门禁模式，验证每rank的 `scope_fastpath`、逐步扫描／同步计数和组件迁移序列。
4. 检查H3进程日志中确实有计时JSON、8个rank与所选fastpath值。没有记录时视为未采集，不能当耗时为0。

**两文件集成补丁不构成完整的启动门禁升级。** 当前旧包 `resident.py` 仍调用 `run_roundtrip(work_root)`，旧 `repair_gate.py` 未接收冻结的fastpath/timing配置，且按旧版逐步记录字典精确比较；新门禁增加逐步计数后，仅替换脚本哈希会被旧验证器拒绝。新包需接入新报告schema、明确验证目标模式，保留两种模式独立的验收目录和回执；不能通过删除字段检查放行。仅显式给H3子进程注入开关，也不能据此声称启动门禁已经验证同一种模式。

默认使用日志输出，不设置TIMING_DIR。计时使用框架的 `init_logger`，日志级别应允许INFO（例如 `VLLM_LOGGING_LEVEL=INFO`）；不要用只显示WARNING的配置采集该轮计时。现有模型日志由resident重定向、经现有导出流程带入结果包，适合继续用分段下载。若选独立TIMING_DIR，必须额外加入导出和完整性清单；现有导出器不会自动收集任意目录。

以下prepare命令仅用于**上述身份与代码已同步的新包**，不要在旧c053包直接执行：

```bat
set "RUN=ref8-fp0-1029-a"
call offload.cmd prepare --id %RUN% --offload a --sr-capable --offload-fastpath 0 --offload-timing 1
```

另一组使用独立RUN与 `--offload-fastpath 1`，例如 `ref8-fp1-1029-a`。两组都是八卡，在0050顺序运行，先留存上一组结果再释放资源。不要通过改已冻结config或原地改env来切换同一RUN。

后续采用原分步命令 `start --preview` → `admission-check` → `start` → `wait` → `h3-start`；原 `01-start.cmd`／`05-start-sr.cmd` 会自行prepare，尚未透传新增开关，不能用它们覆盖已prepare的新RUN。新包封装入口也应同步这些参数。

## 6. 验收顺序与判定

在没有加载H3/SR的独立新八卡Pod中，先执行两种模式的实际GPU门禁，每次用独立目录。以下为Pod内Linux命令：

```bash
env H3_OFFLOAD_FASTPATH=0 H3_OFFLOAD_TIMING=1 python3 -m torch.distributed.run --standalone --nproc-per-node=8 /opt/h3-hsdp-offload/gpu_roundtrip.py --output-dir /tmp/h3-gate-fp0 --cycles 3
env H3_OFFLOAD_FASTPATH=1 H3_OFFLOAD_TIMING=1 python3 -m torch.distributed.run --standalone --nproc-per-node=8 /opt/h3-hsdp-offload/gpu_roundtrip.py --output-dir /tmp/h3-gate-fp1 --cycles 3
```

这两条顺序执行。保留summary、所有rank JSON和日志；开启时forward内扫描／backend同步必须为0，关闭时必须实际发生。仍需每轮DiT只有一次H2D和D2H、参数身份／存储／数值一致。门禁不是完整模型性能测试。

然后按以下顺序，始终固定输入、seed、50步和输出规格：

1. H3-only C02 cold、warm、连续重复；比较stage_0、HTTP、服务动作和新增分项，cold与warm分开。
2. 同一新镜像offload A、fastpath=0/1分别跑Ref7，计时都开；保留每rank数据和视频hash。哈希差异要进一步检查，不能自动判成通过。
3. 在两组中都实际加载SR常驻，再跑完整H3→SR1080→SR1440。对比H3 HTTP和两档SR处理total；剔除下载／校验，避免与内含HTTP重复叠加。
4. 记录初始化、条件编码、去噪峰值及CPU内存；确认收益没有以增加OOM风险、输出变化或状态泄漏为代价。
5. 若fastpath开启后仍慢，按 `native_apply/tensor_to`、`graph_preflight/fsdp_preflight`、`source_sync/reshard` 归因。不要先叠加其他卸载／精度优化。普通Ref7通过也不能代替同事九图三视频极端请求或Gateway压力验收。

成功后再关闭TIMING复测代表case，可估计计时开销。首次对照保持两边TIMING一致，且不能直接用新timing-on结果对比旧timing-off来宣称纯收益。

## 7. 本地汇总命令

将RESULTS设为新运行已完整下载的results目录或某个ZIP；OUT设在该目录之外，避免后续递归读到自己的报告。

```bat
cd /d D:\AI\dynamo-h3-experimental031
set "RESULTS=D:\AI\新实验包\runs\新RUN\results"
set "OUT=D:\AI\新实验包\runs\新RUN\offload-timing-summary"
call delivery\h3-hsdp-offload\summarize-offload.cmd --input "%RESULTS%" --output-dir "%OUT%"
```

限定一个动作的时间窗口（替换为该动作真实证据时刻）：

```bat
call delivery\h3-hsdp-offload\summarize-offload.cmd --input "%RESULTS%" --output-dir "%OUT%\case" --started-after "2026-10-29T10:00:00+08:00" --finished-before "2026-10-29T10:05:00+08:00"
```

可以重复传 `--input`，仅合并已核对属于同一镜像／RUN／冻结配置的证据。工具仅读JSON／日志／ZIP内文字，不提取视频或大张量。输出 `OFFLOAD-TIMINGS.md`、`OFFLOAD-TIMINGS.json`；控制台只报路径与少量统计，不刷SHA。没有任何可选完整scope、冲突重复或损坏计时记录会明确报错，不产出伪零值。

`COMPLETE_LOCAL_SUMMARY` 仅表示选中的scope记录通过格式检查并完成汇总，**不代表8个rank均已采集，也不自动验证镜像／RUN身份一致**；只有rank0一条有效记录也可能汇总成功。分析前必须核对rank0–7齐全、所选fastpath值一致，并单独标记缺失rank／scope。工具没有自动推断case归属或保证采集完整性，不能把汇总状态代替八卡验收。

## 8. 本地检查与剩余边界

本地检查覆盖精确补丁、快路径控制流、嵌套与异常、关闭开关对照、计时输出／失败、GPU门禁契约、统计去重与来源、镜像构筑／提交脚本。镜像CPU门禁另增加实际安装HookRegistry的50次forward与计时JSON测试，需Linux镜像阶段执行。

最新检查结果保存在仓库 `.h3-experimental-evidence/hsdp/local/checks-*/summary.json`。Windows本机未安装实际Torch/Omni，未执行安装后CPU、CUDA多卡或H3生成。因此本交付是已验证的本地候选实现，不是已上线或已测得的加速版本。

镜像构筑侧复核（2026-10-29）：封存与本地检查通过；404项回归中388项实际执行通过、16项运行时测试因本机缺少实际Torch环境跳过，28份Python语法及4段workflow shell检查通过。该批证据为 `.h3-experimental-evidence/hsdp/local/checks-20261029T035837.110171Z/summary.json`。本次复核仅补充交接边界，没有额外修改运行代码或旧实验包；文档更新后需同步配方封存，不将这些结果写成新镜像或GPU性能通过。
