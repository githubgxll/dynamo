# 修复范围与验证边界

本配方固定 vLLM-Omni 0.31.0rc1 / Torch 2.13.0+cu130。2026-10-28 的 DiT 入口修复已在 `c053…` 完成用户 Ref7；2026-10-29 在此基础上加入驻留作用域快速路径和分项计时，尚未做新镜像GPU验收。

## 本次修改

- 构筑从相同 C14 父层 `87365073…` 重放累计补丁，不重复对已修复镜像执行 create-once 安装。
- `sequential_backend.py` 增加仅限DiT的驻留lease。入口先完整卸载选中encoder并恢复DiT，成功后才标记ready；50步内部复用ready，跳过重复参数图遍历与显式同步。退出仍通过原生迁移路径释放DiT。
- `H3_OFFLOAD_FASTPATH=0` 保留原逐步检查用于同镜像对照；默认1。跨线程、跨组件、配置变化、作用域内直接迁移或删除hook均拒绝；嵌套同一作用域不重复加载／提前卸载。异常保留首错并清除lease，不把清除lease等同于故障组件已成功搬回CPU。
- `hsdp_migration.py` 增加 `H3_OFFLOAD_TIMING=1` 可选计时。记录CPU墙钟区间、调用数和rank；不添加CUDA同步，不把数值解释为纯GPU kernel时间。默认每个外层作用域一行结构化日志，可选写独立JSON。
- 本次pipeline仍为原DiT-entry输出，SHA256 `9a2b75e9989a4fdb3dcd09094250f82a39d8cc9956c6773481e3c9cf969c0024`。backend/helper当前哈希以同目录 `source-patches.lock.json` 为准。
- 组件选择、实际H2D/D2H、reshard/native `_apply`、参数身份检查、HSDP、Cache-DiT、SeedVR2、Gateway和依赖保持原策略。未增加异步DTensor搬运或修改Torch私有分片存储。

## 快速路径的证明范围

lease是进程级、线程归属受限的状态，使用重入锁保护进入/退出和迁移。它证明的是已审阅框架路径内，整个DiT在同一串行GPU动作中已经恢复；不能允许外部代码绕开hook直接 `.to()` 或同时使用同一模型。pipeline外的普通hook入口仍走完整检查。本轮只支持现有串行请求与H3/SR串行GPU计算。

实际迁移保留阻塞语义。去掉的是已驻留时每步重复执行的检查和同步，不是将一次请求必要的CPU↔GPU搬运消除。若最终剩余耗时主要来自真实拷贝，需下一轮单独评估更细驻留策略或传输优化；不能据本地测试预先声称回收全部约32秒/例损失。

## 为什么要改迁移生命周期

原代码执行 `p.data = p.data.to(device)`。普通 dense Tensor 的行为不能直接套用到 FSDP 管理的 DTensor：除 wrapper 与 local tensor 外，FSDP 还保存分片存储、已展开参数及模块参数引用。错误在下一次读取 `.data` 时出现，不代表 getter 是根因。

候选实现采用 PyTorch 原生生命周期：

1. 确认组件空闲、参数没有梯度、设备和 layout 受支持。同步源设备，HSDP 迁移始终阻塞。
2. **先对所有 FSDP 单元 reshard**。不能一边搬子节点一边才让父节点 reshard，否则父节点可能重新注册旧分片。
3. 对模块图去重，以子节点在前的顺序执行原生 `_apply(..., recurse=False)`。这既处理 Cache-DiT 的循环引用，又保留 H3 模型自己的 `_apply` 清理缓存行为。
4. 由 FSDP 原生 `_apply` 调用 `reset_sharded_param()`，同步它保存的 shard 数据引用。我们的代码不直接替换 DTensor 的 local tensor 或 FSDP 私有存储。
5. 校验参数身份、普通参数子类与附加属性、buffer 别名、设备和 FSDP 引用。原生方法实现与审核版本不同则停止，保存实际源码供复查。
6. 初始化回滚保存组件所有者和原设备，使用同一转换路径恢复；生成过程若迁移失败，保留故障并终止，不自动继续使用部分迁移的 worker。

CPU 上暂存 CUDA mesh 的分片不是 CPU 推理；组件必须恢复至原执行 GPU 后才能 forward。该方案也不是 FSDP 的 `CPUOffloadPolicy`，两者不得混用。

普通参数转换沿用 PyTorch 的兼容 shallow-copy 分支，DTensor 则由 PyTorch 自己走 swap 分支。转换策略只在串行迁移作用域内调整并恢复；同一个 worker 内不能同时从另一线程执行模型转换或推理。

## 为什么还需要修正 DiT 入口

上一版 helper 已要求空闲组件在 forward 前完整恢复；旧合成测试显式先进入 `sequential_offload_component(model)`，所以能够验证迁移。真实 H3 `diffuse()` 原来只进入管理 DLO 的 `_resident_dit_layers_on_device()`，直到 `forward` 内部 Omni hook 才请求恢复。PyTorch 的原生 FSDP pre-forward 更早运行，可能已进入 FORWARD 并将 root 参数 unshard 到 CUDA；子 block/ignored 参数仍在 CPU，helper 此时拒绝混合驻留是必要保护。

新增上下文只在 `_model_cpu_offload_modules` 启用时调用 `self._component_on_device(transformer)`，使用本次任务选择的 transformer，先于原 DLO 上下文进入并包住整个去噪循环。它提前释放选中的 encoder 并恢复整个 DiT；50 步间保持 DiT 驻 GPU，退出时依已有 `offload_after_context` 处理。未启用 module offload 的 baseline/DLO 使用 `nullcontext()`；TE-only 不会因此卸载未被选中的 DiT。保留 mixed/active-FSDP/pending-all-gather 保护，绝不增加常规 `allow_mixed=True`。

新增 CPU 契约使用真实方法 AST 和模拟生命周期，覆盖50步只激活一次、FL/Ref选择、baseline/DLO、TE-only、DiT-only及异常清理。测试用仓库内相对源码，不依赖移交目录的 Windows 路径；临时产物写可写临时目录，不能假设镜像 `/opt` 对 `dynamo` 可写。这些测试不运行真实 FSDP。

## 测试层次

| 层次 | 能证明的内容 | 不能证明的内容 |
|---|---|---|
| Windows 本地 | 补丁精确性、固定来源、提交/发布脚本、语法与负向回归 | 已安装 Torch 行为、CUDA、H3 生成 |
| Linux 镜像 CPU | 实际模块导入、Torch 方法身份、CPU 对象/别名检查、保留的依赖与六组 RMSNorm 数值回归 | 分片 CPU↔GPU 转换正确性、节省显存 |
| `gpu_roundtrip.py` | 真正八卡 BF16 嵌套 FSDP2；保留原 warm 迁移并新增冷态实际安装 `diffuse()` 入口；每轮50步、至少三轮往返；全部 rank 参数/存储/数值和搬运次数检查 | 完整 H3、真实 Cache-DiT 状态、视频质量、生产并发 |
| 独立 H3 Run | 同一输入 C02 cold/warm/repeat、A 策略阶段驻留、各 rank 峰值显存/CPU 内存、可解码音视频 | 未测配置与未测负载的效果 |

只有八卡合成测试通过后才执行新 H3 Run；C02 完整通过后才执行 Ref7。无卸载 baseline 也应使用同一候选镜像。旧失败 Run 和 blocked 状态原样保留。

GPU 测试现在由脚本自身提供 TP8 `VllmConfig`，构造及整个测试期间保持上下文。独立冷态模型不得先跑隐藏 warmup；其组件激活发生在首次 FSDP lazy-init/forward 之前。正向 fixture 不复用负向失败后的 FSDP 状态。实际安装 `diffuse()`/组件上下文执行，但输入打包、去噪控制与 Cache-DiT refresh 是合成替身；即使该测试通过，也必须继续真实 H3+Cache-DiT 的 C02 cold/repeat 验收。

## 必须保留的限制

不支持训练、混合初始设备、任意 Tensor 包装子类、不同对象共享同一底层 storage 的 view 参数、与 FSDP CPUOffloadPolicy 混用。遇到这些形态会明确拒绝，不能忽略检查继续生成。真实 H3 的模型形态和 Cache-DiT 交互仍需现场验证。

`_apply`/FSDP 内部读取受版本约束，因此不能换 Torch 后直接沿用本补丁。对未来版本应重新审核实际源码与参数类型。

本次没有添加少步蒸馏适配器或修改 Ref8 比例；原始 Base 权重仍使用现有只读挂载。普通 SeedVR2 推理源码保留；旧 C14 `sr_only.py` 是绑定旧来源的诊断工具，应继续在原镜像执行，不能把新镜像冒充旧 C14 发布身份。

## 计时边界

`component_body` 包含去噪调用及其内部hook；`hook_pre_forward`包含其迁移与同步；`migration_total`包含图检查、FSDP检查、源同步、reshard、元数据准备、native_apply与后验校验；`native_apply`包含tensor_to/pin_memory。父子区间不能相加，各rank不能相加作为E2E。所有指标是CPU墙钟，原调用的阻塞会计入，但未主动等待CUDA的调用可能在设备完成前返回。

每次成功/失败作用域结束输出一次记录；硬终止或计时写入失败可能没有完整记录，缺失不能填0。日志汇总工具位于 `delivery/h3-hsdp-offload/summarize_offload_timings.py`，对累计日志按scope_id去重，可限定完整的请求时间窗口。
