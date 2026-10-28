# 修复范围与验证边界

本配方保留固定 vLLM-Omni 0.31.0rc1 / Torch 2.13.0+cu130 的组件迁移补丁，并于 2026-10-28 增加 H3 DiT 前向入口修复。上一版八卡合成迁移通过，真实 C02 却在首次 DiT 前向报混合驻留；新的入口补丁尚待镜像、八卡与媒体分层验收，不能提前承诺显存收益或吞吐。

## 修改范围

- 复用上一版 `4a5d1340…` / `18e218…` 的内容与同一构建配方，从相同 C14 固定父镜像 `87365073…` 重放原补丁，再加入 pipeline 改动。不会在已打补丁的 `18e218…` 上重复执行旧 create-once 安装。
- 安装包仍修改 `diffusion/offloader/sequential_backend.py` 并新增 `hsdp_migration.py`；两文件与上一版分别保持 SHA256 `b7f65e771d755f6f897ac4c8fd6753153ba0d114b5bc03578b3259ff4e987ed0`、`1bec61bb616e797aee598f49a9ba0b4fe400d6cc41e2cf069bf599723ef587af`。
- 新增 `models/minimax_h3/pipeline_minimax_h3.py::diffuse()` 外层组件上下文，最终文件 SHA256 精确为用户候选的 `9a2b75e9989a4fdb3dcd09094250f82a39d8cc9956c6773481e3c9cf969c0024`。pipeline 从 unchanged 移至 changed_sources，同步安装来源、Omni RECORD、CI 与导入回执。
- 组件选择、HSDP 包装、Cache-DiT 配置、SeedVR2/RMSNorm、video-gateway、框架与依赖版本保留；除 `diffuse()` 之外的 H3 pipeline AST 不变。
- 原始源文件、Torch FSDP 参数管理源码、原 RECORD、改后 RECORD 摘要与所有验证结果随镜像/CI artifact 保存。旧 C14 验证器保持原样，新源码使用独立完整来源验证器。

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
