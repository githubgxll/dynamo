# 修复范围与验证边界

本补丁是固定 vLLM-Omni 0.31.0rc1 / Torch 2.13.0+cu130 的组件卸载候选实现，尚未通过真实 H3 八卡验收。它修复的目标是 CPU/CUDA 存储不一致，不能提前承诺节省多少显存或提高吞吐。

## 修改范围

- 从已成功发布的 C14 轻量镜像 `87365073...` 派生。
- 只修改安装包中的 `diffusion/offloader/sequential_backend.py`，新增 `hsdp_migration.py`；完整原文件来自成功镜像来源回执，逐文件校验 SHA256。
- 组件选择、H3 pipeline、HSDP 包装、Cache-DiT 配置、SeedVR2/RMSNorm、video-gateway、框架与依赖版本均沿用父镜像。
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

## 测试层次

| 层次 | 能证明的内容 | 不能证明的内容 |
|---|---|---|
| Windows 本地 | 补丁精确性、固定来源、提交/发布脚本、语法与负向回归 | 已安装 Torch 行为、CUDA、H3 生成 |
| Linux 镜像 CPU | 实际模块导入、Torch 方法身份、CPU 对象/别名检查、保留的依赖与六组 RMSNorm 数值回归 | 分片 CPU↔GPU 转换正确性、节省显存 |
| `gpu_roundtrip.py` | 真正八卡 BF16 嵌套 FSDP2、至少三轮完整往返、重复 forward/reshard、所有 rank 参数/存储/数值检查 | 完整 H3、真实 Cache-DiT 状态、视频质量、生产并发 |
| 独立 H3 Run | 同一输入 C02 cold/warm/repeat、A 策略阶段驻留、各 rank 峰值显存/CPU 内存、可解码音视频 | 未测配置与未测负载的效果 |

只有八卡合成测试通过后才执行新 H3 Run；C02 完整通过后才执行 Ref7。无卸载 baseline 也应使用同一候选镜像。旧失败 Run 和 blocked 状态原样保留。

## 必须保留的限制

不支持训练、混合初始设备、任意 Tensor 包装子类、不同对象共享同一底层 storage 的 view 参数、与 FSDP CPUOffloadPolicy 混用。遇到这些形态会明确拒绝，不能忽略检查继续生成。真实 H3 的模型形态和 Cache-DiT 交互仍需现场验证。

`_apply`/FSDP 内部读取受版本约束，因此不能换 Torch 后直接沿用本补丁。对未来版本应重新审核实际源码与参数类型。

本次没有添加少步蒸馏适配器或修改 Ref8 比例；原始 Base 权重仍使用现有只读挂载。普通 SeedVR2 推理源码保留；旧 C14 `sr_only.py` 是绑定旧来源的诊断工具，应继续在原镜像执行，不能把新镜像冒充旧 C14 发布身份。
