# SeedVR2 官方全链参考镜像

此镜像用于独立离线质量对照，提供官方 **3B 与标准 7B** 入口。它不继承当前 Omni/Torch 2.13 镜像，不接入 Gateway，也不复制 H3 或蒸馏权重。复用 Dynamo 的提交和发布设施，模型仍从只读目录加载。

当前状态：配方、本地契约测试和 Linux 依赖解析已准备；Linux 源码编译、最终镜像导入、H100 与真实视频尚未在本机执行。构建通过不能代替完整 17 帧推理或人工画质判断。

## 锁定组合

| 内容 | 固定值 |
|---|---|
| 官方代码 | `ByteDance-Seed/SeedVR@e4de8c24441a67e1b7df56abea10645059bb1185` |
| CUDA 构建环境 | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`，由交付层解析并锁定完整 digest |
| Python / Torch / torchvision | 3.10 / `2.3.0+cu121` / `0.18.0+cu121` |
| Diffusers / Transformers | 0.29.1 / 4.38.2 |
| Apex | `4138d31ff0acf4071d1dc001ccb7cd6e00800324`（24.04.01），针对已安装 Torch 编译 |
| FlashAttention | 2.5.9.post1，带 CUTLASS 的 PyPI 源码包，强制源码编译 |
| NumPy / Hugging Face Hub | 1.26.4 / 0.25.2，避免旧扩展 NumPy2 ABI 和旧 Diffusers 的 Hub API 漂移 |
| FFmpeg / PyAV | Ubuntu apt / PyAV 12.0.0；镜像中记录实际 dpkg 清单 |

`requirements.lock` 对 Linux CPython3.10 解析 79 项依赖并记录安装文件哈希。Apex 和 FlashAttention 在锁定环境中以 `--no-deps --no-build-isolation` 编译，不采用官方 README 中注明 Torch2.4 的 Apex 预编译 wheel。源码档案 SHA256 在 `sources.lock.json`，运行时源码清单在 `source-tree.json`。

该组合是基于官方 `requirements.txt` 的候选。官方 `environment.yml` 另列 Torch2.4 / Diffusers0.33.1 等版本，不能把两套文件混装后视为官方已验证组合。本轮选择明确，遇到二进制编译/运行不兼容应修订候选并重验，不静默换 Torch 或换 attention 实现。

## 权重布局

`assets.lock.json` 固定五个原始文件：

```text
/models/seedvr2-native/              # 只读挂载
  seedvr2_ema_3b.pth
  seedvr2_ema_7b.pth                 # 标准7B，不是sharp
  ema_vae.pth                       # 3B/7B文件内容完全相同
  pos_emb.pt
  neg_emb.pt
```

3B DiT 约13.57GB，7B DiT约32.96GB，共享 VAE 约1.00GB，总计约47.53GB（十进制）；这些文件不进入镜像。3B 单独试验只需它自己的 DiT 和三个共享文件。两种模型串行运行，不同进程不能同时驻留占显存。

权重来源分别是官方 HF 固定 revisions：3B `37255ff8cccfb01071b87f635a5948ca8d53117c`、7B `eb0c4281d41ba3767d4f14370f0e37e9e9180c16`。两份模型卡声明 Apache-2.0；Prompt assets 从3B仓复用。加载前完整 SHA256 核验，错误尺寸或哈希直接终止。不在启动时联网下载，不转换为 Omni FP16 safetensors，不改动挂载目录。

## GPU 操作顺序

下列是容器内命令，由新的独立实验 Pod 执行。`/outputs` 必须可写；`/inputs` 和 `/models` 只读；四张 H100 独占此任务，不能与 H3/SR 的其他计算叠加。镜像默认入口是 `run.py`；Pod 使用这些命令时显式设置 `command: ["/bin/bash", "-lc"]`，或在 Docker 使用 `--entrypoint`。

先验证匹配的 fused RMSNorm、FlashAttention 和 NCCL；它不加载模型：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  /opt/sr-native/seedvr2/qualify.py --output-dir /outputs/new-run/kernel-check
```

随后运行完整官方 3B、SP4、17帧诊断轨（现有 Omni 基线尺寸）：

```bash
python /opt/sr-native/seedvr2/run.py \
  --variant 3b --weights /models/seedvr2-native \
  --input /inputs/w156.mp4 --output /outputs/new-run/n3-w156-1840 \
  --input-sha256 "$FROZEN_W156_SHA256" \
  --input-width 1280 --input-height 768 --width 1840 --height 1104 \
  --frames 17 --fps 24 --sp 4 --seed 666
```

每个窗口使用独立新目录；w169同样处理。跨模型轨把目标改为 `--width 1920 --height 1152`，3B再跑一组锚点，7B使用 `--variant 7b`，不能拿旧尺寸结果直接归因模型差异。原生输出后统一交付到1800×1080；如开展1440p比较，需另行冻结所有模型共同计算画布与交付规则，不默认把1080轨结果当作1440验收。

`FROZEN_W156_SHA256` 需由冻结输入清单给出完整64位小写哈希。`--input-sha256` 技术上可省略以首次发现文件身份，此时报告明确为 `RECORDED_ONLY_REQUIRES_COHORT_CHECK`；必须先与原基线输入身份核对，才能接纳为正式对照。各rank还输出HWC RGB uint8逐帧SHA清单，与FlashVSR+使用相同格式，便于发现不同视频解码器带来的输入像素差异。

65帧长上下文复核可明确改 `--frames 65`；输入必须确有65帧。入口拒绝会触发官方补帧的帧数，拒绝SP8，拒绝非16倍数目标和改变画幅比例的请求。首轮建议保留17帧/SP4：3B有20 heads，7B有24 heads，SP4是共同配置；SP8不可直接复用Omni参数。

## 参考链与输出证据

`run.py` 实际调用固定官方 `configure_runner()` 与 `generation_loop()`，保留官方 checkpoint loader、前处理、BF16 autocast、VAE BF16、一步/CFG1、条件噪声0和阶段式CPU卸载。DiT实际权重dtype由官方load保留，每rank记录dtype计数，不声称全部为BF16。

容易误读的一点：`NaResize(mode="area")` 指按目标**面积**计算尺寸；固定源码中的空间插值实际默认 **BICUBIC**，不是 PyTorch 的 `area` 插值。尺寸经16整除裁切；入口在VAE编码前核对真实张量shape，任何意外裁切/补帧使对照失败。

运行时包装器只复制输出证据并返回原对象，不改模型源文件。显式禁用可选 wavelet 校色，保存校色和clamp前的完整RGB。这是诊断输出选择，不能混称开启官方wavelet后处理的demo输出。保存：

- `request.json`：输入文件SHA、原权重身份、配置与镜像源码标记。
- `rank-*-loaded.json` / `rank-*-result.json`：每rank加载精度、冻结输入像素SHA、完整链状态、显存峰值。
- `boundaries/`：完整前处理输入、缩放后的condition latent、DiT velocity、恢复latent、实际decoder输入、原生raw RGB；每次调用有独立序号，不复用Omni中间张量。
- `native-png/`：按固定 `clip((x+1)/2,0,1)` 显示的逐帧PNG；`native-video/input.mp4` 为无音轨原生预览，另做帧数、尺寸、FPS检查。
- `result.json`：所有rank通过后才为 `PASS_NATIVE_FULL_CHAIN`，质量验收仍为 `NOT_RUN`。

计时 `elapsed_seconds_rank_including_capture` 包含CPU卸载、同步、证据落盘与PNG导出，**不是纯GPU耗时**，也不能与Omni常驻吞吐直接比较。同seed不保证官方与Omni生成同一随机场。

## 仍需实际验证的风险

1. Apex／FlashAttention源码与当前编译器的真实兼容性，需要 Linux Runner 编译和 H100 数值资格。首次编译慢、RAM消耗较大；默认 `MAX_JOBS=4`，保持Runner管理员的磁盘准入规则。
2. 首次官方加载可能有较大的显存/CPU峰值；共享SP4不代表每卡权重或VAE显存除以4。7B不能由3B成功推定容量通过。
3. 本机没有Torch/CUDA，当前只能跑契约测试和依赖解析；实际镜像非root离线导入/media检查由构建与发布门禁执行。
4. 生成成功仍需检查C14伪影与更长片段；音轨/PTS回封、统一交付尺寸和人工盲看由实验包负责，demo预览不能代替业务成片。
5. 这套候选是官方原生离线参考，不提供H3、Gateway服务或跨模型共享进程；接入服务需要单独的接口与生命周期工作。

官方依据：[固定SeedVR源码](https://github.com/ByteDance-Seed/SeedVR/tree/e4de8c24441a67e1b7df56abea10645059bb1185)、[3B模型](https://huggingface.co/ByteDance-Seed/SeedVR2-3B/tree/37255ff8cccfb01071b87f635a5948ca8d53117c)、[标准7B模型](https://huggingface.co/ByteDance-Seed/SeedVR2-7B/tree/eb0c4281d41ba3767d4f14370f0e37e9e9180c16)。
