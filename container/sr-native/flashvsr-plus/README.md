# FlashVSR+ 原生参考镜像

**用于独立质量对照：官方 Base、TC4、BF16、一步、单张 H100。** 不继承现有 Omni/Torch 2.13 镜像，不加载 H3、SeedVR2、Gateway或蒸馏适配器。Dynamo 只提供构建发布流程。

## 已固定的内容

| 项目 | 固定配置 |
|---|---|
| 官方代码 | OpenImagingLab/FlashVSR_Plus `b548a2cc2c929d3d4676f2973abe27876ec4dee7` |
| 模型资产 | ruchangyao/FlashVSR_plus `46b79520946ac67dec94d1566490690953c7debc`；仅 Base DiT、LQ projection、TCDecoder |
| Python / Torch | Python 3.10 / Torch `2.7.1+cu126`（在官方 `>=2.5,<2.8` 范围内） |
| CUDA 基础 | CUDA12.6.3、cuDNN、Ubuntu22.04 devel；完整 digest 由交付层锁定 |
| BSA | MIT Han Lab `49d6c39e4dc0303442cda3bb758b3925d4399c49`；CUTLASS gitlink `a75b4ac483166189a45290783cb0a18af5ff0ea5` |
| 编译方式 | 与镜像内 Torch 一起源码编译；SM90；MAX_JOBS=2；不猜测预编译 wheel |
| 推理入口 | 未修改的 `inference.run_stream.main()`；外层仅绑定只读资产路径、采集原生输出和证据 |
| 首轮参数 | Base、TC4、BF16、一步、BSA、TinyDecoder serial、KV cache never-offload、decode-every auto、校色关闭 |

`requirements.lock` 是针对 Linux x86_64 / Python3.10 / cu126 求解的 **46个包全量版本和下载哈希锁**，`pip --require-hashes --only-binary=:all:` 安装；它不代表已在 Linux 完成安装。安装后还执行严格 `pip check`。BSA 和官方代码源码包单独固定SHA256；源码原文及许可证保留在 `/opt/sources`。

这里只安装 BSA，`--attn-kernel block_sparse` 强制使用它，缺失或执行失败直接终止。不能把 SDPA 作为1920×1152任务的无条件兜底：官方 SDPA 会展开稀疏 mask，超过20亿项直接拒绝；本场景可以触发此限制。Triton、FlashAttention、BSA各自是不同执行路径，不静默切换。

## 权重准备与挂载

大权重不进入镜像，也不在Pod启动时下载。联网机器用共享资产下载脚本按 `assets.lock.json` 下载并核验，挂到容器 `/weights/flashvsr-plus:ro`：

```text
/weights/flashvsr-plus/
  ckpts/base/dit.safetensors          5,676,070,392 bytes
  ckpts/base/lq_proj_in.safetensors   1,151,384,272 bytes
  ckpts/TCDecoder.ckpt                 189,018,333 bytes
```

三文件共7,016,472,997字节（约6.53GiB）。`prompt_tensor/posi_prompt.pth` 随固定官方代码包进入镜像，已按真实内容SHA256核验；不额外下载Wan VAE或Turbo权重。运行前重新核验每份文件大小及完整SHA256。

## 操作入口

本目录 Dockerfile 用仓库根目录为上下文，构筑参数 `BASE_IMAGE` 和 `DYNAMO_COMMIT_SHA` 由交付流程提供。构筑不会访问GPU。

CPU检查（最终镜像必须以UID1000、无网络重跑）：

```bash
python3 /opt/sr-native/flashvsr-plus/audit.py --output-dir /tmp/sr-native-final
```

它验证源码、实际import位置、Python/Torch/CUDA版本、BSA扩展可加载、pip依赖，以及17帧媒体编码/读取/官方PIL预处理。CPU上的小SDPA参考仅检查媒介和计算接口，不证明BSA CUDA通过。

GPU先单独执行核资格检查；Pod只暴露一张H100，输出写新目录：

```bash
python3 /opt/sr-native/flashvsr-plus/run.py --kernel-only --output-dir /results/bsa-qualification
```

完成后运行冻结的17帧全画幅输入。以下 `<...>` 是部署方填入的真实文件及SHA，不是可直接照抄的值：

```bash
python3 /opt/sr-native/flashvsr-plus/run.py \
  --input /inputs/<frozen-window>.mkv --input-sha256 <full-sha256> \
  --weights /weights/flashvsr-plus --output-dir /results/<new-run-id> \
  --width 1920 --height 1152 --frames 17 --fps 24 --seed 1
```

固定输入须由同一H3原片无损抽取w156或w169，不重新编码为有损短片。保存逐帧解码RGB哈希；该哈希可与其他后端的输入证据交叉核对。此入口拒绝画幅改变、非128整除目标、帧数/FPS/PTS不符、时间padding和输出目录复用。不能使用旧1840×1104作Plus目标；跨模型参考画布为1920×1152。

产物包括 `run.json`、`kernel-validation.json`、`native-rgb.pt`、逐帧 `native-png/`、`native.mp4`。浮点RGB在显示clamp之前保存，校色关闭；PNG按官方[-1,1]到[0,255]规则显示。最终1800×1080交付版由共同媒体工具生成，不二次调用模型。官方CLI会尝试保留源音轨，但本工具仍逐帧解码核验最终MP4；音轨和业务音画同步由部署验收另行检查。

## 验收与风险

- 本机只执行纯Python契约/静态检查和依赖解析；Linux安装、BSA编译、H100算子及模型推理尚未执行。
- BSA固定源码支持SM90，但与所选Torch/CUDA组合仍须实际编译及kernel check；无法仅凭import宣称ABI/数值正确。
- 首轮固定TinyDecoder serial和KV cache never-offload，避免根据闲置显存自动改策略。单H100显存和目标尺寸仍须资格测试；失败时保留证据，不悄悄改精度、缩画布或换backend。
- `decode-every auto`读取完整窗口，适合首轮17帧；长片需独立容量验收。65帧可作为后续同上下文对照，不直接扩到整片。
- `elapsed_seconds`包含加载、输入、原生编码、证据写盘等，不可与其他后端的GPU计时直接计算加速比。
- 全量框架与CUDA镜像比补丁衍生层大，首次BSA编译也较慢；缓存命中后仅修改入口不应重编依赖。
- 不在本镜像内接入Gateway、不更新已有实验Pod，不宣称它已经修复C14或画质胜出。

来源：固定[官方README](https://github.com/OpenImagingLab/FlashVSR_Plus/blob/b548a2cc2c929d3d4676f2973abe27876ec4dee7/README.md)、[官方入口](https://github.com/OpenImagingLab/FlashVSR_Plus/blob/b548a2cc2c929d3d4676f2973abe27876ec4dee7/inference/run_stream.py)、[BSA源码](https://github.com/mit-han-lab/Block-Sparse-Attention/tree/49d6c39e4dc0303442cda3bb758b3925d4399c49)。核查日期2026-10-28。
