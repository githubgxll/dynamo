# SeedVR2 C14：固定张量重放操作页

**本镜像补充的是诊断能力，尚未证明 C14 伪影已经修复。** 保留 Omni 0.31.0rc1、已有 RMSNorm FP32 修复、Gateway 和原依赖。A1/A2/A3 均先重现 A0 的原路径，失败就停止候选；不重跑 H3 或 DiT，不修改旧实验。

本页命令在新诊断镜像的 Linux 容器内执行；路径 `/work/c14` 代表独立可写证据卷。模型目录只读。Windows CMD 的镜像提交命令参见交付目录 README。**不要把本页命令传给旧 Ref7 CLI。**

## 1. 前提与共用路径

先完成新包 `sr_only.py import-base`、`capture` 和 `verify`；A0 必须是完整 362 帧、24 FPS、1280×768 → 1840×1104 → 1800×1080。确认 A0 同类症状是否复现，保留相应人工记录。

```bash
set -euo pipefail
export C14_RUN=/work/c14/c14-a0-17lab-1026-a
export C14_CAPTURE="$C14_RUN/capture/$(basename "$C14_RUN")"
export C14_REPLAY=/work/c14-replays
mkdir -p "$C14_REPLAY"
unset SEEDVR2_C14_PROFILE
python /opt/seedvr2-c14/sr_only.py verify --run "$C14_RUN"
```

`C14_RUN` 必须改为本次实际成功的 A0 目录。旧的 `ref7-rest-1022-a-c14` 没有新增张量，不能直接重放。GPU 命令的 `--id`、CPU 命令的 `--output` 均生成**新目录**，同名拒绝覆盖；失败后保留目录与 `FAILED*.json`，另用新 ID。

每次启动先检查输出卷剩余空间：A1 至少 8 GiB，A2 至少 4 GiB，A3a 至少 4 GiB，A3b 至少 10 GiB。这是准入保留量，不是峰值实测；各实验目录会累计占用空间，存储不足不得继续。

A0 supervisor 会终止自己的 SR 引擎并检查 GPU 空闲。重放前仍应确认 GPU 排期、进程清单；同一组卡不能同时跑 H3、A1、A2 或另一份 SR。下列 A2 只支持单节点 8 张 GPU，所有 rank 均执行。

## 2. A1：同 RGB，完整帧先重现 LAB

```bash
python /opt/seedvr2-c14/gpu_guard.py run \
  --run "$C14_RUN" --phase color --window 156 --id a1-color-w156-a
python /opt/seedvr2-c14/gpu_guard.py run \
  --run "$C14_RUN" --phase color --window 169 --id a1-color-w169-a
```

读取完整 17 帧的实际 `color_input/reference/output`，逐帧恢复 dtype、原 stride、CUDA autocast 和 TF32 设置。先全部重放原 LAB（亮度权重 0.8）；全部基线对上后才计算 none / wavelet / LAB。**先全帧校色，再裁羊头 ROI**，不会先裁局部改变颜色直方图。

每个分支包含浮点 safetensors、同一量化公式的 PNG、全帧/ROI 的 24 FPS 与 12 FPS FFV1 观看副本，及全帧/ROI 数值差异。FFV1 保存后逐像素回读校验；没有自动对比度拉伸。观看副本无音频，原始 float 张量仍保留。只有 `COMPLETE.json` 表示重放流程完成，不表示候选画质通过。

默认原路自对照是逐值相同。若 GPU 后端存在非确定性，先以重复原参数重放测量差异并形成有理由的容差策略；策略 SHA 必须在 A0 profile 中预先锁定。不能看到候选结果以后临时提高容差让它通过。Guard CLI 的 `--tolerance-policy` 只接受已锁定的文件。

## 3. A2：同 latent，SP8 完整 decode2 / decode5

仅在 A1 不能充分解释问题时运行。使用 **A0 同一份 `ema_vae_fp16.safetensors`**，不转换或重新下载。

```bash
python /opt/seedvr2-c14/gpu_guard.py run \
  --run "$C14_RUN" --phase decode --window 156 --id a2-decode-w156-a
```

窗口 169 改用 `--window 169 --id a2-decode-w169-a` 另跑一次。Guard 从冻结 manifest 读取原 FP16 模型目录，不接受临时替换权重。

A1/A2 共用 A0 的 `.c14-gpu.lock`，先检查 8 卡空闲，再启动自己监督的进程组，结束后回收整组并再次检查 GPU 空闲。每个 rank 在使用 CUDA 前另存 PID/启动指纹/进程组，覆盖 torchrun 为各 rank 创建独立 session 的情况。所有输出位于 `C14_RUN` 同级的 `replays/<A0目录名>-<本次id>/`，其中 `result/` 是张量及比较结果，`worker.log`/`rank-logs/` 是日志。入口拒绝绕过 guard 直接运行 GPU 脚本；共享锁不能替代集群排期。

查看状态或仅停止这次重放：

```bash
python /opt/seedvr2-c14/gpu_guard.py status \
  --replay /work/c14/replays/c14-a0-17lab-1026-a-a2-decode-w156-a
python /opt/seedvr2-c14/gpu_guard.py stop \
  --replay /work/c14/replays/c14-a0-17lab-1026-a-a2-decode-w156-a
```

停止命令核验 supervisor 的 PID、启动指纹、命令归属，再让它回收自己的 rank 组，不删除 Pod、不按进程名批量杀进程。`STOP_UNCONFIRMED_DO_NOT_START_NEXT` 表示尚未证明整组退出，不能开始下一次。

实际实现：

1. 建立 Omni 原生 SP8/Ulysses8 进程组，检查 rank 顺序、原 tiling 与高度分片。
2. 只构造原生 VAE；从 meta 模块严格装入完整 FP16 state dict，不构造 DiT/text，不重新抽噪声。权重 SHA 与 A0 相同。
3. 读取真正进入 decode 的 `[1,16,5,138,230]` 张量，保留原始 dtype/stride；不重复除以 0.9152。
4. 所有 rank 依次完整调用 `decode(latent, chunk_size=2)`，各自与捕获的 clamp 前 `[1,3,17,1104,1840]` RGB 比较。
5. 每个 rank 的原路自对照通过后才调用 chunk5。每次完整调用由原 VAE 新建 `TemporalContext`；**内部 chunk 之间不清缓存**。
6. rank0 保存完整 RGB，所有 rank 保存耗时/峰值显存/比较结果。默认执行配置未改成 chunk5；这只是单独的重放候选。

本地文件读写错误通过独立 Gloo 同步传给所有 rank。CUDA 模型内若某 rank 失败，直接让 torchrun 监督器终止全组；不再尝试新的 CUDA collective。WORLD/control 配置 300 秒超时，guard 外层还有 2 小时整体期限。OOM、超时或进程中断均无候选结论，检查所有本次 rank 确实退出和 GPU 空闲后才开始下一次；不得自动降低 SP 或更换 dtype。单卡 CPU 测试不能证明这套 SP8 实测已经通过。

## 4. A3a：同像素重放单窗编码

该步骤无 GPU 推理，可以在模型停机后运行：

```bash
python /opt/seedvr2-c14/replay_encoding.py \
  --window "$C14_CAPTURE/windows/000156" \
  --output "$C14_REPLAY/a3-window-w156-a"
python /opt/seedvr2-c14/replay_encoding.py \
  --window "$C14_CAPTURE/windows/000169" \
  --output "$C14_REPLAY/a3-window-w169-a"
```

用 A0 的量化后完整像素重放原 `_encode_video_bytes`，核对实际编码分支、worker 数、PyAV 版本、合并后的 codec options、颜色标记、回解码像素及时间戳。原路通过后生成同像素 RGB/FFV1 无损中间件并逐像素回读。

这里不会把 `qp=0` 自动解释成 RGB 全链无损。当前源代码的默认 `crf` 与传入 `qp` 如何合并，均记录在 `encoding.json.muxer`，不会“顺便修正”它们。

## 5. A3b：融合 → 完整 raw → final 的同像素比较

A0 额外保存全部 362 帧的融合后 uint8 像素，约 2.06 GiB 原始像素量；逐帧存取，不一次加载全片。只对选定 5 个模型窗口保存大浮点张量。先完整 A0 验证后运行：

```bash
python /opt/seedvr2-c14/replay_chain.py \
  --run "$C14_RUN" \
  --output "$C14_REPLAY/a3-full-chain-a"
```

顺序：

- 从 5 个真实单窗输出复现原 `rint` overlap 融合，并与捕获像素比较。完整可验证范围为全局帧 **134–194**；两侧缺少相邻窗口的边界不冒充已验证。
- 读取全片 362 帧融合像素，按捕获的真实 PyAV 参数重现 raw 视频；逐帧对上 A0 raw 以后，再用原 final 命令重现 1800×1080。
- raw 和 final 原路均对上后，才用相同融合像素生成 RGB/FFV1 的 NUT 中间件；逐像素验证无损且保持精确 24 FPS 时间戳，再走**同一条 final 缩放/编码/原音轨复制命令**。这里不用默认毫秒时间基的 Matroska，避免给 passthrough 时间轴引入新变量。
- 输出完整原链成品与无损中间路径成品、逐帧/ROI 差异和音轨包 hash/时间证明。先看 n144–192 连续视频，再看全部 362 帧。

此比较改变的是 raw 中间视频编码；单窗编码/融合输入仍来自原 A0，不混入不同 SR 随机结果。raw 中间音频重编码不会重复执行（raw 只比较视频）；最终成品始终从原 Base 复制音轨并验包/时间。不要将“候选与原链不同”自动写成“候选更好”。

## 6. 结果、失败与边界

- 先检查 `baseline.json` 或 `*-baseline.json`，再看 `COMPLETE.json`。没有完成标记，或存在失败标记时，不给候选质量结论。
- `sr_only.py collect` 只收集 A0。Guard 的证据在同级 `replays/`，A3 的证据在指定 `C14_REPLAY`；另行归档这两个目录，并保留全部日志和张量元数据。不要只拷贝观看 MP4。
- A1/A2/A3 都是独立新输出，不会修改 A0、旧 manifest、原权重或常规服务默认值。
- 不升级 Torch/CUDA/Omni/Gateway，不新安装 Python 依赖；同一原生组件在最终镜像中导入验证。
- 构建 CI 必须运行 `test_replay_torch.py` 的真实 Torch/Omni/PyAV CPU fixtures，要求零跳过；它们验证张量往返、原生小 VAE 缓存周期和 FFV1 回读，**不能替代真实 FP16 权重、8×H100、完整镜头的验收**。
- 只有完整镜头及其他历史素材回归通过，且关闭 dump 后确认显存/速度/共置安全，才另行决定是否把 wavelet、decode5 或编码策略变成默认生产修复。当前镜像没有预先采用这些候选。
