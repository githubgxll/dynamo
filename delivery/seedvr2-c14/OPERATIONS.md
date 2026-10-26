# C14 诊断镜像：发布后的 SR-only 操作

配方复核日期：2026-10-26；实验基线：2026-10-23。构筑提交与发布步骤见 [README.md](README.md)，重放命令见 [REPLAY.md](REPLAY.md)。

**先发布新诊断镜像，再在独立、已获分配的 8 卡实验 Pod 内导入原片。** 不修改旧 Ref7 包、旧 Pod 或历史产物，不调用 H3，不新增/转换权重。以下命令调用本配方提供的真实入口；其中目录、Pod 与 namespace 必须换成管理员实际交付的值。

本地验证只能检查脚本和固定源码。Linux 镜像构筑、实际模块与 CPU 数值检查由 CI 执行；SP8、VAE 重放和 C14 画质结论必须在 H100 上另验收。

## 1. 先准备 P0 冻结副本

原始工具和详细说明仍在 [SeedVR2_C14_Runbook_20261023/README.md](D:/AI/H3_experiments_H100_HBM3_20261012/delivery/SeedVR2_C14_Runbook_20261023/README.md)。在 Windows CMD 中先预览：

```bat
set "DOC=D:\AI\H3_experiments_H100_HBM3_20261012\delivery\SeedVR2_C14_Runbook_20261023"
set "ISSUE=D:\AI\SeedVR2_C14_运动伪影问题记录_20261023\assets\evidence.json"
set "DIAG_ROOT=D:\AI\H3_experiments_H100_HBM3_20261012\experiments\h3-seedvr2-c14-motion-diag-20261023"
set "H3_PY=C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
"%H3_PY%" -B "%DOC%\freeze_baseline.py" --evidence "%ISSUE%" --target-root "%DIAG_ROOT%" --preview
```

如果尚未冻结，预览通过后执行一次：

```bat
"%H3_PY%" -B "%DOC%\freeze_baseline.py" --evidence "%ISSUE%" --target-root "%DIAG_ROOT%" --freeze
```

`baseline` 已存在时禁止覆盖，先核对是否为完整的 `LOCAL_FROZEN_VERIFIED` 副本。该工具复制 13 个原始证据文件，并生成 `manifest.json` 和 `COPY-STATE.json`。不要只传 Base 视频而漏掉其余身份和媒体证据。

## 2. Pod 交接条件

由管理员按照既有调度、存储与监控方式准备**独立新实验 Pod**。本包不假设 namespace、节点名、PVC 名或可写宿主路径；不以脚本绕过调度准入。

| 条目 | 要求 |
|---|---|
| 镜像 | 本次 CI `deployment-image.txt` 的完整 `repo@sha256:...`，不得直接填父镜像或 ZIP 的哈希 |
| 身份证据 | 本次成功 CI 的 `image.json`，在容器内可读 |
| GPU | 同一节点独占 8 张完整 H100 80GB；SP8，不能与 H3/其他计算进程争用 |
| 输入挂载 | P0 的完整 `baseline/`，只读 |
| 权重挂载 | 已验证 FP16 模型目录，只读；包含 DiT/VAE safetensors、原 pos_emb.pt、model_index.json、conversion-manifest.json |
| 可写目录 | UID1000 可写的独立实验空间；不需要宿主目录写权限，不从 K8s 下载 Github 内容 |
| 空间 | 导入前至少 51 GiB 可用，抓取限额为 50 GiB；收集 TAR 还需额外预留接近实际证据大小的空间 |
| 入口 | 保留镜像继承的默认配置；实验命令单独从容器 shell 执行，不覆盖线上 Gateway 命令 |

本工具读取 GPU UUID、名称、容量、当前占用和功率上限，不改变功率或节点配置。它不能替代管理员的独占分配：没有竞争进程仍不意味着拥有调度排他权。

使用 Windows CMD 进入已排期的新 Pod（下面三个值由实际环境填写）：

```bat
set "C14_NAMESPACE=实际namespace"
set "C14_POD=本次新实验Pod名称"
set "C14_CONTAINER=实际worker容器名称"
kubectl -n "%C14_NAMESPACE%" exec -it "%C14_POD%" -c "%C14_CONTAINER%" -- /bin/bash
```

后文都是 **Pod 内 Linux shell 命令**。先填写已挂载路径，路径示例只是变量说明，不要求创建同名宿主目录：

```bash
export C14_BASELINE=/实际只读挂载路径/baseline
export C14_MODEL=/实际只读挂载路径/seedvr2-3b-fp16
export C14_RELEASE=/实际证据挂载路径/image.json
export C14_ROOT=/实际可写实验路径/c14-runs
export C14_ID=c14-a0-17lab-1026-a
export C14_RUN="${C14_ROOT}/${C14_ID}"
```

`C14_ROOT` 须预先存在且可写；子目录 `C14_ID` 必须尚不存在。不要将输出指向 Base 或模型目录。重放要求八卡单节点，框架和权重均保持原版本。

## 3. 预检和只读导入：不执行推理

```bash
python3 -B /opt/seedvr2-c14/sr_only.py --help
python3 -B /opt/seedvr2-c14/sr_only.py preflight \
  --baseline "${C14_BASELINE}" --model "${C14_MODEL}" \
  --release-evidence "${C14_RELEASE}"

python3 -B /opt/seedvr2-c14/sr_only.py import-base \
  --baseline "${C14_BASELINE}" --model "${C14_MODEL}" \
  --release-evidence "${C14_RELEASE}" --root "${C14_ROOT}" --id "${C14_ID}"
```

导入器校验完整冻结证据、权重转换 manifest、实际安装源码以及本次发布身份。它只复制原 Base 到新实验目录，重新核对 SHA256、完整解码、362 帧/24 FPS/1280×768 和原音轨包序列；不会请求 H3，也不会重新转换模型。

预期状态为 `IMPORTED_VERIFIED_NO_INFERENCE`。生成独立 `manifest.json`、`capture-profile.json`、`parent-baseline.json`、`release.json`、媒体证明和只读 Base 副本。`capture-profile.json` 固定抓取窗口、模型/原片身份、源码和存储限额；不要运行后编辑 profile。

## 4. A0：完整原链复现与指定窗口抓取

```bash
python3 -B /opt/seedvr2-c14/sr_only.py capture --run "${C14_RUN}"
```

该命令在前台监督 SR 进程组：检查 GPU 空闲与存储预算、只启动 SeedVR2、等待服务就绪，然后**仅提交一次**完整 362 帧原片。原生长视频流程负责 17/4/13 窗口、预取和融合；只对起点130/143/156/169/182保存详细张量，不能先裁6–8秒重新推理。

子进程会保留集群已有代理例外，并为 `localhost/127.0.0.1/::1` 补充代理绕过，确保原生长视频内部的本机窗口请求不会误走外网代理。

固定的实验参数：3B FP16、SP8、8×H100、eager、一步Euler、CFG1、seed7723、原LAB亮度0.8、VAE tiling和高度分片，内部1840×1104，最终1800×1080。不升级框架、改并行、切7B、切1440或打开SR Cache-DiT。

完成后监督程序先确认其 GPU 进程退出，再生成并校验最终视频和原音轨、检查抓取证据完整性。预期状态为 `A0_CAPTURE_COMPLETE_QUALITY_UNDETERMINED`，**表示技术链和证据齐全，不表示 C14 已修好**。

使用另一终端进入同一新 Pod，填好相同 `C14_RUN` 后，可只读观察：

```bash
python3 -B /opt/seedvr2-c14/sr_only.py status --run "${C14_RUN}"
tail -n 80 "${C14_RUN}/attempt/engine.log"
tail -n 5 "${C14_RUN}/attempt/progress.jsonl"
```

网络断开或 HTTP 超时后先查状态和日志，不能重跑 `capture` 或重 POST。失败会保留 `attempt/`；需要新实验时先确定旧组已停止，再使用新 ID。

明确要停止这一组时执行：

```bash
python3 -B /opt/seedvr2-c14/sr_only.py stop-own-run --run "${C14_RUN}"
```

该入口验证 supervisor 的 PID、启动时间、命令行和进程组归属。原监督进程消失、身份不符或 rank 未全部退出时拒绝猜测/误杀；留存现场交由管理员核对，不启动竞争任务、不删除他人 Pod。

## 5. 证据检查、收集和下一步

```bash
python3 -B /opt/seedvr2-c14/sr_only.py verify --run "${C14_RUN}"
```

重点检查：28窗实际映射、5窗完整张量与所有rank证明、实际噪声/latent、校色前后/量化/单窗编码/融合身份。原 `raw1104.mp4` 已经编码，不能代替浮点RGB。先对比历史和A0的全片及n144–192，写清“精确复现／同类但位置改变／未复现／数据不全”。未复现时先排查输入、RNG、调度与抓点影响，不直接下修复结论。

确认运行已结束且空间足够后，收集到一个全新 TAR：

```bash
python3 -B /opt/seedvr2-c14/sr_only.py collect --run "${C14_RUN}" \
  --output "${C14_ROOT}/${C14_ID}-evidence.tar"
```

导出包含显式张量、日志、来源和视频；不复制模型、临时缓存或原生API重复文件。伴随 `.tar.json` 保存各文件及 TAR 哈希，禁止覆盖。TAR 可能达到数十 GiB，请按集群可用下载通道传回；下载后验证 TAR 哈希，不能因传输失败重跑 SR。

接着按 [REPLAY.md](REPLAY.md) 执行 A1/A2/A3：原LAB/原chunk2/原编码链自对照必须先通过，再评估候选。所有输出使用新目录，保持相同父张量；只有完整镜头、回归和人工观看通过，才能称为经过验证的画质修复。关闭抓取后的性能验收另做。

## 6. 失败后的处理边界

| 现象 | 操作 |
|---|---|
| preflight身份、hash或权重manifest不符 | 停止，核对实际挂载和镜像；不修改预期hash让检查放行 |
| GPU已有进程、不是8张完整卡、空间不足 | 等待排期或增加本实验空间；不改dtype/SP，不清理别人缓存 |
| import或capture目录已存在 | 保留旧目录和状态；不会覆盖，也不会自动重推理 |
| 权限不足 | 为新实验输出卷配置UID1000写权限；输入/模型仍只读，不把推理改为root |
| rank失联/OOM/stop未确认 | 保留日志并确认所属GPU组退出；禁止马上开始A1/A2或新A0 |
| 原分支重放不等价 | 结果标为无结论，查后端/数值/布局/缓存；不扩大容差掩盖异常 |
| 颜色或编码候选看起来好些 | 只能记候选或缓解，检查全片、时序和其他主体后才能决定修复 |

本包没有执行这些 GPU 命令，也没有创建或修改 Kubernetes 资源。镜像发布不会自动更换任何业务 Pod。
