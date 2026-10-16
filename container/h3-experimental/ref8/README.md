# 0.31.0rc1 Ref8 12:3 定点补丁

**只改变 `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` 对应的 video shift：6→12。音频保持 3，FL8 保持 6:3。** 不改变 LoRA alpha/scale、步数、模型／权重字节和默认分辨率处理。这里的 12:3 是两个采样时间偏移参数。

这是独立的 0.31 补丁：固定官方 Omni `0.31.0rc1`、源码 `6dd0d1f9310f7598b773c796b434c8000c2816ec` 和六份相关源码，不要求任何 0.30 镜像 digest 或旧 53 文件检查器。上游权重作者依据参见 [讨论 51](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/51)。

本文件夹由主交付 Dockerfile 复制为 `/opt/h3-ref8-patch`。源码 fixtures 用于检查，不会复制覆盖整个已安装的 Omni 包。

## 本机只读检查（Windows CMD）

```cmd
cd /d D:\AI\H3_Production_Image_Handoff_20261015
python -B build\ref8-omni031\check_contract.py --local
python -B -m unittest discover -s build\ref8-omni031\tests -v
```

仅使用 Python 标准库，不安装 Torch 或任何 GPU 软件；结果打印到终端，不修改交付文件。需要留存新报告时，可使用 `--report` 指向交付目录之外的工作目录。`PASS_LOCAL_CONTRACT_ONLY` 只表示版本／源码锁、实际解析与验证函数的 AST 执行、请求到采样函数的静态路径通过，不证明 Linux 安装或 GPU 成功。

## Linux 镜像构建中应用（由主 Dockerfile 执行）

安装全部生产运行时与监测依赖后，root 阶段执行：

```bash
python3 -B /opt/h3-ref8-patch/apply_patch.py --apply-installed \
  --report /opt/dynamo/build-info/ref8-patch.json
```

脚本核验实际 `vllm==0.31.0`、`vllm-omni==0.31.0rc1` 的发行信息与源码归属，验证六份文件后，只原子替换 `lora.py`。版本不符、源码已变、重复应用、另一个安装路径抢占导入等均失败。脚本不会安装依赖、联网或触碰模型资产。`lora.py` 的旧 Python 字节码缓存会清除，避免加载过期代码。

普通 wheel RECORD 原有内容保留；预期源码修改通过独立 `patch.lock.json` 和 before/after 报告记录，不能要求修改后该文件继续满足原 wheel 的源码哈希。

切换镜像默认非 root 用户，再执行：

```bash
python3 -B /opt/h3-ref8-patch/check_contract.py --installed \
  --report /opt/dynamo/build-info/ref8-contract.json
```

报告目录须提前授权默认用户写入。检查要求 Linux、非 root；读取上述固定位置的 patch 报告，再验证实际安装源码、Ref8/FL8 规则、错误 task／音频 shift／步数拒绝，并用真实安装的 **CPU Torch** 执行原生 sigma 函数。要求 N=8 得到 9 个边界、端点 1→0、每个内部值符合独立公式，视频 shift12 的内部值与 shift6不同、音频 shift3不变；运行前后不初始化 CUDA。

主 Dockerfile 的这两步应使用 `RUN --network=none`，进一步确保检查不依赖外网。该检查不读取或激活大型 LoRA；权重版本／SHA256、运行时依赖、媒体组件和许可分别由主镜像检查完成。

## 验收与维护边界

- 真实请求使用 `num_inference_steps=8`。0.31 官方 recipe 里“8步传9”的表格与实际代码冲突，已在版本审计中留证。
- CPU 检查覆盖原分辨率、无 latent_refine 的第一遍采样。0.31 新增 refine 会切片第二遍日程，本补丁不替其验收，也不默认开启。
- `PASS_INSTALLED_CONTRACT_CPU_ONLY` 仍不证明 LoRA 绑定、HSDP、GPU kernel、Gateway 协议或生成质量成功。
- 生产监测整合若需要修改所锁的六份 Omni 源码，应明确评审差异并重新产生锁与证据；不要删除 hash 检查。更换版本同理。
- 上游真正修复后应检查是否可删除本补丁；不能对新版源码反复强行替换。

`fixtures` 原始源码的版权／许可证标头保持不变；其 Apache-2.0 原文保存在本目录 `UPSTREAM-LICENSE`。
