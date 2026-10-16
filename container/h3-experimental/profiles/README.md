# 六种实验启动配置

结论：同一个镜像包含三个 Turbo 适配器和一个 FastH3 Dense4 适配器，同时保留无适配器的 Base 模式。**所有 Base checkpoint 都仍在只读挂载中；镜像不包含 Base 权重。** 每次 worker 只选一个 profile，固定到四张 H100。配置文件存在不代表已完成 GPU 或质量验收。

| profile | Base 分区 | 任务 | 请求步数 | 视频/音频 shift | 请求 LoRA |
|---|---|---|---:|---|---|
| `base-fl50` | FL2VA | T2VA、FL2VA | 50 | 12/3 | 不传 |
| `base-ref50` | Ref2VA | Ref2VA | 50 | 12/3 | 不传 |
| `turbo-fl8` | FL2VA | T2VA、FL2VA | 8 | 6/3 | 必须传匹配路径、scale=1 |
| `turbo-ref8` | Ref2VA | Ref2VA | 8 | 12/3 | 必须传匹配路径、scale=1 |
| `turbo-fl4` | FL2VA | T2VA、FL2VA | 4 | 6/3 | 必须传匹配路径、scale=1 |
| `fasth3-dense4` | FL2VA | 仅 T2VA | 4 | 12/3 | 不传；启动融合 |

Base 使用 `/userdata/mengsz/MiniMax-H3/FL2VA` 或 `Ref2VA`。适配器全部放在 `/opt/h3-adapters`，不会被 Base 挂载遮住。未启用超分、量化、Cache-DiT、编译或 offload，避免把新框架、修复和新优化混在首轮验收中。

## 执行位置

以下命令在**新的 Linux GPU Pod 容器内**执行，也可直接作为新 Pod 的 `command/args`；不在本机 Windows CMD 中直接执行，不需要 SSH。镜像构建阶段不会启动这些模型。实际申请 Pod 的 namespace/nodeSelector/image digest 和资源由外层部署明确指定，这个目录不会执行 kubectl。

先查看完整原生启动命令和请求合同，不读 Base、不启动模型：

```sh
python3 -B /opt/h3-experimental/profiles/launch.py --profile turbo-ref8
```

确认新 Pod 有四 GPU、Base 只读挂载和可写 emptyDir `/work` 后启动：

```sh
python3 -B /opt/h3-experimental/profiles/launch.py --profile turbo-ref8 --start
```

对应的 `start-<profile>.sh` 是这一命令的简短入口，需用 `bash /opt/h3-experimental/profiles/start-turbo-ref8.sh` 调用。

原生启动参数：TP1/Ulysses4/HSDP4、text-encoder TP4、VAE4、BF16、FLASH_ATTN、eager、cache none、并发1、无 CPU/offload。`profiles.json` 的 `common_argv` 和各 profile 的 `extra_start_args` 是实际参数来源。`launch.py` 会调用固定发布包明确注册的 `vllm-omni serve ... --omni`，不会改 Dingo Gateway 或生产服务，也不依赖 `vllm` 命令是否安装了额外拦截器。

## FastH3 专项

FastH3 在启动流式读权重时融合；不是普通 PEFT 请求 LoRA，不能与 Base/其他 Turbo 在同一融合 worker 中切换。`fasth3-dense4` 只有 T2VA，不能把请求改为 FL2VA 或 Ref2VA。不要将这份 Preview v1 Dense4 与 0.31 另行支持的 V2 全量/VSA 模型混用。

启动前强制离线完整验证 Base revision `9bfb6693f2cf6de171db46d1aa586f67d773a1da`：81 个文件、144,051,182,625 字节。哈希读取耗时不是推理耗时。若默认目录不匹配，由有存储权限的负责人准备另一个固定 revision 的只读目录，然后指定：

```sh
python3 -B /opt/h3-experimental/profiles/launch.py --profile fasth3-dense4 --model-path /readonly/pinned/FL2VA --start
```

这个选项只换 Base 路径，不会放宽验证或改变适配器、步数。禁止覆盖现有 Base，禁止仅凭目录名/元数据声称 revision 匹配。失败证据保存在 `/work/startup-*/fasth3-base-check.json`。

## 请求与验收

`profiles.json` 的 `request_contract` 包含应发送给原生 Omni API 的固定模型参数；还需由调用方提供提示词、图片/参考图和输出尺寸/时长。它不是可直接提交的完整生成请求。Turbo 必须在请求中激活对应 LoRA；仅启动时列出路径不能替代检查请求与日志中的激活证据。

8 步请求填 8，对应 9 个边界；4 步请求填 4，对应 5 个边界。FastH3 固定未偏移边界 `[0.999,0.749,0.5,0.25,0]`，不走普通均匀四步。Ref8 精确补丁只改指定 v1.0 768p 文件为 12/3，不改 FL4/FL8，也不改 Base 或其他 Ref 版本。

启动前检查记录仅证明版本、文件、Ref8 合同和可见 GPU，不证明模型载入/分片成功。还要等健康检查、核对实际适配器绑定、提交单样例、保存并解码音视频，再做人像/动作/画质/音频与口型检查。Base、各适配器的成功证据应分别保存。

所有启动证据写入 `/work/startup-<profile>-<time>`。删除 Pod 前先下载 `/work` 结果；emptyDir 不是持久化存储。本目录无联网安装或下载逻辑。
