<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Video Gateway 接口使用说明

本文说明 Dingo Video Gateway 当前提供的视频生成接口。示例使用 MiniMax-H3
双模型部署中的模型名：

- `minimax-h3`：FL2VA，可执行纯文本生成视频，也可使用一张或两张关键帧图片。
- `minimax-h3-ref2va`：Ref2VA，使用图片、视频和可选音频作为参考素材。

实际模型名、可用状态和媒体限制应以目标环境的 `GET /v1/models` 返回值为准。

## 快速开始

先设置 Gateway 地址并查询模型：

```bash
export GATEWAY_URL=http://<gateway-or-load-balancer>:8000
curl -sS "$GATEWAY_URL/v1/models" | jq
```

提交一个异步 FL2VA 文生视频任务：

```bash
curl -sS -X POST "$GATEWAY_URL/v1/videos" \
  -H 'Idempotency-Key: example-fl2va-001' \
  -F model=minimax-h3 \
  -F 'prompt=A cinematic sunrise over snow mountains' \
  -F seconds=5 \
  -F size=1344x768 \
  -F fps=24 \
  -F num_inference_steps=50 \
  -F seed=1101
```

成功后返回任务 JSON，响应头 `Location` 指向任务查询地址。保存返回的 `id`：

```bash
export TASK_ID=<video-task-id>
curl -sS "$GATEWAY_URL/v1/videos/$TASK_ID" | jq
```

任务完成后下载 MP4：

```bash
curl -fL "$GATEWAY_URL/v1/videos/$TASK_ID/content" \
  -o "$TASK_ID.mp4"
```

## 接口列表

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/v1/models` | 查询模型、可用状态和媒体能力 |
| `POST` | `/v1/videos` | 异步提交任务，返回任务 JSON |
| `POST` | `/v1/videos/sync` | 同步提交任务，成功时直接返回 MP4 |
| `GET` | `/v1/videos/{task_id}` | 查询单个任务状态和阶段耗时 |
| `GET` | `/v1/videos/{task_id}/diagnostics` | 查询相同任务信息及当前 attempt 的诊断明细 |
| `GET` | `/v1/videos` | 分页查询任务列表 |
| `GET`/`HEAD` | `/v1/videos/{task_id}/content` | 下载或检查生成结果 |
| `DELETE` | `/v1/videos/{task_id}` | 取消活跃任务，或删除终态任务制品 |
| `GET` | `/live` | 进程及 Gateway owner lease 存活检查 |
| `GET` | `/ready`、`/health` | 服务就绪检查 |
| `GET` | `/metrics` | Prometheus 文本指标 |
| `GET` | `/internal/v1/performance/records` | 短期性能快照增量拉取，缓存默认开启，独立认证 |

`POST /v1/videos/stream` 当前未实现，固定返回 `404 unsupported_endpoint`。
内部性能缓存使用 `performance_cache` 配置，默认开启并保留 300 秒（5 分钟），每进程最多 50000 条、64 MiB，容量不足时提前淘汰，不扫描任务存储或制品；
未配置独立 Token 时 Gateway 仍正常启动并缓存，但拉取返回 401。可设置 `enabled: false` 显式关闭。
游标、分页、认证和漏采边界见[HTTP 定时拉取](performance-collection.md#http-定时拉取)。
`/ready` 表示 Gateway 自身、任务存储和制品存储可用，不要求每个模型都有 Worker。
模型是否至少发现一个已注册 Worker，应查看 `/v1/models` 中对应项的 `available`；
`available=true` 也不表示 Worker 当前空闲，满载时任务仍可能进入队列。

## 提交请求

视频提交接口只接受 `multipart/form-data`。普通参数是文本 part，参考文件使用
`input_reference` 或可重复的 `input_references` 文件 part。

### 通用参数

| 参数 | 必填 | 默认值 | 约束 |
|---|---:|---|---|
| `model` | 未配置默认模型且有多个视频模型时必填 | 配置中的默认模型；仅有一个视频模型时可自动选用 | 必须属于 Gateway 已配置的 Worker pool；`/v1/models` 可能还列出不能用于视频提交的上游模型 |
| `prompt` | 是 | 无 | UTF-8，不超过 32 KiB |
| `seconds` | 三选一 | 无 | 与 `num_frames`、`extra_params.duration` 三选一；范围 4–15 秒 |
| `num_frames` | 三选一 | 无 | 范围 96–360；与另外两种时长写法互斥 |
| `size` | 尺寸写法之一 | 无 | `WIDTHxHEIGHT`，也可改用 `width` 与 `height` 或 `resolution`＋`ratio` |
| `width`、`height` | 尺寸写法之一 | 无 | 必须同时出现，不能与 `size` 同时使用 |
| `resolution` | 无显式尺寸时必填 | 无 | `768P`、`1080P`、`2K`；显式尺寸同时提供时校验档位一致 |
| `ratio` | 无显式尺寸且不能自适应时必填 | FL2VA 有关键帧时可自适应 | 六种命名比例或 `adaptive`；Ref2VA 暂不支持自适应 |
| `aspect_ratio` | 否 | 根据尺寸推导 | `ratio` 的兼容名称；同时提供必须相同，`adaptive` 适用相同限制 |
| `fps` | 否 | `24` | MiniMax-H3 当前只接受 24 |
| `num_inference_steps` | 否 | `50` | 2–200；别名为 `steps`，两者不能同时提供 |
| `guidance_scale` | 否 | 模型默认值 | 0–20 |
| `seed` | 否 | 不设置 | 无符号 32 位整数（含 0）；指定时原值下传；未指定时按 `seed_defaults` 配置处理，未配置则不补、不传 |
| `negative_prompt` | 否 | 无 | UTF-8，不超过 32 KiB |
| `generate_sound` | 否 | `true` | 接受 `true/false/1/0` |
| `output_format` | 否 | `mp4` | 当前只支持 `mp4` |
| `num_outputs_per_prompt` | 否 | `1` | 当前只支持 1 |
| `user` | 否 | 无 | 记录在规范化请求中的调用方标签；不用于认证 |

默认基础生成尺寸必须同时满足：

- 宽和高各为 256–2048，且都是 32 的倍数；
- 总像素不超过 `768 × 1344`；
- 宽高比与支持的命名比例之一相差不超过 5%。

### 分辨率档位与比例快捷入口

```bash
curl -sS -X POST "$GATEWAY_URL/v1/videos" \
  -F model=minimax-h3 \
  -F 'prompt=A cinematic sunrise over snow mountains' \
  -F seconds=4 -F resolution=2K -F ratio=16:9
```

| ratio | 768P | 1080P | 2K |
|---|---|---|---|
| 16:9 | 1344x768 | 1920x1080 | 2560x1440 |
| 21:9 | 1536x672 | 2208x936 | 2944x1248 |
| 4:3 | 1024x768 | 1440x1080 | 1920x1440 |
| 1:1 | 768x768 | 1080x1080 | 1440x1440 |
| 3:4 | 768x1024 | 1080x1440 | 1440x1920 |
| 9:16 | 768x1344 | 1080x1920 | 1440x2560 |

768P 六种比例已按2026-11-10官方API纯文生视频的实际输出尺寸对齐；21:9的短边为672，
不保证所有画幅短边均为768。2K的16:9、21:9已与官方实测一致，其余组合沿用现有白名单。
这些交付尺寸与超分内部基础画布分开维护，本次对齐不调整超分基础画布或显式尺寸准入。
1080P 是 Gateway 扩展；高分辨率执行仍使用配置的 SeedVR2 路径。已有基础画布、超分、
裁剪和缩放策略不变；关闭超分或无兼容 Worker 时不会降档执行。
`GET /v1/models` 中 `video_capabilities.resolution_presets` 列出当前配置准入的组合，
不代表 Worker 当前已就绪或任意输入负载均能成功。

参数优先级与一致性规则：

- 有 `size` 或成对宽高时，以显式尺寸确定画布；同时指定的档位、比例必须一致，否则400。
- 无显式尺寸时必须提供 `resolution`，不新增默认档位；具体比例按表选尺寸，FL2VA自适应按下述规则处理。
- `ratio` 和 `aspect_ratio` 同时提供时必须相同。纯文生视频和Ref2VA在无显式尺寸时必须指定具体比例。
- FL2VA 有关键帧且未显式指定尺寸时，可以省略比例或指定 `ratio=adaptive`：
  768P 按首帧实际比例计算画布，短边初值768、取整前像素预算768×1344，最后两轴各自四舍五入到32的倍数。
  支持六种命名比例之外的画布，例如3:2图片得到1152×768。只有尾帧时按尾帧计算。
  首尾帧可以具有不同尺寸和比例，画布由首帧确定；Worker沿用现有图片处理，不新增裁剪。
  1080P和2K仍选择当前档位允许的预设，要求输出比例与首帧相差不超过3%；不匹配时拒绝。
- FL2VA 显式比例与参考图片不匹配时返回错误，不静默忽略。Ref2VA 暂不支持 `adaptive`。
  显式尺寸、具体比例入口保留首尾同尺寸要求和3%的图片比例校验；768P自动计算画布不受预设比例的3%限制。
  官方768P的21:9画布为1536x672，严格21:9图片与它的约2.04%误差可以通过。
  当前H3会直接缩放关键帧到目标画布，允许范围内仍可能有轻微形变；这是Gateway工程阈值，
  不表示已验证官方图生视频的内部处理方式。
- 显式基础尺寸只有映射表中的精确尺寸归类为768P；其他合法自定义基础尺寸仍可单独使用，
  但不能与无法匹配的 `resolution` 混用。1800x1080、2400x1440 等现有兼容尺寸保留显式入口。
- `short_edge` 保持旧校验，仅允许768，不参与查表，也不是每个档位的实际短边。

新旧写法规范化为相同的宽高、比例和执行计划；例如 `resolution=2K,ratio=16:9` 与
`size=2560x1440` 在其他参数及参考文件相同时具有相同幂等摘要。自适应结果在创建任务时固定，
派发、重试和恢复不重新选择尺寸。任务JSON返回最终 `size`、`ratio`，能够明确归类时返回
`resolution`；未知档位不虚构值。下载仍返回视频文件。
768P自适应得到的非预设画布也返回 `resolution=768P`，例如1152×768返回 `ratio=3:2`。
本快捷入口适用于视频生成接口，不改变 `/v1/context-ir` 编译接口的基础画布参数。

### 上传已有视频单独超分（SeedVR2）

这一路径仍使用 `POST /v1/videos`，但 `model` 必须指向配置了 `seedvr2` adapter、
`sr_only` workflow 和 detached 执行模式的独立 Worker 池。不会执行 H3 基础生成。
先通过 `GET /v1/models` 确认目标模型的 `video_capabilities.workflow` 为 `sr_only`。
以下 `seedvr2-3b-sr` 仅为示例名称，以实际部署公开的模型名为准。

```bash
curl -sS -X POST "$GATEWAY_URL/v1/videos" \
  -H 'Idempotency-Key: sr-example-001' \
  -F model=seedvr2-3b-sr \
  -F operation=sr_only \
  -F resolution=1080P \
  -F 'input_references=@source.mp4;type=video/mp4'
```

| 字段 | 要求 |
|---|---|
| `model` | 独立超分池的公开模型名 |
| `operation` | 必须是 `sr_only` |
| `resolution` | `1080P` 或 `2K`，输出比例按源视频匹配支持的画幅 |
| `input_references` | 恰好一个源MP4，也接受单数名称 `input_reference` |

不接收 `prompt`、`seed`、`seconds`、`size`、`ratio` 或 `num_inference_steps` 等生成参数。
源视频要求MP4、H.264/yuv420p、24 FPS、1–362帧、最多128 MiB；最多一条AAC音轨。
源宽高各在256–2048之间，总像素不超过1,100,000，宽高比在0.4–2.5之间，且能匹配支持的超分画幅。
请求还受Gateway配置的单文件、总请求体和存储配额限制；使用128 MiB源文件时需同步配置足够的上传限制。

任务查询、结果下载、取消沿用本文通用接口。输出保留源帧数、帧率及音轨有无，
不生成新对白；最终画幅以任务响应的 `size` 为准。Gateway和独立Worker需要配置一致的
`storage_id`并挂载同一任务制品存储，源视频按任务目录及哈希核验后交给Worker。

### 超分尺寸契约

MiniMax-H3 的超分默认关闭。启用后，`size` 或 `width`＋`height` 表示最终交付尺寸；
`resolution`＋`ratio` 是查表推导最终尺寸的快捷入口；仍只支持所选策略准入的尺寸，
不会把任意高分辨率直接交给基础模型生成。没有 `output_short_edge` 请求字段。

```yaml
adapter:
  name: minimax_h3
  workflow: ref2va
  super_resolution:
    enabled: true
    profile: minimax_h3_seedvr2.v3
```

FL2VA 可配置 `workflow: fl2va`；超分必须使用 `execution_mode: detached`，且匹配
实际声明同一策略并已就绪的组合 Worker。完整的 14 个白名单尺寸、H3 基础尺寸、SR 内部尺寸、窗口、
裁剪区域和缩放规则以[超分画幅契约](super-resolution.md)为准。

v3 的全部 2K 无须额外交付缩放；仅 1920x1080 和 1080x1920 两种 1080P
先裁剪再 Lanczos 等比缩小。其他尺寸直接居中裁剪或直接编码。每轴最多裁掉 3%，
只执行一次最终视频有损编码，Worker 外层保留 Base 音轨并 stream copy 回封。

v1保留1890x1080、2520x1440及对应竖屏的原比例缩放策略；v2保留1800x1080和
2400x1440。策略需显式配置，v3不会被派发给仅声明旧策略的Worker。符合基础尺寸
规则的请求继续走无超分路径；配置超分不会自动提高所有请求的分辨率。

超分能力不可用时返回503 `super_resolution_unavailable`，不静默降低分辨率。
最终尺寸和SR计划参与幂等指纹；仅调整SR配置不会改变基础请求指纹。
`/v1/models`的尺寸列表表示已配置的规格，实际可用性还取决于兼容Worker。
`aspect_ratio`用于一致性校验；兼容字段`short_edge`只接受768，不计算宽高。

组合Worker通过环境变量接收运维配置，以下示例针对已固定的0.32原生SR镜像：

```json
{"model":"/models/SeedVR2-3B-fp16","work_root":"/work/seedvr2","gpus":8,"engine_identity":"<pinned-image-and-model-revisions>","profile":"minimax_h3_seedvr2.v3","image_digest":"<image-repository>@sha256:<64-hex-digest>"}
```

变量名为`DINGO_VIDEO_SEEDVR2_CONFIG`，它不是客户端参数。还需启用detached和
`DINGO_VIDEO_BINARY_RESULTS=1`，执行容量为1；包含四卡适配的 v2／v3 实现支持4或8张共享GPU，
旧镜像仍需8卡。卡数和版本约束见[超分说明](super-resolution.md)。H3和独立的loopback SeedVR2子进程
共用整个任务的执行许可。v3的1080P／2K分别使用33／65帧窗口，4帧重叠、单步Euler、
Lab颜色修正；H3的seed按请求和Gateway默认策略解析，未配置默认策略且请求未指定时不传seed，v2／v3的SR服务使用固定seed 7723。

基础视频检查点绑定完整请求及引擎身份，只能在同一任务后续attempt中验证复用；
不跨任务复用，也不对未确认结束的SR提交盲目重发。内部失败先暂停接单，由视频恢复
监督机制检查引擎及设备同步状态，确认安全才恢复；进程退出或健康检查持续失败时终止
Worker。OOM异常本身不等于进程死亡，内部原因通过诊断和日志保留，对外错误仍遵循
Gateway错误契约。128MiB结果上限保持不变，超限不自动降质。

超分任务的查询`metrics.super_resolution`提供版本1、秒单位、固定白名单的阶段耗时；
原H3阶段保留在`metrics.model_execution`。`diagnostics.stage_durations`继续保留
`h3_base_total_s`、`sr_service_s`、`sr_finalize_s`和后处理细分字段。SR服务总耗时包含
排队和内部处理，不能解释成纯GPU时间；父子阶段和流水重叠阶段不能直接相加。

Gateway的`/metrics`新增：

- `dingo_video_super_resolution_stage_duration_seconds{pool,stage}`：Histogram，
  采用与任务阶段相同的桶；仅在成功接受SR结果交接时记录，每阶段每次一次。
- `dingo_video_super_resolution_requests_total{pool,tier,status}`：Counter，
  tier为`1080p`／`2k`，status为`completed`／`failed`／`cancelled`／`expired`。
  按已提交的终态转换记录事件，内部重试不另计；进行中的请求及任务创建前的拒绝不计入。

> [!WARNING]
> 当前版本的请求 Counter 在已完成、失败或取消的任务随后转为制品过期时，
> 还会增加一次 `status="expired"`。因此不能把所有 status 相加当作唯一业务请求总数。
> 查询 `status="completed"` 可查看成功完成数；首次终态去重修复尚未包含在本次文档提交中。

不超分请求不增加以上指标。Counter为启用SR的pool预先暴露零值；Histogram不补零。
同其他进程内指标一样，重启会清零，采集间隔及崩溃可能造成事件丢失，不替代业务审计。

### 其他兼容字段

`task`、`flow_shift`、`audio_flow_shift` 和 `extra_params` 主要用于兼容底层
vLLM-Omni 调用。普通调用建议省略，由 Gateway 按目标 pool 填充。
`quality`、`image_reference`、`video_reference` 和 `audio_reference` 等字段当前不支持；
参考文件统一使用下文的 `input_reference(s)`。

默认配置还限制单次请求的参考文件合计不超过 256 MiB、multipart part 不超过 32 个；
部署可以把这些限制调低。`GET /v1/models` 中的 `video_capabilities` 会返回模型适配器的
格式、单文件大小、参考数量和结果大小上限。

### 幂等提交

建议每次业务请求携带稳定且唯一的 `Idempotency-Key`：

```text
Idempotency-Key: <1 至 256 字节的业务请求键>
```

在原任务及其幂等记录仍保留、且请求所用 pool 配置与适配器兼容版本未变化时，
相同 key 和相同请求会返回原任务，不会重复生成。请求摘要包含这些版本信息：
滚动升级或配置变更期间，
即使 HTTP 请求内容相同，也可能返回 `409 idempotency_conflict`。原任务及其记录被清理后，
再次使用相同 key 可能创建新任务。网络超时后仍应首先保持 key 和请求内容不变进行重试；
若遇到 409，应查询已知任务 ID 或核对部署配置，不要盲目改用新 key 提交。

### FL2VA 请求

FL2VA 支持三种输入：

| 输入 | 文件字段 | 行为 |
|---|---|---|
| 无参考图 | 不上传文件 | 文本生成视频，内部任务为 `t2va` |
| 一张图片 | `input_reference` 或一个 `input_references` | 默认作为首帧；可用 `frame_indices=[-1]` 指定为尾帧 |
| 两张图片 | 两个 `input_references` | 分别作为首帧和尾帧，`frame_indices` 为 `[0,-1]` |

支持 JPEG、PNG、WebP。每张图片不超过 30 MiB，宽高各为 256–5760，宽高比范围
0.4–2.5。两张关键帧尺寸必须相同；输出尺寸的宽高比必须与关键帧基本一致。

一张首帧图片示例：

```bash
curl -sS -X POST "$GATEWAY_URL/v1/videos" \
  -H 'Idempotency-Key: example-fl2va-image-001' \
  -F model=minimax-h3 \
  -F 'prompt=The camera slowly moves forward' \
  -F seconds=5 \
  -F size=1344x768 \
  -F num_inference_steps=50 \
  -F 'input_reference=@/path/to/first.png;type=image/png'
```

首尾两张图片示例：

```bash
curl -sS -X POST "$GATEWAY_URL/v1/videos" \
  -H 'Idempotency-Key: example-fl2va-two-images-001' \
  -F model=minimax-h3 \
  -F 'prompt=Transition naturally between the two scenes' \
  -F seconds=5 \
  -F size=1344x768 \
  -F num_inference_steps=50 \
  -F 'frame_indices=[0,-1]' \
  -F 'input_references=@/path/to/first.png;type=image/png' \
  -F 'input_references=@/path/to/last.png;type=image/png'
```

### Ref2VA 请求

Ref2VA 至少需要一张图片或一个视频；音频不能单独使用。文件按 multipart 中出现的顺序
传给模型。可组合：

- 最多 9 张图片；
- 最多 3 个视频；
- 最多 3 个音频；
- 所有参考文件合计最多 12 个；
- 参考视频总时长不超过 15 秒，参考音频总时长不超过 15 秒。

媒体限制如下：

| 类型 | 格式/编码 | 单文件大小 | 其他限制 |
|---|---|---:|---|
| 图片 | JPEG、PNG、WebP | 30 MiB | 宽高 256–5760，宽高比 0.4–2.5 |
| 视频 | MP4/MOV，H.264 或 H.265；如含音轨须为 AAC | 50 MiB | 2–15 秒，23.976–60 FPS，宽高 256–5760 |
| 音频 | WAV、PCM 编码 | 15 MiB | 2–15 秒 |

图片、视频和音频混合示例：

```bash
curl -sS -X POST "$GATEWAY_URL/v1/videos" \
  -H 'Idempotency-Key: example-ref2va-001' \
  -F model=minimax-h3-ref2va \
  -F 'prompt=Create a coherent five second scene using these references' \
  -F seconds=5 \
  -F size=1344x768 \
  -F num_inference_steps=50 \
  -F 'input_references=@/path/to/reference.png;type=image/png' \
  -F 'input_references=@/path/to/reference.mp4;type=video/mp4' \
  -F 'input_references=@/path/to/reference.wav;type=audio/wav'
```

Gateway 始终检查上传参考文件的签名、声明的 MIME 类型以及媒体是否可以解码。对于 Worker
生成的最终结果，adapter `validate_media` 默认是 `false`：Gateway 仍检查可信 Worker
结果描述符、任务/attempt fencing、安全路径、普通文件、文件大小及上限，但有声结果在发布
阶段不打开或读取 MP4 内容，避免共享存储首次读取的长尾。部署显式设置
`validate_media: true` 后，Gateway 才会读取 MP4 签名，并通过 PyAV 检查 H.264/AAC、帧率、
帧数、分辨率及音视频时长；该严格校验会增加完成阶段耗时。`generate_sound=false` 仍需打开
结果并按需去除音轨，不属于上述有声快速路径。仅修改上传文件扩展名或伪造
`Content-Type` 不会绕过输入校验。

## 任务与结果

### 任务类型与上传素材用量

视频任务的创建、查询和列表响应均包含 `task` 与 `input_usage`，无需等待生成完成：

```json
{
  "task": "ref2va",
  "input_usage": {
    "image_count": 4,
    "video_duration_seconds": 10.5
  }
}
```

`task` 是归一化后的实际任务类型。H3 FL2VA 池无参考图时为 `t2va`，有参考图时为
`fl2va`；Ref2VA 池为 `ref2va`。不根据公开模型名猜测，不将单首帧另命名为新任务类型。

`image_count` 统计上传的参考图片，不包含视频抽帧。`video_duration_seconds` 是所有上传参考
视频经媒体解析得到的时长之和，单位秒；与目标生成时长无关，不因内部裁剪、重试或后处理
重复计数。无图片或无视频时，对应统计为 `0`。结果来自已持久化的请求元数据，幂等重放、
后续查询和 Gateway 重启后保持一致。

旧任务缺失必要元数据或未测得视频时长时，对应值为 `null`，不把未知值当作零；未知任务类型
也返回 `task: null`。只做超分的任务沿用 `operation: sr_only`，没有生成任务类型时不伪造
`fl2va/ref2va`。音频、内部生成的视频不计入上传参考视频统计。素材校验失败而未创建任务的
请求仍返回原有错误响应。

### 异步任务

`POST /v1/videos` 在任务被持久化并进入队列后返回。默认 HTTP 状态为 `202`；部署可为
兼容客户端配置成 `200`，调用方应以响应体中的任务状态为准。

典型响应：

```json
{
  "id": "video-...",
  "object": "video",
  "model": "minimax-h3",
  "status": "queued",
  "progress": 0,
  "created_at": 1789950000,
  "expires_at": 1790036400,
  "size": "1344x768",
  "seconds": 5.0,
  "num_frames": 120,
  "fps": 24,
  "seed": 1101,
  "seed_generated": false,
  "seed_source": "request"
}
```

`seed_source` 表示来源：`request`（请求指定）、`default_fixed`（配置固定值）、
`default_random`（Gateway 随机生成）。仅随机生成时 `seed_generated: true`。
旧任务保留已有 seed 和标记，不追溯补充来源。

Gateway 顶层可配置以下两种默认策略之一，适用于 H3 FL2VA 和 Ref2VA：

```yaml
seed_defaults:
  strategy: fixed
  value: 42
```

或：

```yaml
seed_defaults:
  strategy: random
```

请求显式 seed 始终优先，包括 `0`；非法请求 seed 返回参数错误，不回退到默认值。
`fixed` 必须提供无符号 32 位整数 `value`（0～4294967295），不接受布尔值、小数或字符串；
`random` 不允许提供 `value`。空配置、未知策略或字段会导致启动校验失败。
随机策略使用 `secrets.randbits(32)`，不保证不同任务 seed 唯一。

**不配置 `seed_defaults` 时保持原行为**：未指定 seed 的请求不补、不传，任务响应也不包含
seed 及来源字段。此时随机行为由 Worker/引擎决定，Gateway 不保证跨重试的结果可复现。
配置策略后，最终 seed 在新任务创建时保存到 request.json 和任务元数据，并在派发、重试、
恢复时复用；配置变更只影响新任务。请求摘要不包含默认补入的 seed 或来源；相同幂等键、
相同原始请求即使经过并发提交或策略变更，仍返回原任务及其 seed。显式传 seed 与省略 seed
是不同请求。固定 seed 不保证跨硬件、版本的输出逐像素一致。

任务详情、诊断、下载响应头 `X-Video-Seed` 可查看最终 seed；性能记录的 `generation`
包含 seed 和 seed_source。它们不加入 Prometheus 标签。此配置不改变 ContextIR 的
`context_ir_seed`、LLM Worker 或 SeedVR2 内部独立的随机参数。

公开状态及含义：

| 状态 | 含义 |
|---|---|
| `queued` | 在 Gateway 队列中等待调度 |
| `in_progress` | 正在下发、在 Worker 预取队列等待、执行或由 Gateway 后处理 |
| `completed` | 结果可下载 |
| `failed` | 任务失败；查看 `error` |
| `cancelled` | 任务已取消 |
| `expired` | 结果已过期或已通过 DELETE 清理 |

当前 `progress` 只表示终态：完成为 100，其余为 0，不代表去噪步数进度。

完成后的状态会增加结果和耗时字段，例如：

```json
{
  "id": "video-...",
  "object": "video",
  "model": "minimax-h3",
  "status": "completed",
  "progress": 100,
  "media_type": "video/mp4",
  "file_name": "video-....mp4",
  "bytes": 20971520,
  "sha256": "...",
  "requested_seconds": 5.0,
  "seconds": 5.175,
  "duration_s": 5.175,
  "metrics": {
    "queue_wait_s": 0.12,
    "worker_queue_wait_s": 0.35,
    "inference_time_s": 5.0,
    "finalize_time_s": 0.28,
    "model_execution": {
      "schema": "minimax_h3.v1",
      "unit": "seconds",
      "stages": {"encode_prompt": 0.1, "diffuse": 3.5, "decode": 0.8}
    }
  },
  "inference_time_s": 5.86
}
```

耗时字段含义：

- 响应中的 `metrics` 是当前任务的逐任务计时元数据，与 Prometheus `/metrics` 端点无关；
  内部结构是 Gateway 定义的扩展。原生 vLLM-Omni 也有 `metrics`，但结构与口径不同。
- 顶层 `inference_time_s`：从 Gateway 创建任务到写入终态前取值时刻的服务端任务总时间，
  包含排队和重试；不包含创建任务前的上传、最终终态写入 RPC 和客户端结果下载。
- `metrics.queue_wait_s`：任务在 Gateway 队列等待的时间。
- `metrics.worker_queue_wait_s`：任务已交给 Worker 后，在 Worker 预取队列等待执行的时间。
- `metrics.inference_time_s`：Worker 执行时间。detached Worker 在取得执行 slot 后、写入
  `running` 状态之前开始计时；结果生成、输出处理及心跳协程收尾完成后，在构造
  `completed` 状态时取值。该计时不包含随后 `_record_terminal()` 的终态状态写入，
  也不包含 `worker_queue_wait_s`；旧的 non-detached Worker 由 Gateway
  记录完整 direct call 耗时。Worker 输出格式化、MP4 编码及制品写入的细分时间另见
  诊断接口的 `diagnostics.stage_durations.output_total_s`。旧 detached Worker 未提供外层计时
  时返回 null，不用输出编码时间替代。历史任务不回填修复后的计时。
- `metrics.finalize_time_s`：Gateway 校验、处理并发布结果制品的耗时；计时在读取最新任务及
  向 etcd 写入 `completed` 终态之前结束，**不包含最终任务状态写回耗时**。
  可恢复 finalizer 当前只保存最后成功处理那一轮的耗时，不包含等待执行名额、先前失败
  尝试、重试退避及接管恢复等待；因此不能视为整个 `finalizing` 状态的驻留时间。
- 普通任务响应不再返回 `stage_durations`；需要详细阶段信息的调用方应改用诊断接口。

发生 Worker 故障重试时，`metrics` 和诊断阶段明细只描述当前/最终 attempt；顶层
`inference_time_s` 仍从任务最初创建开始计算，因此包含此前失败 attempt、重新排队和重试等待。
这些分项不能假定相加后等于顶层总时间。

模型专用阶段放在 `metrics.model_execution`，其 `schema` 标识版本，`unit` 为 `seconds`，
`stages` 保存该 schema 下的阶段值。通用任务序列化不依赖具体模型阶段名称。
当前 `minimax_h3.v1` 从原生 Omni profiler 提取以下键，仅有有效数据时返回，缺失不代表零：

| 键 | 含义与关系 |
|---|---|
| `encode_prompt` | 提示词/条件编码；Ref2VA 可包含图片和视频的视觉条件处理，不是纯文本时间 |
| `diffuse` | 原生 profiler 的去噪计时 |
| `decode` | 输出解码总阶段，包含下列视频/音频解码 |
| `video_decode` | Video VAE 解码，是 `decode` 的子阶段 |
| `audio_decode` | Audio VAE 解码，是 `decode` 的子阶段 |
| `reference_video_prepare` | 参考视频准备方法的计时；不承诺涵盖所有前置上传/解码操作 |
| `reference_visual_encode` | 参考图片和视频的视觉条件/VAE 编码合计；当前原生计时不拆图片与视频 |
| `reference_audio_encode` | 参考视频内音轨及独立音频的条件编码合计 |

原生字段一一映射、计时边界、空调用说明、JSON Schema 导出命令及客户端兼容规则见
[模型阶段计时规范](model-timing-schemas.md)。

Gateway 还会在结果成功交接进入 `finalizing` 时，将这些阶段汇总到
`/metrics` 的 `dingo_video_model_stage_duration_seconds` Histogram，标签为
`pool`、`schema`、`stage`，可用于 Grafana 平均值/P95 展示。不包含 `task_id`，
缺失阶段不补零，不回填历史任务，也不增加 etcd 查询。该指标按交接记录时间统计，
不是按阶段开始时间统计；父子阶段不能相加。详见[Gateway 可观测性与指标](observability.md)。

模型阶段来自 `MiniMaxH3Pipeline` 的原生 profiler，需要 Worker 开启
`--enable-diffusion-pipeline-profiler`。该 profiler 会同步 GPU，可能影响性能；Gateway 不额外
加入 GPU 同步。这里透传原生返回值，不聚合各 rank，也不将父子阶段求和。
当前已核对 request 模式；step 模式多请求交错时，原生实例级 profiler 的逐请求隔离仍需
单独验证，不能仅凭这些字段宣称获得准确的多请求阶段归因。

### 任务诊断

`GET /v1/videos/{task_id}/diagnostics` 返回普通查询的全部字段，再增加 `diagnostics`：

```bash
curl -sS "$GATEWAY_URL/v1/videos/$TASK_ID/diagnostics" | jq
```

```json
{
  "id": "video-...",
  "object": "video",
  "status": "completed",
  "metrics": {
    "inference_time_s": 125.4,
    "model_execution": {
      "schema": "minimax_h3.v1", "unit": "seconds", "stages": {"diffuse": 120.0}
    }
  },
  "diagnostics": {
    "schema_version": 1,
    "internal_status": "completed",
    "attempt": 1,
    "pool_id": "fl-pool",
    "worker_instance_id": 7,
    "scope": "current_attempt",
    "stage_durations": {"output_total_s": 0.58}
  }
}
```

示例省略了其他普通查询字段。诊断读取相同任务记录，不调用 Worker、不扫描日志、
不读取 MP4；不存在的任务返回 404。只展示已持久化的数据，不重建历史 attempt。
模型阶段在 `metrics.model_execution` 中；原生 orchestration 的毫秒字段不会混入。

| 诊断字段 | 含义 |
|---|---|
| `schema_version` | 诊断对象的结构版本，当前为 1；与模型计时的 `model_execution.schema` 独立 |
| `internal_status` | 内部状态，可区分普通查询统一显示为 `in_progress` 的 `dispatching` 和 `finalizing` |
| `attempt` | 当前执行次数，首次执行为 1，重试一次为 2；尚未分配执行的任务可以为 0 |
| `pool_id` | 任务所属 Worker 池 |
| `worker_instance_id` | 当前或最后分配的 Worker 实例标识，未分配时可以为 null |
| `scope` | 当前为 `current_attempt`，不表示全部历史 attempt 的累计数据 |
| `stage_durations` | 已保存的阶段明细，当前主要为 Worker 输出处理细项；没有数据时为 `{}` |

当前耗时信息的覆盖范围：

| 位置 | 覆盖内容 |
|---|---|
| `metrics.queue_wait_s`、`metrics.finalize_time_s` | Gateway 排队和结果处理的高层耗时 |
| `metrics.worker_queue_wait_s`、`metrics.inference_time_s` | Worker 预取等待和完整执行耗时 |
| `metrics.model_execution.stages` | 按模型 schema 定义的模型阶段，例如条件编码、去噪、解码 |
| `diagnostics.stage_durations` | 主要是 Worker 输出归一化、MP4 编码、制品写入的细分耗时 |

诊断明细还可能包含 `worker_queue_wait`，与 `metrics.worker_queue_wait_s` 重复，这是
现有内部存储字段。构造诊断响应时不再额外复制 `queue_wait` 和 `finalize`；历史任务或
其他适配器保存的键可能不同。客户端不应要求这份字典具有固定键集合。

**当前未提供 Gateway 处理细项的 API 计时**，例如媒体打开/检查、finalizer 等待、
结果发布子步骤及终态写入 RPC。Gateway 媒体检查的细分日志按下面的日志开关读取。
诊断接口也不提供完整阶段时间线、历史 attempt 列表、各 GPU rank 计时或下载耗时。

新诊断端点需要更新 Gateway；模型计时还需要更新 Worker 并开启相应 profiler。
旧任务不会自动补采数据，单独更新 Gateway 也不能恢复旧 Worker 未上报的模型阶段。

当前使用 detached binary result 的 Omni Worker 还会附带以下**可选诊断键**。这些键用于
性能定位，属于 Worker 实现细节，不是固定 API 契约；旧 Worker、其他适配器或后续实现均可
缺失、增加或调整它们：

| 键 | 含义与关系 |
|---|---|
| `output_total_s` | Worker 整个输出处理范围，是下面输出子阶段的父级总时间 |
| `output_normalize_s` | 输出帧归一化 |
| `encode_queue_s`、`encode_work_s`、`encode_resume_s` | 编码线程池等待、实际 MP4 编码/混音、协程恢复，均包含在 `output_total_s` 中 |
| `artifact_queue_s`、`artifact_work_s`、`artifact_resume_s` | Worker 制品写入线程等待、实际工作和协程恢复，均包含在 `output_total_s` 中 |
| `artifact_hash_s`、`artifact_open_s`、`artifact_write_s`、`artifact_file_fsync_s`、`artifact_close_s`、`artifact_rename_s`、`artifact_dir_fsync_s` | `artifact_work_s` 内部的文件处理子阶段 |

因此不能把 `output_total_s`、`encode_*` 和 `artifact_*` 全部相加；这会重复计算父子阶段。
`DINGO_VIDEO_MEDIA_TIMING=1` 控制 Gateway 媒体检查详细日志，默认关闭；其
`header_read_s`、`indexed_open_s` 等不会自动进入该诊断对象，与原生模型 profiler 是两个开关。

### 同步任务

`POST /v1/videos/sync` 使用相同的校验、队列、Worker 和制品流程。成功时直接返回
`video/mp4`，常用响应头包括：

- `X-Video-Id`：任务 ID；同步请求也会创建可查询的持久化任务。
- `X-Video-Seed`：任务保存的最终 seed；请求显式指定或 Gateway 默认策略补入时返回，无 seed 时不返回。
- `X-Video-Frames`、`X-Video-FPS`、`X-Video-Duration-Seconds`：实际输出信息。
- `ETag`：基于结果 SHA-256 的实体标签。

示例：

```bash
curl -fL -X POST "$GATEWAY_URL/v1/videos/sync" \
  -H 'Idempotency-Key: example-sync-001' \
  -F model=minimax-h3 \
  -F 'prompt=A short cinematic shot' \
  -F seconds=5 \
  -F size=1344x768 \
  -F num_inference_steps=50 \
  -D /tmp/video-response.headers \
  -o /tmp/result.mp4
```

同步等待超时时返回 `504 gateway_timeout`，响应头 `X-Video-Id` 中的任务仍可继续通过
异步查询接口轮询和下载。生成失败返回 `422 video_generation_failed`，同样携带
`X-Video-Id`。

### 列表与分页

```bash
curl -sS "$GATEWAY_URL/v1/videos?model=minimax-h3&status=completed&limit=20&order=desc" | jq
```

查询参数：

- `model`：按模型对应的 pool 过滤；
- `status`：按任务状态过滤；
- `limit`：1–100，默认 20；
- `order`：`asc` 或 `desc`，默认 `desc`；
- `after`：使用上一页的 `last_id` 作为游标。

响应包含 `data`、`has_more`、`first_id` 和 `last_id`。

### 下载结果

完整下载：

```bash
curl -fL "$GATEWAY_URL/v1/videos/$TASK_ID/content" -o "$TASK_ID.mp4"
```

只检查元数据：

```bash
curl -sSI "$GATEWAY_URL/v1/videos/$TASK_ID/content"
```

接口支持单段 HTTP Range 和 `If-None-Match`：

```bash
curl -sS -H 'Range: bytes=0-1048575' \
  "$GATEWAY_URL/v1/videos/$TASK_ID/content" -o first-megabyte.bin
```

成功响应可能包含 `Content-Length`、`Content-Range`、`Accept-Ranges: bytes`、`ETag`、
`X-Video-Id`、`X-Video-Seed`、`X-Video-Frames`、`X-Video-FPS` 和
`X-Video-Duration-Seconds`。

结果未完成时返回 `409 video_not_ready`；任务失败或取消返回
`422 video_generation_failed`；任务仍保留过期记录、或已完成任务的制品不可用时返回
`410 video_expired`。过期记录最终清理后，任务查询和内容下载均返回 `404 video_not_found`。
不支持多段 Range，无效范围返回 `416 range_not_satisfiable`。

### 取消与删除

```bash
curl -sS -X DELETE "$GATEWAY_URL/v1/videos/$TASK_ID" | jq
```

- 排队任务可立即取消，返回 `200` 和 `video.deleted`。
- Worker 正在执行时，Gateway 记录取消请求并返回 `202`；继续轮询任务直到进入终态。
- 对 `completed`、`failed`、`cancelled` 或 `expired` 任务执行 DELETE 会清理结果并将任务
  标记为 `expired`，返回 `200`。
- 任务记录仍存在时，重复 DELETE 不会重新创建任务；过期记录最终清理后，再对该 ID
  执行 DELETE 返回 `404 video_not_found`，并非始终返回 200。

如果底层推理引擎不能中途停止 GPU kernel，API 已接受取消并不代表 GPU 计算在同一时刻停止。

## 错误处理与接入

### 错误格式与重试建议

所有可预期 JSON 错误使用统一格式：

```json
{
  "error": {
    "message": "video queue is full",
    "type": "invalid_request_error",
    "param": null,
    "code": "queue_full"
  }
}
```

常见 HTTP 状态：

| 状态 | 常见 code | 调用方处理 |
|---:|---|---|
| 400 | `missing_required_field`、`invalid_size`、`invalid_duration` | 修正参数，不自动重试 |
| 404 | `model_not_found`、`video_not_found` | 检查模型名或任务 ID |
| 409 | `idempotency_conflict`、`video_not_ready` | 使用原请求或继续轮询 |
| 413 | `payload_too_large`、`file_too_large` | 缩小参考素材 |
| 415 | `unsupported_media_type`、`unsupported_media_format` | 使用 multipart 和受支持格式 |
| 422 | `video_generation_failed` | 查看任务 `error`，通常不应原样盲重试 |
| 429 | `queue_full` | 保持相同 `Idempotency-Key`，退避后重试 |
| 503 | `no_worker_available`、`not_ready`、`service_unavailable` | 退避并切换/重试 Gateway |
| 507 | `insufficient_artifact_storage` | 服务端需要释放或扩展制品存储 |

对于连接中断、429、502、503、504 等不确定结果，先使用相同 `Idempotency-Key` 重试提交，
不要生成新 key，否则可能创建重复任务。

### 认证与兼容性

Video Gateway 当前不实现最终用户认证和授权，应由上层应用、API Gateway 或入口 LB
完成身份校验、租户隔离和限流。调用方应忽略未知的响应字段，以便 Gateway 在滚动升级中
增加指标和媒体元数据而不破坏客户端。


## ContextIR 后端并发配置

完整的编译任务接口、字段、调用示例和对接机制见[ContextIR改造与Gateway接口](context-ir/03-customizations-and-gateway-api.md)。

每个 ContextIR 后端的 `max_inflight` 默认值为 `2`，显式配置可以覆盖默认值。
多个模型绑定到同一后端时共享容量；使用同一 etcd 任务域的多个 Gateway 也共享
该后端的执行槽。该参数不改变视频生成 Worker 的执行容量。

```yaml
context_ir:
  enabled: true
  backends:
    h3-ir:
      url: http://context-ir:8420
      max_inflight: 2
  model_bindings:
    minimax-h3: h3-ir
    minimax-h3-ref2va: h3-ir
```

模型名应与实际 Gateway 模型池保持一致。需要提高并发时显式配置 `max_inflight`，
升级代码不会覆盖显式配置。等待队列 `queue_limit` 默认仍为 `32`，
素材预处理并发仍为 `2`。Gateway 的 ContextIR HTTP 连接池总上限为 `64`，
单后端地址上限为 `32`，为编译期间的素材上传等调用保留连接余量。

同一后端的多个 Gateway 应使用一致配置；后端配置修订不一致时会阻止领取任务。
升级应安排无在途 ContextIR 任务的窗口，核对所有 Gateway 的配置一致后，
再验证目标并发下的成功率、排队耗时和资源占用。


## ContextIR调用用量

ContextIR异步任务查询返回LLM风格的顶层 `usage`，其中 `prompt_tokens`、
`completion_tokens`、`total_tokens` 汇总整次任务所有Qwen调用，包含复审、纠错和重试。
尚未获得统计时为 `null`，统计不完整时 `complete=false`；上游不要按不完整值直接结算。
字段、幂等计数和失败任务口径见[ContextIR接口说明](context-ir/03-customizations-and-gateway-api.md#qwen调用用量与上游计费)。
