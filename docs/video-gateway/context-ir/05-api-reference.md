---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "ContextIR API：参数、返回值与官方对照"
subtitle: "提示词编译与视频生成分开，画布只供内部校验"
---

更新：2026-11-18。本文为当前Gateway ContextIR接口依据。ContextIR解释素材和意图并生成H3提示词；分辨率、超分方法、GPU容量及视频文件由视频生成阶段决定。示例地址和任务ID为占位值。

## 接口

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/v1/context-ir` | multipart 提交。新任务返回 202，幂等重放返回 200 |
| GET | `/v1/context-ir/{id}` | 查询状态和编译结果 |
| DELETE | `/v1/context-ir/{id}` | 取消活跃任务时返回 200 和 `cancelled` 任务对象。删除已结束结果时返回 204 |
| GET | `/v1/context-ir/{id}/diagnostics` | 查询耗时、用量和提交时的素材信息。推断后的最终角色见任务查询的 `references` |

生成视频仍需调用`POST /v1/videos`，并提交原参考素材、编译后的prompt及调用方选择的分辨率等生成参数。

## 最小调用

纯文本编译只需提供模型、原始提示词、时长和比例。创建响应中的 `id` 用于查询同一个任务。`Location` 响应头也提供查询路径。

```bash
export GATEWAY_URL='https://gateway.example.com'
curl -sS "${GATEWAY_URL}/v1/context-ir" \
  -H 'Idempotency-Key: compile-example-001' \
  -F 'model=minimax-h3' \
  --form-string 'prompt=一只猫坐在窗边，轻轻转头，没有对白。' \
  -F 'seconds=5' -F 'ratio=16:9'
```

将创建响应的 `id` 赋给 `TASK_ID`，然后轮询同一个地址。

```bash
curl -sS "${GATEWAY_URL}/v1/context-ir/${TASK_ID}"
```

状态为 `completed` 时，检查 `warnings`，再将 `content.prompt` 和 `generation_parameters` 用于视频生成。生成请求需要另行提供 `resolution` 或使用显式尺寸。指定显式尺寸时，按视频 API 校验其与 `ratio` 的匹配情况。带素材的任务需要在生成时按原上传顺序重新提交素材。

## 请求参数

`POST /v1/context-ir` 使用 `multipart/form-data`。普通参数各传一个文本字段。JSON 数组也放在一个文本字段里，上传多份素材时重复使用文件字段。枚举值按文档中的大小写传递，`silent` 的布尔文本除外。

### 模型、时长和比例

| 参数 | 类型 | 必填与默认值 | 取值和约束 |
|---|---|---|---|
| `model` | 字符串 | 必填，无默认值 | Gateway 已配置 ContextIR 绑定的公开视频模型名。常见配置为 `minimax-h3`（T2VA/FL2VA）和 `minimax-h3-ref2va`（Ref2VA），名称以部署为准 |
| `prompt` | 字符串 | 必填，无默认值 | 原始提示词，去掉首尾空白后须非空。默认上限 32 KiB，按 UTF-8 字节计 |
| `seconds` | 数值 | 与 `num_frames` 二选一，无默认值 | 4–15 秒（含边界），支持小数，例如 `5.5` |
| `num_frames` | 整数 | 与 `seconds` 二选一，无默认值 | 96–360 请求帧（含边界），按 24 fps 换算时长 |
| `ratio` | 字符串 | 取决于场景，见下表 | `16:9`、`21:9`、`4:3`、`1:1`、`3:4`、`9:16`、`adaptive` |

如果传 `seconds`，Gateway 先将秒数乘以 24 并取整为请求帧数。编译结果中的 `target.frames` 在此基础上向上对齐到最近的 `17n+5`。例如 `seconds=5` 和 `num_frames=120` 都会得到 124 帧，`num_frames=360` 会得到 362 帧。96–360 是输入范围。

### 画幅与分辨率

以下规则适用于省略显式尺寸的请求。

| 场景 | 素材 | `ratio` 的写法 |
|---|---|---|
| T2VA | 无 | 必须选六种具体比例之一 |
| FL2VA | 一张首帧或尾帧，或两张首尾帧 | 可省略或写 `adaptive`。传入合法的具体比例时，仍按最终首帧或唯一尾帧确定画幅 |
| Ref2VA | 至少一张图片或一段视频，可附加独立音频 | 可指定具体比例，省略或传 `adaptive` 时优先参考首个视频，其次首张图片，并选择最接近的基础画布比例 |

### 素材上传

上传一份或多份素材时使用 `input_references`，每份文件重复同名字段。单文件也可使用 `input_reference`，两种字段互斥。按实际上传顺序填写 `reference_options`。

| 类型 | 支持格式 | 默认单文件上限 | 其他约束 |
|---|---|---:|---|
| 图片 | JPEG、PNG、WebP | 30 MiB | 宽高各 256–5760 像素，宽高比 0.4–2.5 |
| 视频 | MP4、MOV，H.264 或 H.265，内嵌音轨使用 AAC | 50 MiB | 单段 2–15 秒，约 23.976–60 fps，宽高与宽高比同图片 |
| 独立音频 | PCM WAV | 15 MiB | 单段 2–15 秒 |

FL2VA 最多两张图片。Ref2VA 最多 9 张图片、3 段视频和 3 段独立音频，全部文件合计最多 12 个。参考视频总时长、独立音频总时长分别最多 15 秒。部署配置可能进一步限制文件和请求大小。

单张首帧的编译请求如下。比例由图片决定。

```bash
curl -sS "${GATEWAY_URL}/v1/context-ir" \
  -F 'model=minimax-h3' --form-string 'prompt=让首帧人物自然微笑。' \
  -F 'seconds=5' -F 'input_references=@first-frame.png'
```

### 素材选项 `reference_options`

`reference_options` 可省略，默认是空数组 `[]`。提供时，将 JSON 数组作为一个 multipart 文本字段传入。按 `input_references` 文件在请求中的上传顺序，从 0 开始给所有素材编号。图片、视频、音频共用一套编号。数组可以只列需要配置的素材，其他素材继续使用推断或默认设置。

| 对象字段 | 类型 | 含义 |
|---|---|---|
| `index` | 必填整数 | 对应素材的上传序号。范围为 0 到文件总数减 1，每个序号最多出现一次 |
| `role` | 可选字符串 | 指定素材用途，取值取决于工作流和素材类型 |
| `note` | 可选字符串 | 帮助编译器理解素材的说明，例如“只参考灰色外套的样式”。UTF-8 编码最多 4096 字节 |
| `soundtrack_policy` | 可选字符串 | 仅用于 Ref2VA 的参考视频。`preserve` 表达保留参考视频原音轨的意图，`ignore` 表达忽略原音轨的意图 |

| 工作流 | 素材 | `role` 可选值 | 省略时的默认用途 |
|---|---|---|---|
| FL2VA | 图片 | `frame_anchor_first` 用作首帧，`frame_anchor_last` 用作尾帧 | 单图为首帧。双图按上传顺序为首帧、尾帧 |
| Ref2VA | 图片 | `subject` 参考人物或物体，`environment` 参考场景，`style` 参考画面风格 | `subject` |
| Ref2VA | 视频 | `edit_source` 将视频作为编辑底稿，`motion_reference` 参考视频中的动作或运动 | `edit_source` |
| Ref2VA | 独立音频 | `voice_timbre` 参考说话音色，`bgm` 用作背景音乐 | `voice_timbre` |

首帧和尾帧各最多指定一张。上传两张图时，可以只指定一张的 `role`，另一张自动补齐。只上传尾帧时，使用 `[{"index":0,"role":"frame_anchor_last"}]`。

对于支持从提示词推断的用途，优先级为**显式参数 > 提示词中可确认的用途 > 默认值**。省略选项时，编译器可从提示词推断 FL2VA 的首尾帧、Ref2VA 独立音频的用途，以及含可解码音轨的参考视频的音轨策略，例如“图 1 作为尾帧”“从图 2 开始，以图 1 结束”或“音频 1 作为背景音乐”。参考视频有音轨且未明确指定处理方式时，默认使用 `preserve`。该选项控制编译提示词中的音轨用途，成片是否保留原声取决于后续生成。

`intent_resolution` 在素材分析和提示词编写之前执行。无法确定用途、调用超时或返回格式无效时使用默认值，并在 `warnings` 中说明。该阶段依据文本判断用途，不做语音识别或听辨。

缺少 `index`、序号重复或越界、对象包含未知字段、`role` 与素材类型不匹配、音轨策略字符串不在枚举中或首尾帧角色重复时，接口返回 HTTP 400。可省略无需配置的数组元素。

`role` 和 `soundtrack_policy` 应使用字符串。当前版本在部分场景会对数组或对象类型返回 HTTP 500。

下面的 Ref2VA 请求按顺序上传人物图、动作参考视频和音色参考音频。视频原声标记为 `ignore`，对白音色由独立音频提供。

```bash
curl -sS "${GATEWAY_URL}/v1/context-ir" \
  -F 'model=minimax-h3-ref2va' \
  --form-string 'prompt=让图1的人物参考视频1的动作，并用音频1的音色说：“我们出发吧。”' \
  -F 'seconds=5' -F 'ratio=16:9' \
  -F 'input_references=@character.png' \
  -F 'input_references=@motion.mp4' \
  -F 'input_references=@voice.wav' \
  --form-string 'reference_options=[{"index":0,"role":"subject","note":"保留人物的发型和衣着"},{"index":1,"role":"motion_reference","soundtrack_policy":"ignore"},{"index":2,"role":"voice_timbre"}]'
```

反向首尾编译返回`generation_parameters.frame_indices=[-1,0]`，表示原上传图1是尾帧、图2是首帧。`references.index`始终对应原上传顺序；`labels`会反映H3实际首尾顺序。调用方使用原素材加返回参数提交视频API即可，自适应比例取最终首帧。仅尾帧返回`[-1]`。

直接调用`POST /v1/videos`不会额外执行本阶段，也不会从自然语言自动推断首尾；需要这种能力时先编译，再复用返回的frame_indices。反向两图当前经Gateway整理为原生Worker的首尾顺序，使用受原有内联消息限额约束的路径；普通正向共享文件路径不变。

### 编译控制

| 参数 | 类型 | 默认值 | 取值和约束 |
|---|---|---|---|
| `context_ir_seed` | 整数 | `7` | 0 到 2⁶³−1（含边界）。控制编译采样，视频生成的 `seed` 单独传给视频 API |
| `creativity` | 字符串 | `balanced` | `restrained`、`balanced`、`bold`、`extreme` |
| `dialogue` | JSON 数组 | `[]` | 每项提供非空的 `text`，可附加字符串 `language` 和 `speaker` |
| `silent` | 布尔文本 | `false` | `true` 或 `false`，不区分大小写。`true` 禁用编译中的背景音乐，仍允许对白和环境音 |

`creativity` 控制原提示词未指定的内容可以补充到什么程度。提示词明确给出的要求继续生效。

| 档位 | 编译器允许补充的内容 |
|---|---|
| `restrained` | 保守改写，避免自行新增配乐、对白或画面文字 |
| `balanced` | 可补充背景配乐 |
| `bold` | 可补充配乐、对白或画面文字 |
| `extreme` | 补充内容的范围与 `bold` 相同，镜头运动和构图等表达倾向更强 |

`dialogue` 用于明确台词原文、语言和说话人。例如将以下 JSON 作为一个 multipart 文本字段传入。省略 `language` 时，编译器可根据台词判断中英文。省略 `speaker` 时，编译器结合提示词处理说话人。建议在原提示词里也明确谁说哪一句。

```json
[
  {"text": "我们出发吧。", "language": "Chinese", "speaker": "图1中的人物"}
]
```

### 兼容尺寸参数

| 参数 | 类型 | 默认值 | 用法 |
|---|---|---|---|
| `aspect_ratio` | 字符串 | 省略 | `ratio` 的别名，支持相同取值。两者都传时必须一致 |
| `resolution` | 字符串 | 省略 | 接受 `768P`、`1080P`、`2K`，仅校验枚举值。输出分辨率在调用视频 API 时另行指定 |
| `size` | 字符串 | 省略 | 格式为 `WIDTHxHEIGHT`，例如 `1344x768` |
| `width`、`height` | 整数 | 省略 | 成对提供，与 `size` 互斥 |

已知的 768P、1080P、2K 画布尺寸可以用于兼容编译。其他尺寸的宽高各须在 256–2048 像素内，均为 32 的倍数，总像素数不超过 1344×768，宽高比与最接近的支持比例相差不超过 5%。同时传入具体比例时，须与尺寸匹配。对于 T2VA，合法显式尺寸可以替代比例。编译返回的 `generation_parameters` 提供时长或请求帧数、比例、fps 和适用时的首尾帧映射。

### 请求头与常见校验失败

| 请求头 | 必填与默认值 | 用法 |
|---|---|---|
| `Content-Type` | 必填 | 使用 `multipart/form-data`，由客户端生成 boundary，例如使用 curl 的 `-F` |
| `Idempotency-Key` | 可选，无默认值 | 1–256 字节。同一业务提交重试复用原键和原请求。同键、同请求重放返回 200，同键但请求发生变化返回 409 |

| HTTP 状态 | 常见情形 |
|---|---|
| 400 | 缺少必填项、两种时长同时传、重复文本字段、JSON 格式错误、非法枚举、素材组合或规格不合规 |
| 404 | 模型或任务 ID 未找到 |
| 408 | 上传未在当前部署的时限内完成 |
| 409 | 幂等键与已提交请求冲突 |
| 410 | 查询的任务已过期 |
| 413 | 上传超过大小限制 |
| 415 | 请求未使用 multipart，或素材格式和编码不受支持 |
| 429 | 编译队列已满 |
| 503 | 编译接口未启用、后端暂不可接单或存储预算不足 |

`duration`、`extra_params`、`fps`、`seed`、`num_inference_steps` 和 `generate_sound` 不在此编译接口的字段列表内。官方 JSON `content[]` 格式及 `callback_url` 也不适用。

## 返回字段

| 字段 | 含义 | 上层是否需要 |
|---|---|---|
| `id`、`status` | `context-ir-<ULID>` 和任务状态 `queued`、`in_progress`、`completed`、`failed`、`cancelled`。过期查询返回 410 | 需要 |
| `content.prompt` | completed后提供编译提示词 | 需要 |
| `usage` | 聚合实际Qwen调用的用量，包含意图解析、视觉分析、编写及纠错 | 计费需要 |
| `warnings` | 编译警告；合法降级草稿以context_ir_degraded_result标记 | 建议检查 |
| `generation_parameters` | `ratio`、`seconds` 或 `num_frames`、`fps`，有关键帧时还包含 `frame_indices`。FL2VA 关键帧任务返回 `ratio: adaptive` | 可复用。生成时另外指定分辨率或显式尺寸 |
| `references` | 输入index、kind、SHA256、模型标签和最终role | 联调和素材对应时使用 |
| `target` | 对齐后的frames和fps；不含canvas | 辅助信息 |
| `phase`、`metrics` | 内部阶段、排队和执行等耗时 | 诊断使用 |
| `model`、`object`、各时间戳 | 任务元数据 | 按需 |
| `error` | 失败任务的code/message/retryable | 失败时需要 |

成功响应节选如下，usage数字为示意：

```json
{
  "id": "context-ir-<id>",
  "object": "context_ir",
  "model": "minimax-h3",
  "status": "completed",
  "content": {"prompt": "<compiled H3 prompt>"},
  "generation_parameters": {"ratio": "16:9", "seconds": 5, "fps": 24},
  "target": {"frames": 124, "fps": 24},
  "references": [],
  "warnings": [],
  "usage": {
    "prompt_tokens": 4200,
    "completion_tokens": 300,
    "total_tokens": 4500,
    "prompt_tokens_details": {"cached_tokens": 3200},
    "llm_requests": 1,
    "requests_with_usage": 1,
    "requests_without_usage": 0,
    "complete": true
  }
}
```

`metrics.intent_resolution_s`记录该独立阶段耗时；public phase仍表示任务总体状态，不保证实时显示细分阶段。阶段决策依据随内部诊断证据保存，不新增顶层返回字段。没有待判断项或命中持久阶段结果时，不新增本阶段模型调用。

`usage` 汇总编译过程中所有模型调用。上游报告的视觉 token 计入 `prompt_tokens`。`complete: false` 表示有模型调用缺少用量回执。`prompt_tokens_details.cached_tokens` 来自上游缓存统计，缺少完整计数时可为 `null`。旧后端可能省略 `prompt_tokens_details`。进行中、未调用模型或缺少有效用量对象时，`usage` 可为 `null`。

公开结果对历史存储结果也隐藏像素字段；原始内部结果及摘要不重写，已受理任务与幂等记录保持可用。completed且带降级警告时，content.prompt是实际回退草稿，调用方可根据warnings决定是否使用。

## 与MiniMax官方ContextIR对比

官方依据为[创建ContextIR任务](https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir)及[查询任务](https://platform.minimax.io/docs/api-reference/video-generation-v2-query)，核对日期 2026-11-18。

| 方面 | 官方 | 我们 |
|---|---|---|
| 输入协议 | JSON content[]，使用媒体URL等引用 | multipart，直接上传素材 |
| 提交 | `/v2/h3_context_ir` 返回 200 和 `task_id` | `/v1/context-ir` 新任务返回 202 和含 `id` 的任务对象 |
| 查询结果 | 共用任务查询，示例外层task | 独立编译任务查询，平铺任务对象 |
| 工作流 | 同一MiniMax-H3模型，通过素材role表达 | 公共model路由到FL2VA或Ref2VA池，再按素材确定具体模式 |
| 时长 | duration，4–15整数秒 | seconds；兼容浮点及num_frames |
| 比例 | ratio；首尾帧自适应，Ref2VA允许自适应 | 同类输入语义；Ref2VA自动选幅算法为本地策略 |
| 像素尺寸 | ContextIR未定义resolution/size等输入；成功示例无canvas | 不返回像素尺寸；旧尺寸字段仅兼容接收 |
| 用量 | 示例提供prompt/completion/total_tokens | 相同基础计数，增加聚合调用数、缓存与完整性信息 |
| 可选控制 | 支持callback_url | 不支持回调；提供编译seed、creativity、dialogue、silent等高级字段 |
| 诊断与降级 | 按官方任务协议 | 现有warnings、metrics及diagnostics；合法草稿也可completed |

接口语义可以对齐，质量与完整算法不能据此宣称等同。官方ContextIR为闭源实现；我们基于open-h3-ir与Qwen，已验证样本不代表所有输入都具有相同输出。

## ContextIR监控指标

HTTP与Worker两种接入均从Gateway的`/metrics`暴露下列指标：

| 指标 | 标签 | 含义 |
|---|---|---|
| `dingo_context_ir_submitted_total` | `backend,model` | 新建编译任务，幂等重放不增加 |
| `dingo_context_ir_terminal_total` | `backend,model,state` | 完成、失败、取消及过期事件；过期是保留期事件，不是第二次编译失败 |
| `dingo_context_ir_stage_duration_seconds` | `backend,model,stage` | 编译阶段耗时直方图，秒单位 |
| `dingo_context_ir_queued` | `backend` | Gateway排队任务数 |
| `dingo_context_ir_inflight` | `backend` | 已占用执行槽的任务数；取消后可能等待实际计算结清 |
| `dingo_context_ir_reserved_bytes` | `backend` | 编译任务预留存储字节数 |
| `dingo_context_ir_quarantined` | `backend` | 后端隔离状态，0或1 |
| `dingo_context_ir_max_inflight` | `backend` | 后端兼容配置值；Worker模式实际容量由已发现Worker的物理槽决定 |

`stage`白名单为`prepare_time_s`、`asset_upload_time_s`、`compile_time_s`、
`intent_resolution_s`、`finalize_time_s`、`queue_wait_s`、`execution_time_s`。
仅记录该接入路径实际测得的值；Worker共享文件模式没有HTTP素材上传阶段，
未启用意图解析时也没有`intent_resolution_s`，均不补零。
Worker模式的`compile_time_s`包含编译调用与结果校验；`execution_time_s`包含领取后
到结果结清的时间，二者有包含关系。它们同样保存在任务API的`metrics`中。

统计在业务状态变更确认后执行，重复状态查询、CAS冲突以及取消后的物理结清
不重复计数；结果未知时不提前记录失败。过期事件不重复产生编译耗时样本。
Prometheus Counter/Histogram在Gateway进程重启后重置，不回填历史任务。
Qwen自身的token、吞吐和推理引擎指标属于Qwen服务的`/metrics`，不由以上编译指标代替。

## 验证记录

本轮素材意图解析验证：399项Gateway/Worker相关本地回归、1294项编译器测试及23项格式控制通过；真实Qwen分类22/22、Gateway编译36/36、负向接口8/8通过。原有14项Ref2VA编译均未降级；一个刻意构造的“显式音色参数与背景音乐提示词矛盾”请求返回合法降级草稿，角色仍遵循显式参数。技术完成不等同于音色、配乐或首尾画面质量已获人工认可。
