# Dingo Video Gateway API 协议与兼容性

本文描述当前 DingoRouter Video Gateway 对外提供的视频 API，以及它与 OpenAI Videos API、vLLM-Omni Videos API 的关系。以 2026-09-23 的 DingoRouter 实现为准；实际模型名、可用状态和媒体限制应查询目标部署的 `GET /v1/models`。

## 一句话说明

我们的接口是 **OpenAI 风格的视频任务 API，加上 vLLM-Omni 常用的视频生成参数和 MiniMax-H3 专用适配**。它复用 `/v1/videos` 的提交、查询、下载等资源路径，但不是完整的 OpenAI API，也不是对 vLLM-Omni HTTP 服务的透明代理。

```text
客户端 --multipart /v1/videos--> Dingo Video Gateway
        --持久化任务、排队、租约、重试--> FL2VA / Ref2VA Worker
        <--共享制品目录中的 MP4 与任务结果--
客户端 <--任务状态 / MP4 下载-- Gateway
```

Gateway 将公开请求规范化为 Worker 请求。Worker 执行仍依赖 vLLM-Omni；当前部署通过 Dynamo 内部请求平面和共享制品目录交换任务与结果。双 Gateway 的任务协调使用 etcd。

例如，Gateway 会把 `seconds`/`size` 等公开字段换算为 Worker 使用的帧数和尺寸，再构造包含 `model`、`prompt`、`size`、`output_format: mp4`、`nvext.fps`、`nvext.num_frames`、`nvext.num_inference_steps` 和 `nvext.seed` 的内部请求。参考文件经校验后才编码为底层 `input_reference`；这层转换不属于公开 API。

## 对外接口

| 方法 | 路径 | 当前行为 | 来源与差异 |
|---|---|---|---|
| `GET` | `/v1/models` | 列出模型、`available` 和 `video_capabilities` | OpenAI 风格列表；后两个字段是 Gateway 扩展 |
| `POST` | `/v1/videos` | multipart 异步提交，返回持久化任务 JSON 和 `Location` | 与 OpenAI、vLLM-Omni 的基本路径一致；当前部署配置返回 HTTP 200，代码默认 202 |
| `GET` | `/v1/videos/{id}` | 查询状态、错误、结果信息和阶段耗时 | 同类资源查询；Gateway 增加持久化指标和清理状态 |
| `GET` | `/v1/videos` | 分页查询任务 | 同类资源列表；Gateway 支持按模型、状态等过滤 |
| `GET`/`HEAD` | `/v1/videos/{id}/content` | 下载 MP4；支持 Range、ETag | 基本下载路径相同；HEAD/Range 是 Gateway 提供的能力 |
| `DELETE` | `/v1/videos/{id}` | 活跃任务请求取消；终态任务清理制品 | Gateway 同时承担取消与制品清理；取消不保证底层 GPU 计算立即停止 |
| `POST` | `/v1/videos/sync` | 使用相同任务流程，完成后直接返回 MP4 字节 | 与 vLLM-Omni 的同步扩展同路径；Gateway 同步任务仍可按 ID 查询 |
| `POST` | `/v1/videos/stream` | 返回 `404 unsupported_endpoint` | 当前未实现 |

`/live`、`/ready`、`/health` 和 `/metrics` 是部署与运维接口。当前 Gateway 未配置 `http.upstream_url` 时，其他未列出的路径返回 404；不能把它当作通用 OpenAI Chat/Responses 代理。

## 逐项对照 OpenAI Videos API

这里的“完全一致”严格指**所列出的那一部分协议**一致，不代表整个接口的请求、响应、错误和认证都可以原样替换。按 2026-09-23 的[官方 Videos API 参考](https://developers.openai.com/api/reference/resources/videos)与本仓库 `api.py`、`models.py`、MiniMax-H3 adapter 对照：**目前没有一个完整的 Videos 端点可以声明与 OpenAI 全契约完全一致**。

### 完全一致的部分

| 范围 | 一致之处 | 边界 |
|---|---|---|
| 基础资源路由 | `POST /v1/videos`、`GET /v1/videos`、`GET /v1/videos/{id}`、`GET /v1/videos/{id}/content`、`DELETE /v1/videos/{id}` 的方法与路径相同 | 路径相同不等于入参、返回体或生命周期完全相同 |
| 异步基本流程 | 提交取得 ID，随后按 ID 查询；完成后通过 `/content` 下载 MP4 | 具体提交格式、参数范围、状态和错误码见下文 |
| 任务对象的共同字段 | `id` 为字符串，`object` 为 `"video"`；`model`、`status`、`created_at`、`size` 使用同名字段；`created_at`/已完成时的 `completed_at` 是 Unix 秒时间戳 | `model` 的可用值不同；`status` 的取值集合不同；`size` 的可用值不同 |
| 列表外壳 | 返回 `object: "list"`、`data`、`first_id`、`last_id`、`has_more`；共同支持 `after`、`limit`、`order=asc/desc` | 条目仍是上述有差异的任务对象；`limit` 边界不同 |
| 删除终态任务的成功返回体 | 对已完成/失败任务，成功时返回 `{ "id": "...", "object": "video.deleted", "deleted": true }` | 对活跃任务的 DELETE 是我们的取消扩展；内部清理与保留语义不能据此视为完全一致 |

### 相同端点上的差异与我们的扩展

| 端点/项目 | OpenAI | 当前 Gateway | 分类 |
|---|---|---|---|
| `POST /v1/videos` 的编码 | 官方示例可用 multipart；官方参数参考定义 `prompt` 等字段 | **只接受** `multipart/form-data`；提交后返回任务 JSON 与 `Location` | 共同路径，编码能力不等同 |
| `POST /v1/videos` 的 HTTP 状态 | 返回新建视频任务 | 状态码可配置；当前部署为 **200**，代码默认 **202**；幂等命中返回 200 | 差异 |
| `prompt` | 必填文本，官方长度上限 32000 字符 | 必填文本；按 UTF-8 **字节数**受部署/adapter 限制 | 字段用途相同，校验边界不同 |
| `model` | 可选，默认 Sora 模型；接受官方 Sora 型号 | 只能选已配置 Worker pool 的模型，如 `minimax-h3`、`minimax-h3-ref2va`；是否可省略取决于默认模型/池数 | 字段名相同，取值与默认规则不同；**不支持 Sora 模型** |
| `seconds` | 可省略，默认 `4`；允许字符串 `"4"`、`"8"`、`"12"` | 必须在 `seconds`、`num_frames`、`extra_params.duration` 中恰选一种；对应 4–15 秒，支持如 5 秒 | 字段名相同，必填规则和范围不同 |
| `size` | 可省略，默认 `720x1280`；只允许官方列出的四种尺寸 | 必须提供 `size` 或 `width`+`height`，按 MiniMax-H3 尺寸约束校验 | 字段名相同，必填规则和值域不同 |
| `input_reference` | 官方参数是含 `file_id` **或** `image_url` 的引用对象 | 同名字段只接收一个 **multipart 上传文件**，由 Gateway 保存并校验 | **同名但格式不兼容**；不支持直接传官方 `file_id`/`image_url` 引用对象 |
| `GET /v1/videos/{id}` | 返回官方视频对象 | 返回任务对象，并增加阶段耗时、结果媒体信息、取消/过期信息；不保证官方所有字段 | 共同路径、扩展响应 |
| `GET /v1/videos` | `after`、`limit`（官方允许 0–100）、`order` | 还支持 `model`、`status` 过滤；`limit` 只允许 **1–100** | 扩展筛选，边界差异 |
| `GET /v1/videos/{id}/content` | 默认 MP4；`variant=video/thumbnail/spritesheet` 可选 | 只提供 MP4；额外支持 `HEAD`、单区间 `Range`、`ETag`、`If-None-Match` 和 `X-Video-*` 响应头 | MP4 下载同路径；预览变体不支持，HTTP 下载能力是扩展 |
| `DELETE /v1/videos/{id}` | 删除已完成/失败视频及其资产 | 终态清理返回同形状 JSON；排队或执行中也可请求取消，部分情况返回 202 `video.cancel`；取消不保证 GPU 立即停算 | 终态成功体相同，活跃任务行为是扩展 |
| `GET /v1/models` | 标准模型列表端点 | 列表项额外有 `available`、`video_capabilities`；可能同时列出上游模型，但只有已配置 Worker pool 的模型可提交视频任务 | 扩展，不是 Sora 模型目录 |

特别注意：当前 `/content?variant=thumbnail` 或 `spritesheet` **不会生成对应预览图**；现有 handler 不解析 `variant`，仍按 MP4 下载处理。客户端不能仅以 HTTP 200 判断拿到了所请求的变体。

任务对象本身也要逐字段看，不能只看到 `object: "video"` 就按 OpenAI 类型反序列化：

| 字段 | 对照结论 |
|---|---|
| `id`、`object`、`created_at`、已完成时的 `completed_at` | 字段名和基本类型/时间单位一致；ID 的具体前缀没有兼容保证 |
| `model`、`size` | 字段名及字符串形态一致，可用模型/尺寸集合不同 |
| `status` | 官方的 `queued`、`in_progress`、`completed`、`failed` 均有；我们另有 `cancelled`、`expired` |
| `seconds` | 官方视频对象是**字符串**；我们是**数字**，完成后可能改为测得的实际时长，并另给 `requested_seconds` |
| `progress` | 官方是近似 0–100 的完成百分比；我们仅返回未完成 0、完成 100 |
| `expires_at` | 同名 Unix 秒时间戳；我们按本地任务保留策略设置，不能假定与 OpenAI 保留期限一致 |
| `error` | 官方视频错误对象可含 `headers`、`misalignment`；我们失败时仅返回本地 `code`、`message`、`retryable`，且正常状态通常不返回 `error` 键 |
| `prompt`、`remixed_from_video_id` | 官方视频对象定义了这些字段；当前任务 JSON 不提供 |
| `metrics`、`stage_durations`、`inference_time_s`、`num_frames`、`fps`、`seed`、`bytes`、`sha256` 等 | 我们的附加字段，不是 OpenAI Videos 对象的必备/标准字段 |

### 我们新增、但不是 OpenAI Videos API 的能力

| 能力 | 当前接口/字段 |
|---|---|
| 同步返回 MP4 | `POST /v1/videos/sync`；沿用持久化任务流程，超时后还可凭 `X-Video-Id` 查询 |
| 幂等提交 | 请求头 `Idempotency-Key`；任务与幂等记录保留、pool 配置及适配器兼容版本不变时，相同请求复用任务；同 key 的请求摘要不一致返回 409；记录清理后同 key 可能创建新任务 |
| 多参考与 MiniMax-H3 生成控制 | 重复的文件 part `input_references`，以及 `width`、`height`、`num_frames`、`fps`、`num_inference_steps`、`seed`、`generate_sound`、`frame_indices` 等；按模型/pool 校验，不是任意透传 |
| 更多任务状态和观测字段 | `cancelled`、`expired`；`metrics`、`stage_durations`、`num_frames`、`fps`、`seed`、`bytes`、`sha256`、`duration_s` 等；`inference_time_s` 也不是 OpenAI Videos 标准字段 |
| 下载与运维 | `HEAD`、Range/ETag/条件下载；`/live`、`/ready`、`/health`、`/metrics` |

`user` 虽可作为我们请求中的标签，但不是上述 OpenAI **Videos Create** 参考列出的参数，不负责认证或租户隔离。`response_format`、`output_format` 等字段源于 vLLM-Omni/内部适配，也不属于 OpenAI Videos Create 的公开参数。

### OpenAI 有、当前 Gateway 不支持或不等价的部分

| 官方能力 | 当前情况 |
|---|---|
| `POST /v1/videos/{id}/remix` | 无本地实现 |
| `POST /v1/videos/edits` | 无本地实现 |
| `POST /v1/videos/extensions` | 无本地实现 |
| `POST /v1/videos/characters` | 无本地实现 |
| `GET /v1/videos/characters/{id}` | 无本地实现 |
| Sora 型号、官方尺寸/时长默认值与取值集合 | 不支持；只能按本部署模型和约束提交 |
| 官方 `input_reference.file_id` / `image_url` | 不支持；需上传媒体文件 part |
| 下载 `variant=thumbnail` / `spritesheet` | 不生成；只能获得 MP4 |
| 官方视频对象的 `prompt`、`remixed_from_video_id`，及官方错误对象的 `headers`/`misalignment` 等完整契约 | 当前任务查询不提供这些官方字段/结构；失败任务的 `error` 使用 Gateway 自身格式 |
| OpenAI 平台 Bearer API key、项目权限与隔离 | Gateway 自身未实现；必须由上层 LB/API 网关或应用负责鉴权和租户边界 |

此表按**当前未配置 `http.upstream_url` 的部署**说明。若将来配置通用 upstream，未注册路由可能由上游处理；这并不意味着 Gateway 自己实现了这些 OpenAI 能力，也不能把上述“不支持”理解为对任意上游配置的断言。OpenAI 的 Chat、Responses、Files 等其他 API 也不属于本视频 Gateway 的本地实现范围。

## Dingo 协议细节与 vLLM-Omni 适配

### 请求格式与 MiniMax-H3 适配

提交接口只接受 `multipart/form-data`。普通字段是文本 part，媒体输入是文件 part。业务调用方通常只需提供 `model`、`prompt`、时长、尺寸、推理步数和参考文件；`task`、`flow_shift` 等 Worker 参数由 Gateway 根据 pool 填充。

| 字段 | 与 OpenAI / vLLM-Omni 的关系 | Gateway 当前行为 |
|---|---|---|
| `model`、`prompt`、`seconds`、`size` | 与 OpenAI Videos Create 同名，vLLM-Omni 也支持 | 模型必须属于已配置 pool；时长和尺寸按 MiniMax-H3 限制校验；必填/默认值/取值并不与 OpenAI 相同 |
| `user` | 不是当前 OpenAI Videos Create 参考列出的字段 | 仅作标签，不用于认证 |
| `width` + `height`、`num_frames` | vLLM-Omni 扩展 | 可替代 `size`、`seconds`；时长必须在 `seconds`、`num_frames`、`extra_params.duration` 中恰选一种 |
| `fps`、`num_inference_steps`、`seed`、`guidance_scale`、`negative_prompt` | vLLM-Omni 常见生成参数 | MiniMax-H3 当前只接受 24 FPS；步数允许 2–200；seed 可省略并由 Gateway 生成 |
| `generate_sound` | vLLM-Omni 扩展 | Gateway 默认 `true`；请求 `false` 时，输出不得带音轨，必要时 Gateway 去音轨 |
| `input_reference` | 与 OpenAI 同名，但官方要求 `file_id`/`image_url` 引用对象 | 当前只接收一个 multipart 上传文件；FL2VA 可作为首帧或尾帧；不能直接使用官方引用对象 |
| 重复的 `input_references` | Gateway 的多参考文件扩展 | FL2VA 支持两张首尾帧；Ref2VA 支持图片、视频及可选音频组合，文件顺序保留 |
| `frame_indices`、`extra_params` | 为 MiniMax-H3 与 vLLM-Omni 参数映射保留 | 只允许实现明确支持的键；不是任意扩展 JSON 的透传 |
| `response_format`、`output_format` | vLLM-Omni 输出参数 | 仅接受内部 `b64_json` 和 `mp4`；对外结果通过下载接口或同步响应返回 MP4，不在任务 JSON 中返回 Base64 视频 |

当前对外模型名通常是：

- `minimax-h3`：FL2VA，文本生成视频，或使用一张/两张关键帧图片。
- `minimax-h3-ref2va`：Ref2VA，至少一张图片或一个视频，可附带音频等参考素材。

Ref2VA 的多媒体组合会被适配器转换成底层 `input_reference` 可理解的编码；Gateway 不要求客户端自行构造该内部封装。具体格式、大小、时长和示例见[接口使用说明](video-gateway-api.md)。

目前不接受 `quality`、`image_reference`、`video_reference`、`audio_reference` 等直传字段，也未开放 vLLM-Omni 最新文档中的 `control_reference`、MiniMax-H3 latent-mask 编辑等能力。不能因为底层 vLLM-Omni 支持某字段，就假定 Gateway 已对外开放。

#### 最小调用示例

```bash
RESPONSE=$(curl -sS -X POST "$GATEWAY_URL/v1/videos" \
  -H 'Idempotency-Key: order-123' \
  -F model=minimax-h3 \
  -F 'prompt=A sunrise over snow mountains' \
  -F seconds=5 \
  -F size=1344x768 \
  -F num_inference_steps=50)
TASK_ID=$(printf '%s' "$RESPONSE" | jq -r '.id')

curl -sS "$GATEWAY_URL/v1/videos/$TASK_ID"
curl -fL "$GATEWAY_URL/v1/videos/$TASK_ID/content" -o "$TASK_ID.mp4"
```

`Idempotency-Key` 是 Gateway 扩展：在原任务及幂等记录仍保留、pool 配置及适配器兼容
版本不变时，同一 key、同一请求重复提交会返回原任务；请求摘要不一致则返回 409。
摘要包含这些版本信息，因此滚动升级期间即使 HTTP 请求相同也可能冲突；记录清理后
同 key 可能创建新任务。
发生网络超时时，仍应先复用原 key 重试，遇到 409 时查询已知任务 ID 或核对配置。

### 响应和任务生命周期

异步提交先返回任务 ID 与 `object: "video"`。公开状态为 `queued`、`in_progress`、`completed`、`failed`、`cancelled`、`expired`。内部 `dispatching` 和 `finalizing` 都映射为 `in_progress`；`in_progress` 也可能表示 Worker 内预取等待。当前 `progress` 只有完成时 100、其他状态 0，不提供逐去噪步进度。

完成后，`GET /v1/videos/{id}` 提供结果字节数、SHA256、实际视频时长以及 `metrics` / `stage_durations`。其中 `queue_wait_s`、`worker_queue_wait_s`、`inference_time_s`、`finalize_time_s` 用于区分 Gateway 排队、Worker 预取、推理与 Gateway 结果处理。已受理任务失败时查看任务的 `error.code`，例如 `worker_failed` 或 `finalization_timeout`；没有可用 Worker 时，提交请求可直接收到 HTTP 503 `no_worker_available`。

这里也有类型和语义差异：OpenAI 视频对象的 `seconds` 是字符串，当前 Gateway 返回数值型时长；OpenAI 文档中的 `progress` 表示近似完成百分比，而当前 Gateway 仅使用 0/100。客户端不应依赖官方 SDK 对视频对象的全部类型约束与可选字段都完全一致。

任务与结果会过期，也可通过 DELETE 主动清理。调用方应保存任务 ID，完成后及时下载；不能把制品目录当成永久对象存储。

### 与 vLLM-Omni 的边界

| 主题 | vLLM-Omni Videos API | 当前 Dingo Video Gateway |
|---|---|---|
| 服务定位 | 单模型推理服务的视频 HTTP 接口 | 多 Gateway、多 Worker pool 的持久化调度与制品服务；不是到 Worker HTTP 接口的透明代理 |
| 同步生成 | 提供 `/v1/videos/sync` | 同路径，但同步任务也持久化，超时后可按 ID 查询 |
| 参考素材 | 提供单参考与其他模型扩展字段 | 接收单文件 `input_reference` 或多文件 `input_references`，按 FL2VA/Ref2VA 限制校验，再转换为 Worker 输入 |
| 模型参数 | 可依模型支持更多视频生成和编辑字段 | 只开放当前 MiniMax-H3 适配器明确支持的字段；不直接透传任意 Worker 参数 |
| 任务与结果 | 由单个推理服务管理 | etcd 协调跨 Gateway 任务、Worker 租约与重试；共享制品存储支持跨 Gateway 查询和下载 |
| 认证 | 由具体部署决定 | Gateway 自身不做最终用户认证与租户鉴权，需由上层入口处理 |

因此，兼容性的准确表述是：**可复用 OpenAI 风格的视频资源路径和 vLLM-Omni 的一部分请求字段；不能保证任意 OpenAI SDK 调用或任意 vLLM-Omni 模型扩展只替换 `base_url` 就能工作。** 调用方应以本文和[接口使用说明](video-gateway-api.md)为契约。

## 依据与版本说明

- [OpenAI Videos API 参考](https://developers.openai.com/api/reference/resources/videos)：用于对照公开资源路径和任务对象；该页面在本文写作时将相关 Videos 接口标为 Deprecated，不能据此推断本项目接口也已废弃。
- [vLLM-Omni Videos API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/videos_api/)：用于对照最新公开协议；`latest` 文档可能超前于本部署所用的 vLLM-Omni 0.29 系列镜像。
- 本仓库 `dingo/video_gateway/api.py`、`adapters/minimax_h3.py`、`models.py` 和 `docs/video-gateway/video-gateway-api.md`：决定当前 Gateway 的实际行为。
