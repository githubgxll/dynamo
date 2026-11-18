---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: MiniMax-H3 独立超分 API
subtitle: 源视频限制、任务响应和调用示例
---

独立超分将一条已有视频升级到 1080P 或 2K。使用现有 Video Gateway 的地址和鉴权方式提交文件，拿到任务 ID 后异步查询，完成后下载 MP4。

## 提交任务

`POST /v1/videos/regenerate` 使用 `multipart/form-data`。以下三项均必填，每项只传一次。

| 字段 | 类型 | 取值 |
|---|---|---|
| `model` | 字符串 | 本次部署使用 `minimax-h3`。源视频来自纯文本、首尾帧或参考生成时均使用这个值 |
| `resolution` | 字符串 | `1080P` 或 `2K`，区分大小写 |
| `base_video` | 文件 | 上传恰好一个 MP4 文件，可使用 `video/mp4` MIME 类型 |

`base_video` 只接受表单文件上传，URL、服务器路径和 Base64 文本会被拒绝。`operation`、`prompt`、`seed`、`seconds`、`fps`、`size`、`ratio`、`width`、`height`、`input_reference` 和 `input_references` 均不属于本接口。

旧客户端如通过 `POST /v1/videos` 或 `POST /v1/videos/sync` 提交独立超分，请迁移到
`POST /v1/videos/regenerate`。继续用原任务 ID 查询、下载或取消已创建的任务。

`Idempotency-Key` 请求头可选，建议由调用方为一次业务提交生成唯一值。键长度为 1–256 字节。在同一 Gateway 任务存储及相同路由配置下，重试时复用原键、模型、档位与文件内容，可返回原任务 ID。同键对应不同请求时返回 `409 idempotency_conflict`。提交超时或断连后可用原键重试。

```bash
export GATEWAY_URL=https://your-video-gateway.example.com
curl -fsS "$GATEWAY_URL/v1/videos/regenerate" \
  -H 'Idempotency-Key: regenerate-example-001' \
  -F 'model=minimax-h3' \
  -F 'resolution=1080P' \
  -F 'base_video=@source-768p.mp4;type=video/mp4'
```

按部署要求添加鉴权请求头。客户端使用 `-F` 或 multipart 库时，由客户端生成带 boundary 的 Content-Type。

## 源视频限制

| 检查项 | 要求 |
|---|---|
| 容器和视频轨 | MP4，恰好一条 H.264 视频轨，像素格式为 `yuv420p`，全片画布固定 |
| 音轨 | 恰好一条 AAC 音轨 |
| 帧率和时间戳 | 恒定 24 FPS，首帧时间戳为 0，各帧时间间隔为 1/24 秒 |
| 帧数 | 107 至 362 帧，以 107 为起点每隔 17 帧一档，例如 107、124、141、…、345、362 |
| 时长 | 由帧数和帧率决定，最短约 4.46 秒，最长约 15.08 秒 |
| 宽高和面积 | 宽、高各为 256–2048 像素，均是 32 的倍数。宽高比为 0.4–2.5，每帧像素总数为 589,824–1,032,192 |
| 画幅 | 16:9、21:9、4:3、1:1、3:4、9:16。优先识别 H3 768P 预设及已知 SR 基础画布。其余尺寸与最近的受支持比例的相对误差最多为 3% |
| 文件大小 | 测试配置为 50 MiB（52,428,800 字节）。调用前可从 `GET /v1/models` 的 `video_capabilities.regeneration.max_video_bytes` 获取当前部署的上限 |

可以优先使用 H3 生成的常规 768P 成片。任意比例的自适应 768P 视频仍需通过上表校验。

## 输出尺寸

`resolution` 选择输出档位。Gateway 根据源画幅查表，任务响应的 `size` 是实际交付尺寸。尺寸按宽 × 高列出。

| 源画幅 | 1080P | 2K |
|---|---|---|
| 16:9 | 1920×1080 | 2560×1440 |
| 21:9 | 2208×936 | 2944×1248 |
| 4:3 | 1440×1080 | 1920×1440 |
| 1:1 | 1080×1080 | 1440×1440 |
| 3:4 | 1080×1440 | 1440×1920 |
| 9:16 | 1080×1920 | 1440×2560 |

输出保留源帧数、帧率和 AAC 音轨。音轨以流复制方式保留音频包内容、时间戳和包时长。输出上限见能力接口的 `max_result_bytes`，测试配置为 128 MiB。

## 任务响应和下载

提交成功返回任务对象，并通过 `Location: /v1/videos/{id}` 提供查询路径。新任务默认返回 HTTP 202。部署可通过 `http.async_submit_status_code` 配置为 200。幂等重放返回 200。创建响应表示已受理，后续按 `status` 判断结果。

以下是一个 21:9 视频提交到 1080P 后的精简响应示例：

```json
{
  "id": "video-01M00000000000000000000000",
  "object": "video",
  "model": "minimax-h3",
  "operation": "sr_only",
  "task": null,
  "input_usage": {
    "image_count": 0,
    "video_duration_seconds": null
  },
  "status": "queued",
  "progress": 0,
  "size": "2208x936",
  "resolution": "1080P",
  "ratio": "21:9",
  "num_frames": 107,
  "fps": 24,
  "seconds": 4.458333333333333,
  "created_at": 1794975710,
  "expires_at": 1794997310
}
```

| 字段 | 含义 |
|---|---|
| `id` | 后续查询、下载、取消用的 `video-<ULID>` |
| `object`、`model` | 资源类型 `video` 和请求中的公开模型 |
| `operation`、`task` | 独立重生成返回 `sr_only`、`null` |
| `input_usage` | 沿用视频任务结构。独立重生成中 `image_count` 为 `0`，`video_duration_seconds` 为 `null` |
| `resolution`、`size`、`ratio` | 输出档位、实际像素尺寸、命名画幅 |
| `num_frames`、`fps`、`seconds` | 帧数、帧率及秒数。`seconds` 在提交响应中为源视频的帧数÷24。完成后按已发布结果的元数据返回 `seconds`，并在有实测时长时用 `requested_seconds` 保留原值 |
| `status`、`progress` | 当前任务状态。成功完成时进度为 100，其余为 0 |
| `created_at`、`completed_at`、`expires_at` | Unix 时间戳，单位为秒。完成时间在结束后返回，过期时间以最新查询响应为准 |
| `media_type`、`file_name`、`bytes`、`sha256` | 任务完成后返回的文件信息，可用大小和 SHA-256 校验下载结果 |
| `error` | 失败时可查看 `code`、`message` 和 `retryable` |
| `metrics` | 任务耗时信息，包括队列、执行、后处理及 `super_resolution` 阶段，按实际执行返回 |

用 `GET /v1/videos/{id}` 轮询。状态依次可能是 `queued` 和 `in_progress`，以及终态 `completed`、`failed`、`cancelled` 或 `expired`。故障自动重试时，任务可能重新显示 `queued`，继续使用原 ID 查询。建议轮询间隔 1–3 秒。列表接口按 `model` 过滤时只查该模型的生成池，查询重生成结果请使用上述任务 ID。

```bash
# 使用提交结果的 id
TASK_ID=video-01M00000000000000000000000
curl -fsS "$GATEWAY_URL/v1/videos/$TASK_ID"

# status=completed 后，下载返回的 MP4 数据
curl -fsS "$GATEWAY_URL/v1/videos/$TASK_ID/content" -o regenerated.mp4
```

排障可调用 `GET /v1/videos/{id}/diagnostics` 查询阶段耗时和尝试次数。需要停止任务时调用 `DELETE /v1/videos/{id}`，正在执行的任务可能返回 202，继续查询确认终态。对已完成任务调用 DELETE 会将结果标记为过期并清理制品。

## 常见错误和重试

HTTP 错误响应使用 JSON，关键字段为 `error.code` 和 `error.message`。已受理任务的执行失败通过查询响应的 `status: failed` 和 `error` 返回。

| 阶段 | HTTP | 常见原因与处理 |
|---|---:|---|
| 提交 | 400 | 表单缺字段、重复或含额外参数，档位非法，文件或源视频规格不合规，模型未开通重生成。按响应提示修正 |
| 提交 | 404 | 请求使用未知模型，或部署尚未提供该路径 |
| 提交 | 409 | 幂等键与其它请求冲突。核对原请求，新的业务请求使用新键 |
| 提交 | 413 | 上传文件、请求体或 Gateway 转换后的源视频超过限额。核对能力声明 |
| 提交 | 415 | 请求的 Content-Type 不是 `multipart/form-data` |
| 提交 | 429 | `queue_full`。等候队列已满，稍后退避重试，复用本次提交的幂等键 |
| 提交 | 503 | 暂无可接单的兼容 SR Worker，可在退避后重试 |
| 查询或下载 | 404 | 任务 ID 未找到 |
| 下载 | 409 | 结果未就绪，继续查询任务 |
| 下载 | 422 | 任务已失败或取消，查看任务状态和错误 |
| 下载 | 410 | 结果过期或不可用 |

上线后可先调用 `GET /v1/models`，在 `minimax-h3` 的 `video_capabilities.regeneration` 下确认 `supported`、`available`、`endpoint`、尺寸表及媒体限制。重生成能力的 `available` 表示有兼容 SR Worker 可接单，剩余排队名额以创建请求结果为准。
