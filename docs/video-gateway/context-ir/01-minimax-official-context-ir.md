---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "01 · MiniMax-H3 官方 ContextIR 接口介绍"
---

核对日期：2026-11-11。面向接口使用者；公开接口可能更新，以链接中的官方文档为准。

关联阅读：[open-h3-ir](02-open-h3-ir.md) · [我们的修复与Gateway接口](03-customizations-and-gateway-api.md)。

## 作用与边界

ContextIR将用户描述及参考素材整理为适合H3生成的结构化提示词。它只编译提示词，视频生成需要另行提交。官方说明其实现尚未开源；不能把第三方开源实现视为官方服务的等价版本。[官方模型说明](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/README.md)

```text
用户意图＋参考素材 → ContextIR任务 → 结构化提示词 → 另行提交视频生成
```

## 调用方式

| 操作 | 官方接口 |
|---|---|
| 创建编译任务 | `POST /v2/h3_context_ir` |
| 查询状态和结果 | `GET /v2/query/video_generation/{task_id}` |

创建使用JSON和Bearer API Key，返回`task_id`。查询返回的任务类型为`h3_context_ir`；成功状态为`succeeded`，提示词位于`task.content.prompt`。它与我们Gateway的`completed`状态及顶层`content.prompt`不同。[创建接口](https://platform.minimax.cn/docs/api-reference/video-generation-v2-h3-context-ir) · [查询接口](https://platform.minimax.cn/docs/api-reference/video-generation-v2-query)

## 核心输入

- `model`：`MiniMax-H3`。
- `content`：文本及图片、视频、音频引用；必须有非空文本。
- `duration`：目标时长，整数4～15秒。
- `ratio`：目标比例；纯文本需明确比例，首尾帧按图片自适应。
- `callback_url`：可选状态回调。

任务场景由素材及role表达：首帧`first_frame`、尾帧`last_frame`；参考素材使用`reference_image`、`reference_video`、`reference_audio`。首尾帧与参考模式不能混用。具体媒体大小、数量及URL格式按官方限制处理，不能照搬我们的multipart字段。[官方参数说明](https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir)

## 最小示例

下面是自行编写的纯文本示例，不会自动触发视频生成。API Key从调用环境取得，不写进文档或日志。

```bash
export MINIMAX_BASE_URL='https://api.minimax.cn'
# MINIMAX_API_KEY由调用环境提供
curl -sS "${MINIMAX_BASE_URL}/v2/h3_context_ir" \
  -H "Authorization: Bearer ${MINIMAX_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-H3","content":[{"type":"text","text":"生成5秒写实视频：红色陶瓷杯静置木桌，摄影机缓慢靠近，无对白。"}],"duration":5,"ratio":"16:9"}'

# 将创建响应中的task_id填入TASK_ID
curl -sS "${MINIMAX_BASE_URL}/v2/query/video_generation/${TASK_ID}" \
  -H "Authorization: Bearer ${MINIMAX_API_KEY}"
```

查询状态包括`queued/running/succeeded/failed/cancelled`；HTTP查询成功不代表任务成功，应检查任务状态及错误。成功后再读取提示词并进入生成步骤。[官方创建与回调状态](https://platform.minimax.cn/docs/api-reference/video-generation-v2-h3-context-ir)

## 与我们测试的关系

本地对照曾观察到：中文输入也可能得到英文场景描述，而台词仍保持中文。该观察不代表所有请求的语言承诺。公开创建字段未列出独立的输出描述语言开关；不要把提示词中的语言要求误当作API参数。

官方接口、open-h3-ir和我们Gateway的字段、状态、鉴权及素材提交方式不同，不能只替换URL直接互用。实际使用账户是否有调用权限、额度和相应计费资格，应在官方账户侧确认，不能从订阅名称推断。
