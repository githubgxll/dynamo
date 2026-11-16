---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "ContextIR API：参数、返回值与官方对照"
subtitle: "提示词编译与视频生成分开，画布只供内部校验"
---

更新：2026-11-16。本文为当前Gateway ContextIR接口依据。ContextIR解释素材和意图并生成H3提示词；分辨率、超分方法、GPU容量及视频文件由视频生成阶段决定。示例地址和任务ID为占位值。

本轮素材意图解析验证：399项Gateway/Worker相关本地回归、1294项编译器测试及23项格式控制通过；真实Qwen分类22/22、Gateway编译36/36、负向接口8/8通过。原有14项Ref2VA编译均未降级；一个刻意构造的“显式音色参数与背景音乐提示词矛盾”请求返回合法降级草稿，角色仍遵循显式参数。技术完成不等同于音色、配乐或首尾画面质量已获人工认可。

## 接口

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/v1/context-ir` | multipart提交；新任务202，幂等重放200 |
| GET | `/v1/context-ir/{id}` | 查询状态和编译结果 |
| DELETE | `/v1/context-ir/{id}` | 取消执行中的任务，或删除已结束结果 |
| GET | `/v1/context-ir/{id}/diagnostics` | 查询耗时、用量及素材角色等诊断信息 |

生成视频仍需调用`POST /v1/videos`，并提交原参考素材、编译后的prompt及调用方选择的分辨率等生成参数。

## 建议传的参数

| 参数 | 必要性 | 说明 |
|---|---|---|
| `model` | 必填 | 已配置ContextIR绑定的公开视频模型名；当前模型路由区分FL2VA和Ref2VA |
| `prompt` | 必填 | 用户原始提示词，不能空 |
| `seconds` | 与num_frames二选一必填 | 目标名义时长，4–15秒；也可以改用num_frames，两者不能同时传 |
| `ratio` | T2VA按下表要求提供；其它可省略 | `21:9/16:9/4:3/1:1/3:4/9:16/adaptive` |
| `input_references` | 首尾帧、Ref2VA必填 | 重复上传字段；T2VA无附件，首尾帧传图片，Ref2VA传参考图片/视频/音频 |
| `reference_options` | 可选 | JSON数组，通过从0开始的上传index指定role、note、soundtrack_policy |
| `Idempotency-Key` | 可选请求头，建议重试时用 | 同一业务提交重试复用；不同意图或素材使用同键返回409 |

### 必填项与素材用途默认值

| 场景 | 必须提供 | 说明 |
|---|---|---|
| T2VA纯文本编译，未使用旧尺寸参数 | `ratio`或兼容别名`aspect_ratio` | 六种具体比例之一，不能是adaptive；合法旧size或成对宽高可代替 |
| 首帧、尾帧或首尾帧编译 | `input_references`上传相应图片 | 无图片按T2VA处理 |
| Ref2VA编译 | 至少一张图片或一段视频 | 可以附加音频，不能只有音频 |

`reference_options`以及其中的`role`、`soundtrack_policy`均可省略。处理顺序为：**显式参数 > 提示词中可确认的用途 > 默认值**。用于识别用途的短文本模型调用是独立的`intent_resolution`阶段，在素材分析和提示词编写之前完成。

| 省略的配置 | 提示词没有明确说明时 | 提示词明确说明时 |
|---|---|---|
| FL2VA单张图片role | 首帧 | 可识别仅尾帧，例如“图1作为尾帧” |
| FL2VA两张图片role | 上传图1首、图2尾 | 可识别反向关系，例如“从图2开始，以图1结束”；只明确一张时补齐另一张 |
| Ref2VA独立音频role | `voice_timbre`，参考音色 | 可选择`bgm`，作为背景音乐 |
| 参考视频soundtrack_policy | `preserve`，保留原音轨的编译语义 | 可选择`ignore`，忽略原音轨 |

`soundtrack_policy`只用于参考视频，不用于独立音频。默认preserve是本地产品策略；官方公开ContextIR文档未承诺统一的保留/忽略默认值。识别未知、超时或格式无效时采用上述默认值，并使用现有warnings说明，不仅因意图识别失败而拒绝编译。该阶段不进行ASR或听辨，也不保证判断音频本身是人声还是音乐。

填写`reference_options[]`时，仍必须提供从0开始、不重复且不越界的`index`；显式未知role、非法policy和互相矛盾的首尾参数继续返回400。每条`dialogue[]`必须有非空`text`；旧尺寸width与height必须一起提供，不能与size并用。

反向首尾编译返回`generation_parameters.frame_indices=[-1,0]`，表示原上传图1是尾帧、图2是首帧。`references.index`始终对应原上传顺序；`labels`会反映H3实际首尾顺序。调用方使用原素材加返回参数提交视频API即可，自适应比例取最终首帧。仅尾帧返回`[-1]`。

直接调用`POST /v1/videos`不会额外执行本阶段，也不会从自然语言自动推断首尾；需要这种能力时先编译，再复用返回的frame_indices。反向两图当前经Gateway整理为原生Worker的首尾顺序，使用受原有内联消息限额约束的路径；普通正向共享文件路径不变。

## 高级参数与兼容参数

| 参数 | 默认/作用 | 是否推荐普通调用方传 |
|---|---|---|
| `context_ir_seed` | 默认7，控制编译模型采样；与视频seed不同 | 通常省略；复现测试时使用 |
| `creativity` | 默认balanced；restrained/balanced/bold/extreme | 通常省略 |
| `dialogue` | 可选JSON数组，明确text以及可选language/speaker | 提示词已经说清台词时可以省略 |
| `silent` | 默认false；true表示不添加背景音乐，仍允许对白及环境音 | 通常省略；不等于视频静音 |
| `aspect_ratio` | ratio的兼容名称，同时传必须一致 | 新调用统一用ratio |
| `num_frames` | seconds的高级替代，96–360帧，按24fps计算 | 新调用优先seconds |
| `resolution` | 兼容接收768P/1080P/2K，检查值的格式；不决定编译画布、不检查超分能力、不回显 | 编译时不需要传，留给视频生成请求 |
| `size`、`width`＋`height` | 兼容旧调用，提取比例及校验既有尺寸；不返回具体像素尺寸 | 新调用不需要传 |

`seed`、`num_inference_steps`、`generate_sound`、超分开关等视频生成参数不属于编译接口。`callback_url`和官方JSON `content[]`格式当前也不接受。未知字段或重复标量会返回400。

## 画幅与分辨率

- T2VA：传具体ratio即可，无需resolution或size。
- FL2VA首尾帧：跟随首帧或唯一尾帧的比例。省略ratio或传adaptive均可；按官方ContextIR语义，传其它合法具体ratio也会被忽略。首尾帧可有不同尺寸。
- Ref2VA：可传具体ratio，也可省略或传adaptive。本实现优先取首个参考视频，否则首张参考图，并选择最接近的基础预设比例；官方未公开完整的自动选幅算法，不保证两边选择结果完全一致。
- `resolution=1080P/2K`不会引发编译接口的超分拒绝。编译不读取视频池的SR profile或Worker是否具备超分能力。原生、超分或其它实现由生成阶段选择。
- open-h3-ir内部需要canvas做格式、帧数及资源估算。Gateway使用基础编译画布并验证返回契约，公开响应不返回该canvas，也不返回size、width、height或resolution。
- 旧size请求仍做基础尺寸/已知交付尺寸及比例一致性校验；合法高分辨率size只选择编译比例，不执行视频或超分。

`generation_parameters`只提供编译阶段的比例、时长及关键帧映射，不是一份完整的视频生成请求。调用方应保留原参考素材，并在生成时另外选择resolution或显式输出尺寸。Ref2VA adaptive在编译结果中返回本地选中的具体比例；FL2VA返回adaptive，生成阶段继续按原关键帧确定尺寸。

## 最小调用

```bash
export GATEWAY_URL='https://gateway.example.com'
curl -sS "${GATEWAY_URL}/v1/context-ir" \
  -H 'Idempotency-Key: compile-example-001' \
  -F 'model=minimax-h3' \
  --form-string 'prompt=一只猫坐在窗边，轻轻转头，没有对白。' \
  -F 'seconds=5' -F 'ratio=16:9'

# 首帧模式：图片决定比例，不需要resolution或size
curl -sS "${GATEWAY_URL}/v1/context-ir" \
  -F 'model=minimax-h3' --form-string 'prompt=让首帧人物自然微笑。' \
  -F 'seconds=5' -F 'input_references=@first-frame.png'

# TASK_ID取创建响应的id，未完成时继续查询，不重新创建
curl -sS "${GATEWAY_URL}/v1/context-ir/${TASK_ID}"
```

## 返回字段

| 字段 | 含义 | 上层是否需要 |
|---|---|---|
| `id`、`status` | 任务ID与状态：queued/in_progress/completed/failed/cancelled；过期查询410 | 需要 |
| `content.prompt` | completed后提供编译提示词 | 需要 |
| `usage` | 聚合实际Qwen调用的用量，包含意图解析、视觉分析、编写及纠错 | 计费需要 |
| `warnings` | 编译警告；合法降级草稿以context_ir_degraded_result标记 | 建议检查 |
| `generation_parameters` | ratio、seconds或num_frames、fps，以及所需frame_indices | 可复用；生成时另外指定分辨率 |
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

usage为多次模型调用之和，不是只统计最终改写文本。上游报告的视觉token包含在prompt_tokens内；缓存计数由上游返回后汇总，不自行估算。回执缺失时complete=false，未知缓存为null。进行中或未调用模型的任务usage可能为null。

公开结果对历史存储结果也隐藏像素字段；原始内部结果及摘要不重写，已受理任务与幂等记录保持可用。completed且带降级警告时，content.prompt是实际回退草稿，调用方可根据warnings决定是否使用。

## 与MiniMax官方ContextIR对比

官方依据为[创建ContextIR任务](https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir)及[查询任务](https://platform.minimax.io/docs/api-reference/video-generation-v2-query)，核对日期2026-11-15。

| 方面 | 官方 | 我们 |
|---|---|---|
| 输入协议 | JSON content[]，使用媒体URL等引用 | multipart，直接上传素材 |
| 提交 | /v2/h3_context_ir，返回task_id | /v1/context-ir，202返回任务对象及id |
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
