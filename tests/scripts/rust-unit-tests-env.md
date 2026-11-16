# DingoRouter Rust 单元测试 — 环境依赖与准备记录

记录在 `172.30.14.203`（root/jn@123）的 `/root/gjn/dynamo`（git HEAD `9645f83f6`）执行 23 行 Rust 单元测试所需的全部环境依赖、下载安装、升级、覆盖操作，以及复现命令。配套脚本：同目录 `env-setup.sh`（环境准备）、`run-rust-unit-tests.sh`（串行跑测试）。

## 环境基线（执行前已具备）

| 项 | 状态 | 说明 |
|---|---|---|
| 源码 | `/root/gjn/dynamo`，HEAD `9645f83f6` | 已存在 |
| rustup | `/root/.cargo/bin`，工具链 `1.93.1-x86_64-unknown-linux-gnu` + `stable 1.82.0` | 已存在 |
| 系统 rustc | `/usr/bin/rustc` 1.79.0（Red Hat 包） | **过旧，不可用** |
| gcc-12 | `/usr/local/gcc-12`（GCC 12.3.0），已是默认 gcc/g++ | 已存在 |
| 系统 libstdc++ | `/lib64/libstdc++.so.6`（GCC 8） | **过旧，需让位 gcc-12** |
| protoc | `/usr/bin/protoc` 3.5.0 | **过旧，需升级** |
| 外网 | 可访问 github.com / pypi.org | 可用 |
| 核数 | 16 | — |

## 下载安装 / 升级 / 覆盖清单

### 1. 下载安装 protoc 21.12（覆盖系统 3.5.0）

- 原因：`/usr/bin/protoc`（3.5.0）早于 `--experimental_allow_proto3_optional` 引入（3.12），不识别该标志；`modelexpress-common v0.3.0` 的 `build.rs` 显式传入该标志，导致第1行 `dynamo-backend-common` 编译失败退出码 101。
- 来源：`https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-linux-x86_64.zip`
- 安装动作：
  - `protoc` 二进制 → `/usr/local/bin/protoc`（PATH 优先于 `/usr/bin/protoc`）
  - well-known proto 头（`google/protobuf/*.proto`）→ `/usr/local/include/`（解决第5行 `dynamo-ext-proc` 的 `google/protobuf/struct.proto not found`）
- 结果：`/usr/local/bin/protoc` 报 `libprotoc 3.21.12`。
- 覆盖关系：不删除 `/usr/bin/protoc`，靠 PATH 优先级覆盖。

### 2. 配置 gcc-12 libstdc++（让位系统 GCC 8 的 libstdc++）

- 原因：`zeromq-src v0.2.6+4.3.4` 用 `/usr/local/gcc-12` 的 C++ 头编译，但链接 `-lstdc++` 时 rust-lld 默认解析到系统 GCC 8 的 `/lib64/libstdc++.so`，缺 `std::__throw_bad_array_new_length` 等 C++17 符号，第13行 `dynamo-runtime` 链接失败。
- 安装动作：
  - 写 `/etc/ld.so.conf.d/gcc12.conf` 内容 `/usr/local/gcc-12/lib64`，执行 `ldconfig`（运行时解析优先 gcc-12 的 `libstdc++.so.6`）。
  - 执行命令中设 `LIBRARY_PATH=/usr/local/gcc-12/lib64`（让 `cc` 驱动链接器在链接阶段也找 gcc-12 的 libstdc++）。
  - 执行命令中设 `LD_LIBRARY_PATH=/usr/local/gcc-12/lib64`（运行时兜底）。
- 覆盖关系：不改 `/lib64/libstdc++.so.6`，通过 ldconfig 优先级 + `LIBRARY_PATH` 覆盖解析顺序。
- 前提：gcc-12 已安装在 `/usr/local/gcc-12`（本脚本不负责装 gcc-12）。

### 3. 工具链选择 1.93.1

- 原因：源码 `Cargo.toml` 用 `resolver = "3"` 与部分 2024 edition 特性，需 cargo 1.93.1；系统 rustc 1.79 不可用。
- 动作：通过 `cargo +1.93.1` 指定工具链（rustup 已装该工具链，无需再装）。

## 关键陷阱：RUSTFLAGS 覆盖 tokio_unstable

源码 `.cargo/config.toml` 在 `[build]` 与 `[target.x86_64-unknown-linux-gnu]` 都设了 `rustflags = ["--cfg", "tokio_unstable"]`。第13行 `dynamo-runtime` 用了 `tokio::runtime::Builder::enable_metrics_poll_time_histogram` 等 **unstable metrics API**，依赖此 cfg。

- 错误做法：用 `RUSTFLAGS="-L native=/usr/local/gcc-12/lib64"` 提供链接路径——env 级 `RUSTFLAGS` 会**整体覆盖** `.cargo/config.toml` 的 rustflags，导致 `tokio_unstable` 丢失，报 `E0599 no method named ... found`。
- 正确做法：用 `LIBRARY_PATH`（而非 `RUSTFLAGS`）提供链接路径，并 `unset RUSTFLAGS`，保留 `.cargo/config.toml` 原有 rustflags。

## 执行环境变量（run-rust-unit-tests.sh 已固化）

```bash
export PATH="/root/.cargo/bin:/usr/local/bin:/usr/local/gcc-12/bin:$PATH"
export CARGO_BUILD_JOBS=16
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export LD_LIBRARY_PATH="/usr/local/gcc-12/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/local/gcc-12/lib64:${LIBRARY_PATH:-}"
export CC="/usr/local/gcc-12/bin/gcc"
export CXX="/usr/local/gcc-12/bin/g++"
unset RUSTFLAGS
```

`NO_PROXY/no_proxy`：避免本地临时 HTTP 服务（测试用 Mockito）被代理截获返回 502（第一轮第8行 9 个用例失败即因此）。

## 复现步骤

```bash
cd /root/gjn/dynamo
bash tests/router/env-setup.sh          # 环境准备（幂等）
bash tests/router/run-rust-unit-tests.sh # 串行跑 23 行，日志在 /tmp/dingoRouter-rust-tests-<时间戳>/
```

## 本轮结果（20260812）

23 行全部通过。唯一数量变化：第8行 `dynamo-llm` 由 1484 → 1507 passed（+23，源码新增测试，0 failed）。汇总日志 `/tmp/dingoRouter-rust-tests-20260812/SUMMARY.txt`（脚本另生成带时间戳的目录）。

| 序号 | 包 | 结果 | passed |
|---|---|---|---|
| 1 | dynamo-backend-common | 通过 | 132 |
| 2 | dynamo-bench | 通过 | 22 |
| 3 | dynamo-codegen | 通过 | 0 |
| 4 | dynamo-data-gen | 通过 | 43 |
| 5 | dynamo-ext-proc | 通过 | 6 |
| 6 | dynamo-kv-hashing | 通过 | 0 |
| 7 | dynamo-kv-router | 通过 | 587 |
| 8 | dynamo-llm | 通过 | 1507（+23） |
| 9 | dynamo-memory | 通过 | 147 |
| 10 | dynamo-mocker | 通过 | 495 |
| 11 | dynamo-mocker-backend | 通过 | 19 |
| 12 | dynamo-rl | 通过 | 10 |
| 13 | dynamo-runtime | 通过 | 439（3 ignored） |
| 14 | dynamo-tokens | 通过 | 53 |
| 15 | dynamo-vllm-rs-backend | 通过 | 0 |
| 16 | kvbm-common | 通过 | 0 |
| 17 | kvbm-config | 通过 | 63 |
| 18 | kvbm-consolidator | 通过 | 28 |
| 19 | kvbm-engine | 通过 | 126 |
| 20 | kvbm-kernels | 通过 | 0 |
| 21 | kvbm-logical | 通过 | 504 |
| 22 | kvbm-physical | 通过 | 13 |
| 23 | libdynamo_llm | 通过 | 3 |

## 2026-11-16：响应 schema 与测试接口兼容修复

- `NvCreateChatCompletionResponse` 和 `NvCreateChatCompletionStreamResponse` 恢复
  `ToSchema` 派生；内部事件、消息边界和流式 prompt logprobs 不进入公开 schema，
  手写序列化实现保持不变。
- 响应序列化测试补齐新增的内部字段；稀疏 usage 测试显式使用非权威计数模式，
  保留跨 chunk 累加的断言。
- Agentic replay 测试使用 `AgenticSnapshotOptions::seeded(7)`，
  由依赖库初始化新增快照选项，不改变原有采样种子。

这类 Cargo `101` 编译错误发生在测试执行前，不能将汇总中的 `failed=0`
解释为通过。针对受影响包的验证命令：

```bash
cargo +1.96.1 test --locked \
  -p dynamo-backend-common -p dynamo-llm -p dynamo-mocker --lib
```

本次隔离 Linux 验证（Rust 1.96.1）：

- `dynamo-backend-common`：165 passed，0 failed。
- `dynamo-llm`：4086 passed，4 failed，17 ignored；已进入测试执行阶段，
  本节列出的 schema、字段及参数编译错误均已消除。
- OpenAPI 自定义路由契约、响应序列化和稀疏 usage 计数测试通过。
- 仍失败的用例为
  `test_chat_typed_metrics_fast_path_observes_without_annotation`、
  `media_schemas_match_the_committed_files`（`video.json`）、
  `responses_and_chat_accept_qualified_functions_but_reject_invalid_namespaces`、
  `text_message_content_remains_valid`。本次没有跳过这些测试或修改其断言，
  不能将编译修复表述为全部单元测试通过。
- `dynamo-mocker` 单独执行：322 passed，0 failed；包含本节修复的四处快照初始化。

## 2026-11-16：05b38d45 的四项测试失败

Actions run `43852384781` / job `133247953771` 已完成编译：
`dynamo-llm` 为 4086 passed、4 failed、17 ignored；其余三个 smoke 包通过。
这次退出码 101 来自测试断言，不是 Rust 编译错误。

- `test_chat_typed_metrics_fast_path_observes_without_annotation`：
  手写响应序列化漏传 `model_eos`，跨组件往返后 `[0]` 变成 `[]`。
  序列化现同时保留工具完成证据和模型 EOS；HTTP SSE 转换仍移除两者，
  不向 OpenAI 客户端暴露内部字段。
- `media_schemas_match_the_committed_files`：
  视频 Rust 类型新增的 `input_references`、`aspect_ratio`、`frame_indices`
  未同步到 `video.json` 及 Python 协议模型。
- `responses_and_chat_accept_qualified_functions_but_reject_invalid_namespaces`：
  请求入口原来只校验转换后的扁平工具名，遗漏原始 namespace。
  入口现复用已有的 `validated_effective_tools` 方法，分别校验 namespace
  和叶子名称；普通 Chat Completions 仍走原有路径。
- `text_message_content_remains_valid`：
  测试中的 tool 消息没有对应的前置 assistant 调用。补齐 `call_1` 调用，
  不放宽对孤立或重复 tool result 的拒绝规则。

在具备完整依赖的 Linux 环境中执行针对性复测：

```bash
for filter in \
  test_chat_typed_metrics_fast_path_observes_without_annotation \
  media_schemas \
  responses_and_chat_accept_qualified_functions_but_reject_invalid_namespaces \
  text_message_content_remains_valid \
  tool_results_require_an_unanswered_preceding_call \
  test_forced_tool_choice_without_function_tools_is_invalid
do
  cargo +1.96.1 test --locked -p dynamo-llm --lib "$filter" || exit "$?"
done
```

本次本机验证：

- Rust 1.96.1 隔离程序编译并执行实际源码中的响应序列化、消息历史和
  工具名称校验函数；EOS 与工具完成证据保留、SGLang 字段提升、空字段省略、
  有效工具结果接收及孤立结果拒绝均通过。此验证不覆盖完整 HTTP/Responses 调用链。
- 提取实际视频类型与 schema 生成函数，使用 utoipa 5.5.0 生成的
  `video.json` 与提交文件逐字节一致。
- 修正生成器遗留的 `components/src/dynamo` 输出路径及图片类型导入，
  使用当前 `dingo.common.protocols` 包；固定依赖
  `datamodel-code-generator==0.83.0` 的 `--check` 通过。
- Python 实际协议模型验证通过：请求级 `input_references` 往返保留，
  `nvext.aspect_ratio/frame_indices` 往返保留，`input_references`
  不再错误地属于 `VideoNvExt`。
- 未运行完整 Linux `dynamo-llm` 测试：远端 master 的 Teleport 连接被拒绝，
  本机无完整 Linux 构建依赖。以上局部验证不等同于 Actions 已通过。
