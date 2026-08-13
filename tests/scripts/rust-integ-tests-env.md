# DingoRouter Rust 集成测试 — 环境依赖与准备记录

记录在 `172.30.14.203`（root/jn@123）的 `/root/gjn/dynamo`（git HEAD `9645f83f6`）执行 5 个 crate 的 Rust 集成测试（`--test '*'`）所需的全部环境依赖、下载安装、升级、覆盖操作，以及本轮结果。配套脚本：同目录 `env-setup-integ.sh`（环境准备）、`run-rust-integ-tests.sh`（串行跑测试）。

## 环境依赖

集成测试的环境依赖与单元测试**完全相同**（详见 `rust-unit-tests-env.md`），复用 `env-setup.sh`：

| 依赖 | 要求 | 系统现状 | 动作 |
|---|---|---|---|
| rust 工具链 | 1.93.1 | rustup 已装 | `cargo +1.93.1` |
| protoc | ≥21.12 | 系统 3.5.0 过旧 | 升级到 21.12，装 well-known proto 到 `/usr/local/include` |
| gcc-12 libstdc++ | 链接/运行时可用 | 系统 GCC 8 的 libstdc++ 缺 C++17 符号 | ldconfig + `LIBRARY_PATH` 让位 gcc-12 |
| tokio_unstable cfg | 保留 `.cargo/config.toml` | — | `unset RUSTFLAGS` 避免覆盖 |
| NO_PROXY | 本地 Mockito 服务 | 无代理 | `NO_PROXY/no_proxy=127.0.0.1,localhost` |

**唯一额外步骤**：kvbm-consolidator 的 e2e fixture 重新生成（见下节）。

## 额外修复：kvbm-consolidator e2e fixture

### 失败现象

`cargo test -p kvbm-consolidator --test e2e` 报：
```
thread 'e2e_full_vllm_replay' panicked at lib/kvbm-consolidator/tests/e2e.rs:198:
deserialize fixture: Syntax("invalid type: integer `-26`, expected a sequence")
```

### 根因

- `e2e.rs:198` 用 `rmp_serde::from_slice::<Vec<Vec<u8>>>` 反序列化 `tests/fixtures/vllm_capture.msgpack`。
- 仓库内的 fixture 文件（153 字节）是**旧格式**，反序列化时读到 `integer -26` 而非 `sequence`。
- `ensure_fixture()`（e2e.rs:117）只在文件**缺失或空**时才重新生成；文件存在且非空就直接用，于是格式不匹配。
- 上轮报告的失败信息是 `fixture regeneration failed: No such file or directory`（fixture 缺失），本轮是格式不匹配——说明 fixture 文件曾被手工/旧流程写入错误内容。

### 修复

`e2e.rs` 内有一个 `#[ignore]` 的 `e2e_regenerate` 测试，调用 `regenerate_fixtures()` 用 `make_synthetic_payload_blobs()` + `rmp_serde::to_vec` 写出正确格式 fixture。手动运行它：

```bash
cargo +1.93.1 test -p kvbm-consolidator --test e2e -- --ignored e2e_regenerate
```

执行后 `lib/kvbm-consolidator/tests/fixtures/vllm_capture.msgpack` 被覆盖为 130 字节的正确格式，`e2e_full_vllm_replay` 随即通过。`e2e_full_trtllm_replay` 当前被 `#[ignore]`（不跑）。

> 注意：`regenerate_fixtures()` 用相对路径 `tests/fixtures/...` 写入，工作目录须为 crate 根（`lib/kvbm-consolidator`）。`cargo test` 默认以 crate 根为工作目录，所以从源码根目录跑即可。

## 执行环境变量（run-rust-integ-tests.sh 已固化）

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

## 复现步骤

```bash
cd /root/gjn/dynamo
bash tests/router/env-setup-integ.sh      # 环境准备（含 fixture 重新生成，幂等）
bash tests/router/run-rust-integ-tests.sh # 串行跑 5 个 crate，日志在 /tmp/dingoRouter-rust-integ-tests-<时间戳>/
```

## 本轮结果（20260812）

5 个 crate 集成测试全部通过（退出码 0）。

| 序号 | Crate | 命令 | 结果 | passed | failed | ignored | 与上轮对比 |
|---|---|---|---|---|---|---|---|
| 1 | dynamo-kv-hashing | `--test '*'` | 通过 | 20 | 0 | 0 | 一致 |
| 2 | dynamo-kv-router | `--test '*' --features standalone-indexer` | 通过 | 3 | 0 | 0 | 一致 |
| 3 | dynamo-runtime | `--test '*'` | 通过 | 8 | 0 | 1 | 一致（上轮详情记 6，总览记 8，本轮确认 8） |
| 4 | kvbm-consolidator | `--test '*'` | 通过 | 18 | 0 | 1 | e2e 由 2 failed 修复为 1 passed |
| 5 | kvbm-kernels | `--test '*'` | 通过 | 2 | 0 | 0 | +2（源码新增 stub_build 测试） |

明细（kvbm-consolidator 7 个测试文件）：chaos_properties 3、dedup 3、e2e 1+1ignored、kvbm_bridge 2、lifecycle 2、output_contract 3、zmq_ingress 4。

### 与上轮关键差异

1. **kvbm-consolidator e2e**：上轮 `e2e_full_vllm_replay` + `e2e_full_trtllm_replay` 共 2 failed（fixture 格式/路径问题）；本轮重新生成 fixture 后，`e2e_full_vllm_replay` 通过，`e2e_full_trtllm_replay` 转为 `#[ignore]`，0 failed。**本轮通过率 100%（上轮 96%）**。
2. **kvbm-kernels**：上轮 0 tests（需 GPU）；本轮 HEAD 新增 `stub_build.rs`（2 passed，无 GPU 的 stub 模式可跑），`kernel_roundtrip`/`memcpy_batch` 仍 0 tests（需 CUDA）。
3. **dynamo-runtime**：上轮文档因 tokio unstable 失败改为"直接运行预编译二进制"；本轮保留 `.cargo/config.toml` 的 `tokio_unstable` 后 `cargo test` 直接编译通过，无需预编译二进制权宜。

## 脚本说明

| 脚本 | 作用 |
|---|---|
| `env-setup.sh` | 通用环境准备（protoc/gcc-12/tokio_unstable），单元/集成测试共用。 |
| `env-setup-integ.sh` | 集成测试环境准备：调用 `env-setup.sh` + 重新生成 kvbm-consolidator e2e fixture。 |
| `run-rust-integ-tests.sh` | 5 个 crate 集成测试串行执行，汇总到 `/tmp/dingoRouter-rust-integ-tests-<时间戳>/SUMMARY.txt`。 |
