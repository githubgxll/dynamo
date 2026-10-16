#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DingoRouter Rust 单元测试环境准备脚本（底层，链式自包含）
# 在源码目录执行测试前先跑本脚本。幂等：已满足的步骤会跳过。
# 会自动检测并安装：rustup+1.93.1、gcc-12、python3.11、protoc 21.12。
# 支持 root 和非 root 用户；非 root 时安装到 ~/.local。
set -uo pipefail
# 注意：不加 -e，让单个安装步骤失败不中断脚本（幂等设计：已装则跳过，装不了则 WARN）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUST_TOOLCHAIN="1.93.1"
PROTOC_VERSION="21.12"
PROTOC_URL="https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-linux-x86_64.zip"
PY_VERSION="3.11"
PY_MINOR="3.11.11"

# 非 root 时使用 ~/.local 作为安装前缀
if [ "$(id -u)" -eq 0 ]; then
    LOCAL_PREFIX="/usr/local"
    SUDO=""
else
    LOCAL_PREFIX="${HOME}/.local"
    SUDO=""
fi
mkdir -p "${LOCAL_PREFIX}/bin" "${LOCAL_PREFIX}/include" "${LOCAL_PREFIX}/lib"

# 检测 gcc-12 的路径（可能在 /usr/local/gcc-12 或系统已装 gcc-12）
GCC12_PREFIX="/usr/local/gcc-12"
GCC12_LIB64="${GCC12_PREFIX}/lib64"
# 如果 /usr/local/gcc-12 不存在，检查系统是否有 gcc-12
if [ ! -x "${GCC12_PREFIX}/bin/gcc" ]; then
    if command -v gcc-12 >/dev/null 2>&1; then
        GCC12_PREFIX="$(dirname $(which gcc-12))/.."
        GCC12_LIB64="${GCC12_PREFIX}/lib64"
    elif [ -x /opt/rh/gcc-toolset-12/root/usr/bin/gcc ]; then
        GCC12_PREFIX="/opt/rh/gcc-toolset-12/root/usr"
        GCC12_LIB64="${GCC12_PREFIX}/lib64"
    else
        GCC12_PREFIX=""
        GCC12_LIB64=""
    fi
fi

export PATH="${LOCAL_PREFIX}/bin:${HOME}/.cargo/bin:${PATH}"

echo "=== [0/7] 检测/安装 rustup + 工具链 ${RUST_TOOLCHAIN} ==="
CARGO_HOME="${HOME}/.cargo"
if ! command -v rustup >/dev/null 2>&1 && [ ! -x "${CARGO_HOME}/bin/rustup" ]; then
    echo "rustup 未安装，使用官方脚本安装 ..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"
    # shellcheck disable=SC1091
    source "${CARGO_HOME}/env" 2>/dev/null || true
else
    echo "OK: rustup 已安装"
fi
export PATH="${CARGO_HOME}/bin:${PATH}"
if ! rustup toolchain list 2>/dev/null | grep -q "${RUST_TOOLCHAIN}"; then
    echo "安装工具链 ${RUST_TOOLCHAIN} ..."
    rustup toolchain install "${RUST_TOOLCHAIN}"
else
    echo "OK: 工具链 ${RUST_TOOLCHAIN} 已存在"
fi
echo "cargo +${RUST_TOOLCHAIN} -> $(cargo +${RUST_TOOLCHAIN} --version)"

echo "=== [1/7] 检测 gcc（编译 zmq-sys / Rust binding 所需）==="
# 优先用 gcc-12；没有就用系统 gcc（只要 >= 11 即可）
if [ -n "${GCC12_PREFIX}" ] && [ -x "${GCC12_PREFIX}/bin/gcc" ]; then
    echo "OK: gcc-12 已安装于 ${GCC12_PREFIX}"
    export CC="${GCC12_PREFIX}/bin/gcc"
    export CXX="${GCC12_PREFIX}/bin/g++"
elif command -v gcc-12 >/dev/null 2>&1; then
    echo "OK: 系统 gcc-12 可用 ($(gcc-12 --version | head -1))"
    export CC="$(which gcc-12)"
    export CXX="$(which g++-12)"
else
    SYS_GCC_VER=$(gcc --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+\.\d+' | head -1)
    echo "WARN: gcc-12 未找到，使用系统 gcc: ${SYS_GCC_VER}"
    echo "  如 zmq-sys 链接失败，需手动安装 gcc-12 或确保 libstdc++ >= C++17"
    export CC="$(which gcc)"
    export CXX="$(which g++)"
fi
echo "CC=${CC}  CXX=${CXX}"

# 确保 cc 链接器可用（cargo build script 默认调 cc）
if ! cc --version >/dev/null 2>&1; then
    mkdir -p "${LOCAL_PREFIX}/bin"
    ln -sf "${CC}" "${LOCAL_PREFIX}/bin/cc" 2>/dev/null || true
    ln -sf "${CXX}" "${LOCAL_PREFIX}/bin/c++" 2>/dev/null || true
fi

echo "=== [2/7] 检测/安装 protoc ${PROTOC_VERSION} ==="
PROTOC_BIN="${LOCAL_PREFIX}/bin/protoc"
need_protoc=0
if [ -x "${PROTOC_BIN}" ]; then
    cur=$("${PROTOC_BIN}" --version | awk '{print $2}' | sed 's/^3\.//')
    [ "$cur" != "${PROTOC_VERSION}" ] && need_protoc=1 || echo "OK: ${PROTOC_BIN} 已是 ${PROTOC_VERSION}"
else
    need_protoc=1
fi
# 也检查系统 protoc
if [ "$need_protoc" -eq 1 ] && command -v protoc >/dev/null 2>&1; then
    SYS_PROTOC_VER=$(protoc --version | awk '{print $2}' | sed 's/^3\.//')
    if [ "$SYS_PROTOC_VER" = "${PROTOC_VERSION}" ]; then
        echo "OK: 系统 protoc ${PROTOC_VERSION} 可用"
        need_protoc=0
    fi
fi
if [ "$need_protoc" -eq 1 ]; then
    tmpdir=$(mktemp -d)
    echo "下载 ${PROTOC_URL} ..."
    curl -sS -L -o "${tmpdir}/protoc.zip" "${PROTOC_URL}"
    unzip -o -q "${tmpdir}/protoc.zip" -d "${tmpdir}/protoc-pkg"
    cp "${tmpdir}/protoc-pkg/bin/protoc" "${PROTOC_BIN}"
    chmod +x "${PROTOC_BIN}"
    cp -r "${tmpdir}/protoc-pkg/include/"* "${LOCAL_PREFIX}/include/" 2>/dev/null || true
    rm -rf "${tmpdir}"
    echo "OK: protoc $(${PROTOC_BIN} --version) 已安装到 ${PROTOC_BIN}"
fi

echo "=== [3/7] 配置 libstdc++ 链接/运行时路径 ==="
if [ -n "${GCC12_LIB64}" ] && [ -d "${GCC12_LIB64}" ]; then
    # 有 gcc-12 的 libstdc++：写 ldconfig 或用 LD_LIBRARY_PATH
    if [ "$(id -u)" -eq 0 ]; then
        conf=/etc/ld.so.conf.d/gcc12.conf
        echo "${GCC12_LIB64}" > "$conf" 2>/dev/null && ldconfig 2>/dev/null || true
        echo "OK: 已配置 ldconfig (${conf})"
    else
        export LD_LIBRARY_PATH="${GCC12_LIB64}:${LD_LIBRARY_PATH:-}"
        echo "OK: LD_LIBRARY_PATH=${GCC12_LIB64} (非 root)"
    fi
    export LIBRARY_PATH="${GCC12_LIB64}:${LIBRARY_PATH:-}"
else
    echo "OK: 无独立 gcc-12 libstdc++ 路径，使用系统默认"
fi

echo "=== [4/7] 检测/安装 python${PY_VERSION} ==="
# 优先 /usr/bin/python3.11，其次 ~/.local/bin/python3.11
PY_BIN=""
for p in /usr/bin/python3.11 "${LOCAL_PREFIX}/bin/python3.11" python3.11; do
    if "$p" --version >/dev/null 2>&1; then PY_BIN="$p"; break; fi
done
if [ -n "${PY_BIN}" ] && "${PY_BIN}" --version 2>&1 | grep -q "${PY_VERSION}"; then
    echo "OK: ${PY_BIN} 已存在 ($(${PY_BIN} --version 2>&1))"
else
    echo "python${PY_VERSION} 未找到，尝试安装 ..."
    if [ "$(id -u)" -eq 0 ]; then
        if command -v dnf >/dev/null 2>&1; then
            dnf install -y python3.11 python3.11-pip || true
        elif command -v apt-get >/dev/null 2>&1; then
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -y && apt-get install -y python3.11 python3.11-venv python3.11-distutils || true
        fi
    fi
    PY_BIN="/usr/bin/python3.11"
    if [ ! -x "${PY_BIN}" ]; then
        echo "从源码编译 python ${PY_MINOR} ..."
        tmpdir=$(mktemp -d)
        curl -sS -L -o "${tmpdir}/Python.tar.xz" "https://www.python.org/ftp/python/${PY_MINOR}/Python-${PY_MINOR}.tar.xz"
        tar xf "${tmpdir}/Python.tar.xz" -C "${tmpdir}"
        cd "${tmpdir}/Python-${PY_MINOR}"
        ./configure --prefix="${LOCAL_PREFIX}" --enable-optimizations --with-ensurepip=install
        make -j"$(nproc)"
        make install || { echo "WARN: python make install 失败" >&2; }
        cd "${SCRIPT_DIR}/../.."
        rm -rf "${tmpdir}"
        PY_BIN="${LOCAL_PREFIX}/bin/python3.11"
    fi
    [ -x "${PY_BIN}" ] && echo "OK: ${PY_BIN} 已安装 ($(${PY_BIN} --version 2>&1))" || { echo "ERROR: python${PY_VERSION} 安装失败" >&2; exit 1; }
fi
# ensurepip / pip bootstrap
if ! "${PY_BIN}" -m pip --version >/dev/null 2>&1; then
    if "${PY_BIN}" -m ensurepip >/dev/null 2>&1; then
        echo "OK: pip 已通过 ensurepip 安装"
    else
        echo "ensurepip 不可用，使用 get-pip.py 安装 pip ..."
        curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        "${PY_BIN}" /tmp/get-pip.py >/dev/null 2>&1 || { echo "ERROR: pip 安装失败" >&2; exit 1; }
        rm -f /tmp/get-pip.py
        echo "OK: pip 已通过 get-pip.py 安装"
    fi
fi
export PY="${PY_BIN}"

echo "=== [5/7] 检测/安装构建工具链（libclang for bindgen + dialog）==="
# Rust build script / maturin 需要 cc 链接器；binding 的 bindgen 需要 libclang。
if [ "$(id -u)" -eq 0 ]; then
    if command -v dnf >/dev/null 2>&1; then
        rpm -q clang-devel >/dev/null 2>&1 || dnf install -y clang-devel
        rpm -q libclang >/dev/null 2>&1 || dnf install -y libclang
        rpm -q dialog >/dev/null 2>&1 || dnf install -y dialog
    elif command -v apt-get >/dev/null 2>&1; then
        export DEBIAN_FRONTEND=noninteractive
        dpkg -s build-essential >/dev/null 2>&1 || apt-get install -y build-essential
        dpkg -s libclang-dev >/dev/null 2>&1 || apt-get install -y libclang-dev
        dpkg -s dialog >/dev/null 2>&1 || apt-get install -y dialog
    fi
else
    echo "非 root，跳过系统包安装（libclang/dialog 需预装）"
fi
# 检测 libclang（覆盖 Ubuntu/RHEL/CentOS 常见路径 + llvm 子目录）
LIBCLANG_DIR=""
for d in \
    /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib "${LOCAL_PREFIX}/lib" \
    /usr/lib/llvm-*/lib /usr/lib/llvm*/lib /usr/lib64/llvm*/lib \
    /usr/lib/clang*/lib /opt/rh/llvm*/root/usr/lib64 \
    /usr/local/lib /usr/local/lib64
do
    if ls "$d"/libclang.so >/dev/null 2>&1 || ls "$d"/libclang-*.so >/dev/null 2>&1; then
        LIBCLANG_DIR="$d"
        break
    fi
done
# fallback: find 全盘搜索（最后手段，慢但可靠）
if [ -z "${LIBCLANG_DIR}" ]; then
    _found=$(find / -name "libclang.so" -o -name "libclang-*.so" 2>/dev/null | head -1)
    if [ -n "${_found}" ]; then
        LIBCLANG_DIR="$(dirname "${_found}")"
    fi
fi
if [ -n "${LIBCLANG_DIR}" ]; then
    if [ "$(id -u)" -eq 0 ]; then
        echo "export LIBCLANG_PATH=${LIBCLANG_DIR}" > /etc/profile.d/libclang_path.sh
    fi
    export LIBCLANG_PATH="${LIBCLANG_DIR}"
    echo "OK: LIBCLANG_PATH=${LIBCLANG_DIR}"
else
    echo "WARN: 未找到 libclang，binding 构建可能失败"
fi
command -v cc >/dev/null 2>&1 && echo "OK: cc 链接器可用 ($(cc --version | head -1))" || echo "WARN: cc 不可用"

echo "=== [6/7] 检查 .cargo/config.toml 的 tokio_unstable cfg ==="
SRC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cfg="${SRC_ROOT}/.cargo/config.toml"
if grep -q 'tokio_unstable' "$cfg" 2>/dev/null; then
    echo "OK: ${cfg} 含 tokio_unstable cfg"
else
    echo "WARN: ${cfg} 未发现 tokio_unstable，dynamo-runtime 可能编译失败 (E0599)"
fi

echo
echo "=== 环境准备完成 ==="
echo "rustup/${RUST_TOOLCHAIN}、protoc、python${PY_VERSION} 已就绪"
echo "CC=${CC:-系统默认}  LIBCLANG_PATH=${LIBCLANG_DIR:-未设置}"
echo "后续运行测试请执行同目录的 run-all-tests-oneclick.sh"
