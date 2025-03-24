#!/usr/bin/env bash
set -eu

DIR_RAMDISK=/ramdisk
DIR_ZIG_CACHE_ROOT=${DIR_RAMDISK}/mlir_toy
CACHE_DIR_NAME=zig-cache
ARG_DIR_BUILD_CACHE=""
MODE=${1:-"Debug"}

# NOTE: remember to update these paths according to your case.
# It's fine to set both LLVM_DIR and MLIR_DIR below with the same path. We use
# both of them in case users are building with a pre-installed LLVM and a
# self-built MLIR.
# In $LLVM_DIR/lib, it should contain "libc++" and "libc++abi".
LLVM_DIR=~/workspace/tool/llvm-17/out/mlir
MLIR_DIR=~/workspace/tool/llvm-17/out/mlir

check_var() {
    local file_type=$1
    local var_name=$2
    local var_value=${!var_name}

    if [[ $file_type != "d" && $file_type != "f" ]]; then
        echo "[ERROR] Only these file types are available to check: d, f"
        exit 1
    fi

    if [[ -z $var_value ]]; then
        echo "[ERROR] Please specify a value for $var_name"
        exit 1
    fi
}

# ==============================================================================
# Do some checks before building

if [[ -z ${MODE} ]]; then
    echo "Please specify optimization mode to build"
    exit 1
fi

if [[ -d ${DIR_RAMDISK} ]] && [[ ! -z ${CACHE_DIR_NAME} ]]; then
    echo "[INFO] RAM disk is available, setting zig-cache to ${DIR_ZIG_CACHE_ROOT}/${CACHE_DIR_NAME}"
    mkdir -p ${DIR_ZIG_CACHE_ROOT}
    ARG_DIR_BUILD_CACHE=" --cache-dir ${DIR_ZIG_CACHE_ROOT}/${CACHE_DIR_NAME}"
    ARG_DIR_BUILD_CACHE+=" --global-cache-dir ${DIR_ZIG_CACHE_ROOT}/${CACHE_DIR_NAME}"
fi

check_var d LLVM_DIR
check_var d MLIR_DIR

# ==============================================================================
# Start building

# ## Build and run Ch1
# zig build ${ARG_DIR_BUILD_CACHE} -freference-trace -Doptimize=${MODE} -Dchapters=ch1
# ./zig-out/bin/toyc-ch1 ./toy_examples/Ch1/ast.toy --emit=ast

## Build and run other chapter
zig build ${ARG_DIR_BUILD_CACHE} -Doptimize=${MODE} -freference-trace \
    -Dllvm_dir=${LLVM_DIR} -Dmlir_dir=${MLIR_DIR} \
    -Dlink_mode=dynamic -Duse_custom_libcxx=false \
./zig-out/bin/toyc-ch2 ./toy_examples/Ch2/codegen.mlir --emit=mlir \
    --mlir-print-stacktrace-on-diagnostic=true \
    --mlir-print-op-on-diagnostic=true \
    --mlir-print-op-generic=false

# ## Build and run dialect sample
# zig build ${ARG_DIR_BUILD_CACHE} -Doptimize=${MODE} -freference-trace \
#     -Dllvm_dir=${LLVM_DIR} -Dmlir_dir=${MLIR_DIR} \
#     -Dchapters=sample -Dbuild_dialect=true
# ./zig-out/bin/sample

# ## Debug with lldb/gdb
# lldb -s .lldbinit -- ./zig-out/bin/toyc-ch1 ./toy_examples/Ch1/ast.toy --emit=ast
# gdb --args ./zig-out/bin/toyc-ch1 ./toy_examples/Ch1/ast.toy --emit=ast

echo "[INFO] Build finished"

# XXX: Sometimes zig compiler would create a cache directory here even we have
# set both local and global cache directory to other place, so we would remove
# it manually.
# (for Zig < 0.13.0, cache folder was named as "zig-cache")
if [[ ! -z ${ARG_DIR_BUILD_CACHE} ]] && [[ -d .zig-cache ]]; then
    ./scripts/remove_zig_cache.sh
fi

