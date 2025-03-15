#!/usr/bin/env bash
set -eu

DIR_RAMDISK=/ramdisk
DIR_ZIG_CACHE_ROOT=${DIR_RAMDISK}/mlir_toy_mlir_capi_test
CACHE_DIR_NAME=zig-cache
ARG_DIR_BUILD_CACHE=""
MODE=${1:-"Debug"}

if [[ -z ${MODE} ]]; then
    echo "Please specify optimization mode to build"
    exit 1
fi

LLVM_DIR=~/workspace/tool/llvm-17/llvm_mlir_reldeb_rtti
MLIR_DIR=~/workspace/tool/llvm-17/llvm_mlir_reldeb_rtti

# (optional) It's recommended to supply it in order to run a complete test
FILECHECK_BIN=~/workspace/tool/llvm-17/llvm_mlir_reldeb_rtti/bin/FileCheck

# (optional) Extra options for FileCheck, see also:
# https://llvm.org/docs/CommandGuide/FileCheck.html
#
# Example:
# FILECHECK_OPTS='--dump-input=fail --match-full-lines'
#
# NOTE: Please don't specify `--check-prefix`. Because we are using different
# prefix per case to make us able to check each single test case without being
# bothered by the ordering.
FILECHECK_OPTS=''

check_path() {
    local file_type=$1
    local var_name=$2
    local var_value=${!var_name}

    if [[ ${file_type} != "d" && ${file_type} != "f" ]]; then
        echo "[ERROR] Only these file types are available to check: d, f"
        exit 1
    fi

    if [[ -z ${var_value} ]]; then
        echo "[ERROR] Please specify a value for ${var_name}"
        exit 1
    fi

    if [[ ${file_type} == "d" ]] && [[ ! -d ${var_value} ]]; then
        echo "[ERROR] directory ${var_name} does not exist, value: ${var_value}"
        exit 1
    elif [[ ${file_type} == "f" ]] && [[ ! -f ${var_value} ]]; then
        echo "[ERROR] file ${var_name} does not exist, value: ${var_value}"
        exit 1
    fi
}

check_path d LLVM_DIR
check_path d MLIR_DIR

# Run unit tests
env FILECHECK_BIN="$FILECHECK_BIN" FILECHECK_OPTS="$FILECHECK_OPTS" \
zig build ${ARG_DIR_BUILD_CACHE} -Doptimize=${MODE} \
    -Dllvm_dir=${LLVM_DIR} -Dmlir_dir=${MLIR_DIR} \
    test --summary all

echo "[INFO] Build finished"
