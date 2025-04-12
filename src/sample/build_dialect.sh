#!/usr/bin/env bash
set -eu -o pipefail

DEFAULT_MLIR_DIR=/usr/lib/llvm-17

# The directory of MLIR cmake modules (to search "AddMLIR.cmake")
MLIR_DIR=${MLIR_DIR:-"${DEFAULT_MLIR_DIR}"}

# XXX: We cannot build the dialect as shared library for now, we might need to
# come back to fix it after a stable release of LLVM/MLIR is available.
# https://github.com/llvm/llvm-project/issues/108253
build_shared_lib=OFF

build_dir=build_sample
install_dir=inst_sample

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

    if [[ $file_type == "d" ]]; then
        if [[ ! -d $var_value ]]; then
            echo "[ERROR] directory does not exist: $var_value"
            exit 1
        fi
    elif [[ $file_type == "f" ]]; then
        if [[ ! -f $var_value ]]; then
            echo "[ERROR] file does not exist: $var_value"
            exit 1
        fi
    fi
}

check_var d MLIR_DIR

# For CMAKE_PREFIX_PATH
prefix_paths=()
prefix_paths+=($MLIR_DIR)
agg_prefix_paths=$(IFS=';' ; echo ${prefix_paths[*]})

# ---------- Building procedure ----------
rm -rf ${build_dir}
rm -rf ${install_dir}

mkdir -p ${install_dir}
mkdir -p ${build_dir}

cmake -G Ninja \
    -S./ \
    -B$build_dir \
    -DCMAKE_INSTALL_PREFIX=$install_dir \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$agg_prefix_paths \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DBUILD_SHARED_LIBS=$build_shared_lib

cmake --build $build_dir --target install

