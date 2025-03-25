#!/usr/bin/env bash
set -eu -o pipefail

# NOTE: it's fine to set both LLVM_DIR and MLIR_DIR below with the same path.
# We use both of them in case users are building with a pre-installed LLVM
# and a self-built MLIR.
# See also link below to know how cmake searches the module for configuration.
# https://cmake.org/cmake/help/latest/command/find_package.html#search-procedure
DEFAULT_LLVM_DIR=~/workspace/tool/llvm-17/out/mlir
DEFAULT_MLIR_DIR=~/workspace/tool/llvm-17/out/mlir
DEFAULT_BUILD_SHARED=0

# Root directory of LLVM (to search "LLVMConfig.cmake")
# - We assume that libc++ is installed under this directory.
LLVM_DIR=${LLVM_DIR:-"${DEFAULT_LLVM_DIR}"}
# Root directory of MLIR (to search "MLIRConfig.cmake")
MLIR_DIR=${MLIR_DIR:-"${DEFAULT_MLIR_DIR}"}
# Build this Toy dialect as shared library or not
BUILD_SHARED=${BUILD_SHARED:-"${BUILD_SHARED}"}

# ==============================================================================
# Prepare paths and flags
# - To maximize the compatibility to Zig, we would compile this dialect library
#   with `libc++` rather than `libstdc++`.
lib_paths=(
    "${LLVM_DIR}/lib"
    "${MLIR_DIR}/lib"
)
inc_paths=(
    "${LLVM_DIR}/include/c++/v1"
    "${MLIR_DIR}/include"
)

cmake_lib_paths=$(IFS=';' ; echo "${lib_paths[*]}")
cmake_inc_paths=$(IFS=';' ; echo "${inc_paths[*]}")

cxx_flags=(
    "-stdlib=libc++"
    "-I ${LLVM_DIR}/include/c++/v1"
)
ld_flags=(
    "-L ${lib_paths[0]}"
    "-Wl,-rpath-link ${LLVM_DIR}/lib"
    "-lc++"
    "-lc++abi"
)

cmake_cxx_flags=$(IFS=' ' ; echo "${cxx_flags[*]}")
cmake_ld_flags=$(IFS=' '; echo "${ld_flags[*]}")

# For CMAKE_PREFIX_PATH
prefix_paths=(
    ${LLVM_DIR}
    ${MLIR_DIR}
)
cmake_prefix_path=$(IFS=';' ; echo "${prefix_paths[*]}")

# XXX: It's possible to build dialect library as a shared library now, but we
# need to:
# - add `$MLIR_DIR/lib` to `CMAKE_INSTALL_RPATH`
# - make sure all required MLIR libraries to link are added in
#   `add_mlir_dialect_library()`
# - in build.zig, mark the linkage to `libMLIRToy` as needed to avoid it being
#   omitted.
build_shared_lib=${BUILD_SHARED}

install_rpath=( "\$ORIGIN/../lib" )
if [[ ${build_shared_lib} -eq 1 ]]; then
    install_rpath+=( "${LLVM_DIR}/lib" )
    install_rpath+=( "${MLIR_DIR}/lib" )
fi

cmake_install_rpath=$(IFS=';' ; echo "${install_rpath[*]}")

build_dir=build_toy
install_dir=inst_toy

# ==============================================================================
# Further checks before building
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

check_var d LLVM_DIR
check_var d MLIR_DIR

# ==============================================================================
# Start building

if [[ ! -d ${build_dir} ]]; then
    mkdir -p ${install_dir}
    mkdir -p ${build_dir}

    cmake -G Ninja \
        -S./ \
        -B$build_dir \
        -DCMAKE_INSTALL_PREFIX=$install_dir \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=$cmake_prefix_path \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_FLAGS="${cmake_cxx_flags}" \
        -DCMAKE_EXE_LINKER_FLAGS="${cmake_ld_flags}" \
        -DCMAKE_SHARED_LINKER_FLAGS="${cmake_ld_flags}" \
        -DCMAKE_MODULE_LINKER_FLAGS="${cmake_ld_flags}" \
        -DCMAKE_LIBRARY_PATH="${cmake_lib_paths}" \
        -DCMAKE_INCLUDE_PATH="${cmake_inc_paths}" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DCMAKE_INSTALL_RPATH="${cmake_install_rpath}" \
        -DBUILD_SHARED_LIBS=$build_shared_lib
fi

cmake --build $build_dir --parallel --target install

