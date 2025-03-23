#!/usr/bin/env bash
# ref: https://mlir.llvm.org/getting_started/
# ref: https://github.com/ziglang/zig-bootstrap/blob/0.13.0/build
set -eu -o pipefail

# Source directory of LLVM. Here we assume this script is put and run under
# the cloned "llvm-project" directory.
llvm_src_dir=.

build_root=${llvm_src_dir}/out
build_type=Release
enable_rtti=ON
build_mlir_c_dylib=ON

if [[ ! -f "$llvm_src_dir/llvm/CMakeLists.txt" ]]; then
    echo "Cannot find CMakeLists.txt under $llvm_src_dir/llvm/"
    exit 1
fi

mkdir -p $build_root

# ==============================================================================
# 1. Build libc++
# https://releases.llvm.org/17.0.1/projects/libcxx/docs/BuildingLibcxx.html
_build_dir=build_stage1
cmake \
    -G Ninja \
    "-S$llvm_src_dir/runtimes" \
    "-B$build_root/$_build_dir" \
    -DCMAKE_INSTALL_PREFIX="$build_root/stage1" \
    -DCMAKE_PREFIX_PATH="$build_root/" \
    -DCMAKE_BUILD_TYPE=$build_type \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_ASM_COMPILER=clang \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind"
cmake --build "$build_root/$_build_dir" --target install

# =============== Do some checks before proceeding to next stage ===============
stage1_dir=$(realpath "$build_root/stage1")

if [[ ! -d ${stage1_dir} ]]; then
    echo "[ERROR] Stage 1 output folder does not exist: ${stage1_dir}"
    exit 1
fi

# XXX: there is not extra folder named with `TARGET_PREFIX` generated if we
# build `libc++` without other projects enabled. (or maybe it's affected by
# `LLVM_TARGETS_TO_BUILD`?)
lib_paths=( "${stage1_dir}/lib" )
inc_paths=( "${stage1_dir}/include/c++/v1" )

cmake_lib_paths=$(IFS=';' ; echo "${lib_paths[*]}")
cmake_inc_paths=$(IFS=';' ; echo "${inc_paths[*]}")

cxx_flags=(
    "-stdlib=libc++"
    "-I ${inc_paths[0]}"
)
# XXX: we need to set RPATH to the libc++ folder built in previous stage,
# because some tools pre-built in this stage would link to it (e.g.,
# llvm-min-tblgen).
# https://github.com/llvm/llvm-project/issues/53561#issuecomment-1129301944
# XXX: for binaries of unit test (e.g., MLIRTableGenTests), they might not
# locate at "inst_dir/bin" which make it still unable to find the just-built
# libc++ from "$ORIGIN/../lib". So we need to hard-code the full path for it.
ld_flags=(
    "-L ${lib_paths[0]}"
    "-Wl,-rpath-link ${lib_paths[0]}"
    "-Wl,--rpath,${lib_paths[0]}"
    "-Wl,--disable-new-dtags"
    "-lc++"
    "-lc++abi"
)

cmake_cxx_flags=$(IFS=' ' ; echo "${cxx_flags[*]}")
cmake_ld_flags=$(IFS=' '; echo "${ld_flags[*]}")

# XXX: when building runtimes, we have to pass these settings/flags via
# `RUNTIMES_CMAKE_ARGS`.
rt_args=(
    "-DCMAKE_C_COMPILER=clang"
    "-DCMAKE_CXX_COMPILER=clang++"
    "-DCMAKE_ASM_COMPILER=clang"
    "-DCMAKE_CXX_FLAGS='${cmake_cxx_flags}'"
    "-DCMAKE_EXE_LINKER_FLAGS='${cmake_ld_flags}'"
    "-DCMAKE_SHARED_LINKER_FLAGS='${cmake_ld_flags}'"
    "-DCMAKE_MODULE_LINKER_FLAGS='${cmake_ld_flags}'"
    "-DCMAKE_LIBRARY_PATH='${cmake_lib_paths}'"
    "-DCMAKE_INCLUDE_PATH='${cmake_inc_paths}'"
)
rt_cmake_args=$(IFS=';'; echo "${rt_args[*]}")

# ==============================================================================
# 2. Build MLIR with libc++
# NOTE: OpenMP has dependencies to "clang" project, so we need to include
# "clang" in `LLVM_ENABLE_PROJECTS`.
_build_dir=build_mlir
cmake \
    -G Ninja \
    "-S$llvm_src_dir/llvm" \
    "-B$build_root/$_build_dir" \
    -DCMAKE_INSTALL_PREFIX="$build_root/mlir" \
    -DCMAKE_PREFIX_PATH="$build_root/stage1" \
    -DCMAKE_BUILD_TYPE=$build_type \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_ASM_COMPILER=clang \
    -DCMAKE_CXX_FLAGS="${cmake_cxx_flags}" \
    -DCMAKE_EXE_LINKER_FLAGS="${cmake_ld_flags}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${cmake_ld_flags}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${cmake_ld_flags}" \
    -DCMAKE_LIBRARY_PATH="${cmake_lib_paths}" \
    -DCMAKE_INCLUDE_PATH="${cmake_inc_paths}" \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_ENABLE_PROJECTS="clang;llvm;mlir" \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="Native" \
    -DLLVM_ENABLE_RTTI=$enable_rtti \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_LIBCXX=ON \
    -DLLVM_CCACHE_BUILD=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DRUNTIMES_CMAKE_ARGS="${rt_cmake_args}" \
    -DMLIR_BUILD_MLIR_C_DYLIB=$build_mlir_c_dylib \
    -DLLVM_ENABLE_RUNTIMES="openmp"

lit_args=(
    # Exclude this test since we haven't figure out how to pass those flags to
    # link against libc++ to it properly.
    "--filter-out='Examples/standalone/test.toy'"
)
llvm_lit_args=$(IFS=' '; echo "${lit_args[*]}")
LIT_OPTS="${llvm_lit_args}" \
    cmake --build "$build_root/$_build_dir" --target install check-mlir

# ==============================================================================
# Last step: copy libc++ files from stage1 folder to the final installation folder
pushd $build_root > /dev/null
cp -r stage1/* mlir
popd $build_root > /dev/null
