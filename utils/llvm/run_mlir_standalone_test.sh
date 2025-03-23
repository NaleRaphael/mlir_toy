#!/usr/bin/env bash
# This script is trying to run the test case as follows to prove that it can
# pass if it's built with libc++.
# - ${LLVM_SRC}/mlir/test/Examples/standalone/test.toy
#
# This script requires MLIR to be built already.
set -eu

# NOTE: please update these paths according to your case.
LLVM_SRC=~/workspace/tool/llvm-17
LLVM_DIR=~/workspace/tool/llvm-17/out/mlir
MLIR_DIR=~/workspace/tool/llvm-17/out/mlir

# The build tree of MLIR
MLIR_BLD_DIR=~/workspace/tool/llvm-17/out/build_mlir

PY3_BIN=$(which python3)

if [[ -z ${PY3_BIN} ]]; then
    echo "Cannot find Python 3"
    exit 1
fi

build_dir=build_mlir_standalone_test

# =============================== required flags ===============================
lib_paths=( "${LLVM_DIR}/lib" )
inc_paths=( "${LLVM_DIR}/include/c++/v1" )

cmake_lib_paths=$(IFS=';' ; echo "${lib_paths[*]}")
cmake_inc_paths=$(IFS=';' ; echo "${inc_paths[*]}")

cxx_flags=(
    "-stdlib=libc++"
    "-I ${inc_paths[0]}"
)
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

lit_args=(
    "-v"
    "--show-unsupported"
)
llvm_lit_args=$(IFS=' '; echo "${lit_args[*]}")

# =============================== start building ===============================
# Clear previous build tree
rm -rf $build_dir

# NOTE: we have to set compiler/linker flags to make sure it can link against
# the libc++ built by ourselves.
cmake "${LLVM_SRC}/mlir/examples/standalone" \
    -G "Ninja" \
    -B $build_dir \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_FLAGS="${cmake_cxx_flags}" \
    -DCMAKE_EXE_LINKER_FLAGS="${cmake_ld_flags}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${cmake_ld_flags}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${cmake_ld_flags}" \
    -DCMAKE_LIBRARY_PATH="${cmake_lib_paths}" \
    -DCMAKE_INCLUDE_PATH="${cmake_inc_paths}" \
    -DLLVM_ENABLE_LIBCXX=ON \
    -DMLIR_DIR=${MLIR_BLD_DIR}/lib/cmake/mlir \
    -DLLVM_USE_LINKER=lld \
    -DPython3_EXECUTABLE="${PY3_BIN}"

# NOTE: we have to use process substitution when running FileCheck to show the
# progress of building, but it's fine to run without it (as how it's written in
# "Example/standalone/test.toy").
LIT_OPTS="${llvm_lit_args}" \
    cmake --build $build_dir --target check-standalone | \
    tee ./test.toy.tmp | \
    tee >(${MLIR_BLD_DIR}/bin/FileCheck ${LLVM_SRC}/mlir/test/Examples/standalone/test.toy)

