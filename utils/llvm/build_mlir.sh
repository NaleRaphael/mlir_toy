#!/usr/bin/env bash
# ref: https://mlir.llvm.org/getting_started/
# ref: https://github.com/iml130/mlir-emitc/blob/main/build_tools/build_mlir.sh
set -e

build_type=Release

# Source directory of LLVM. Here we assume this script is put and run under
# the cloned "llvm-project" directory.
llvm_src_dir=.
# Build directory (for generated makefiles, cmake files, and build artifacts)
build_root=${llvm_src_dir}/out
build_dir=build_mlir
install_dir=mlir

if [[ ! -f "$llvm_src_dir/llvm/CMakeLists.txt" ]]; then
    echo "Cannot find CMakeLists.txt under $llvm_src_dir/llvm/"
    exit 1
fi

mkdir -p $build_root

# Note for building with sanitizers:
# In my case (LLVM/Clang-17), building MLIR with address sanitizers would
# result in symbol lookup error as below, so I have to disable it for now.
# > libLLVMDemangle.so.17: undefined symbol: __asan_option_detect_stack_use_after_return
#
# Otherwise, to enable sanitizers, we have to build project "compiler-rt"
# as well, and add the following arguments:
# - `-DCOMPILER_RT_SANITIZERS_TO_BUILD=asan`
#    (other available sanitizers: msan, tsan, safestack, sfi, esan)
# - `-DCOMPILER_RT_BUILD_SANITIZERS=ON`
# - `-DLLVM_USE_SANITIZER="Address;Undefined"`

# Also, when building with LLVM_ENABLE_RUNTIMES="openmp", OpenMP should not be
# enabled in LLVM_ENABLE_PROJECTS.
# ref: https://openmp.llvm.org/SupportAndFAQ.html#q-how-to-build-an-openmp-gpu-offload-capable-compiler

cmake \
    -G Ninja \
    "-S$llvm_src_dir/llvm" \
    "-B$build_root/$build_dir" \
    -DCMAKE_INSTALL_PREFIX="$build_root/$install_dir" \
    -DCMAKE_BUILD_TYPE=$build_type \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_ASM_COMPILER=clang \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_ENABLE_PROJECTS="llvm;clang;mlir" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_CCACHE_BUILD=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_RUNTIMES="openmp" \
    -DBUILD_SHARED_LIBS=ON

# Build with OpenMP support (required by ExecutionEngineDump)
cmake --build "$build_root/$build_dir" --target install check-mlir

