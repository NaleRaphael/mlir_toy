#!/usr/bin/env bash
# ref: https://mlir.llvm.org/getting_started/
# ref: https://github.com/iml130/mlir-emitc/blob/main/build_tools/build_mlir.sh
set -e

# Source directory of LLVM. Here we assume this script is put and run under
# the cloned "llvm-project" directory.
llvm_src_dir=.
# Build directory (for generated makefiles, cmake files, and build artifacts)
build_dir=build_filecheck
install_dir=llvm_filecheck

if [[ ! -f "$llvm_src_dir/llvm/CMakeLists.txt" ]]; then
    echo "Cannot find CMakeLists.txt under $llvm_src_dir/llvm/"
    exit 1
fi

mkdir -p $build_dir
mkdir -p $install_dir

cmake \
    -G Ninja \
    "-S$llvm_src_dir/llvm" \
    "-B$build_dir" \
    -DCMAKE_INSTALL_PREFIX=$install_dir \
    -DLLVM_ENABLE_PROJECTS="clang;llvm" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_CCACHE_BUILD=ON \
    -DBUILD_SHARED_LIBS=ON

# NOTE: we don't include `install` target here to avoid building the whole
# Clang and LLVM.
cmake --build "$build_dir" --target FileCheck

# NOTE: after running this script, we have to copy files related to FileCheck
# manually since we didn't include `install` target above.
# - $build_dir/bin/FileCheck
# - $build_dir/lib/libFileCheck.*
cp -av $build_dir/bin $install_dir
mkdir -p $install_dir/lib
cp -av $build_dir/lib/*.so* $install_dir/lib
