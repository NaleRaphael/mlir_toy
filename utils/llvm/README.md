# Build MLIR and related tools
## Preface
We are going to build MLIR with pre-installed Clang instead of bootstraping all
of them from scratch.

And these scripts are just made for convenience to run under the LLVM source
directory. You can run them after Clang and other requirements are installed,
or follow the instructions in section `Steps` below.
- `build_mlir.sh`: It's written for my own use case (with OpenMP support,
    target to both CPU and nvidia GPU). If you just want a minimal bulid,
    please refer to the instructions below.
- `build_filecheck.sh`: Build `FileCheck` without extra CMake targets.

## Steps
### Prepare Clang and clone LLVM repository
1. Install Clang-17 and related tools
    ```bash
    $ curl -L -o llvm.sh https://apt.llvm.org/llvm.sh
    $ chmod +x llvm.sh
    $ sudo ./llvm.sh 17
    $ rm llvm.sh

    # (optional) use the following script to set default symlinks of Clang tools
    # src: https://gist.github.com/junkdog/70231d6953592cd6f27def59fe19e50d
    $ ./update-alternatives-clang.sh 17 100

    # (optional) install ninja-build and ccache
    $ sudo apt-get install ninja-build ccache

    # (optional) config ccache to store caches on RAM disk
    # Run `ccache -s -v` or `ccache -p` to checkout the updated settings
    $ ccache -o cache_dir=/ramdisk/ccache
    $ ccache -o temporary_dir=/ramdisk/ccache-tmp
    $ ccache -o max_size=4G
    ```
2. Clone llvm-project
    ```bash
    $ git clone https://github.com/llvm/llvm-project.git --depth 1 --single-branch --branch release/17.x
    ```

### Build MLIR and FileCheck
- Build MLIR (**without** OpenMP and nvidia GPU target)
    ```bash
    # ref: https://mlir.llvm.org/getting_started/
    $ cd llvm-project

    # Definitions
    # - `llvm_src_dir`: source directory of llvm-project
    # - `build_dir`: build directory for generated makefiles, cmakefiles, ... etc
    # - `install_dir`: root directory for files to install

    # Recommended build options:
    # - Set `BUILD_SHARED_LIBS=ON` to reduce binary size
    #   - MLIR only: build/install: 32.4G + 25.6G -> 4.4G + 1.5G
    #   - LLVM + Clang + MLIR + OpenMP
    # - Set `LLVM_INSTALL_UTILS=ON` to build and install utils (FileCheck, not,
    #   ...) to $install_dir.
    # - Set `LLVM_ENABLE_RUNTIMES="openmp"`

    # Options that I've tried for resolving compilation issues:
    # - Set `LLVM_ENABLE_RTTI=ON` to avoid linker errors as below while building
    #   a C bindings for dialect:
    #   > symbol lookup error: .../BINDINGS.so: undefined symbol: _ZTIN4mlir7DialectE
    #   See also: https://discourse.llvm.org/t/undefined-reference-to-typeinfo-for-llvm-genericoptionvalue/71526

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
    $ cmake \
        -G Ninja \
        "-S$llvm_src_dir/llvm" \
        "-B$build_dir" \
        -DCMAKE_INSTALL_PREFIX=$install_dir \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DLLVM_TARGETS_TO_BUILD="Native" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_CCACHE_BUILD=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_INSTALL_UTILS=ON

    # NOTE: Remove `install` from target list if you just want to build examples
    $ cmake --build "$build_dir" --target install check-mlir
    ```

### Optional
- Build `FileCheck` only (if you didn't build LLVM with `LLVM_INSTALL_UTILS=ON`)
    ```bash
    $ cmake \
        -G Ninja \
        "-S$llvm_src_dir/llvm" \
        "-B$build_dir" \
        -DCMAKE_INSTALL_PREFIX=$install_dir \
        -DLLVM_ENABLE_PROJECTS="clang;llvm" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
        -DLLVM_CCACHE_BUILD=ON \
        -DBUILD_SHARED_LIBS=ON

    # Build `FileCheck`
    $ cmake --build "$build_dir" --target FileCheck

    # -------------------------------------------------------------------------
    # NOTE: The installation config for `FileCheck` is not complete, so we cannot
    # install it by running `$ cmake -DCOMPONENT=FileCheck -P cmake_install.cmake`.
    # (setting `COMPONENT` to "LLVMFileCheck" would install "libFileCheck.so" only)
    # So we have to copy required files manually as below.

    # 1. Copy the binary `FileCheck`
    $ cp -av $build_dir/bin $install_dir

    # 2. Copy shared libraries required by `FileCheck`
    $ mkdir -p $install_dir/lib
    $ cp -av $build_dir/lib/*.so* $install_dir/lib
    ```
