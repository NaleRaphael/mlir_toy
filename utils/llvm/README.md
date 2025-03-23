# Build MLIR and related tools
## Preface
We are going to build MLIR with pre-installed Clang instead of bootstraping all
of them from scratch.

And these scripts are just made for convenience to run under the LLVM source
directory. You can run them after Clang and other requirements are installed,
or follow the instructions in section `Steps` below.
- `build_llvm_mlir.sh`: Build LLVM/MLIR with libc++, it's recommended to run
    this to make sure those toy dialect libraries we are going to build are
    compatible with Zig's `libc++`.
- `build_mlir.sh`: It's written for my own use case (with OpenMP support,
    target to both CPU and nvidia GPU). If you just want a custom bulid,
    please change any setting in it based on your case.
- `build_filecheck.sh`: Build `FileCheck` without extra CMake targets.

## Steps
### Prepare Clang and clone LLVM repository
1. Install Clang-17 and related tools
    ```bash
    $ curl -L -o llvm.sh https://apt.llvm.org/llvm.sh
    $ chmod +x llvm.sh
    $ sudo ./llvm.sh 17
    $ rm llvm.sh

    # Install ninja-build
    $ sudo apt-get install ninja-build

    # (optional) use the following script to set default symlinks of Clang tools
    # src: https://gist.github.com/junkdog/70231d6953592cd6f27def59fe19e50d
    $ ./update-alternatives-clang.sh 17 100

    # (optional) install and config ccache
    # Run `ccache -s -v` or `ccache -p` to checkout the updated settings
    $ sudo apt-get install ccache
    $ ccache -o cache_dir=/ramdisk/ccache
    $ ccache -o temporary_dir=/ramdisk/ccache-tmp
    $ ccache -o max_size=4G
    ```
2. Clone llvm-project
    ```bash
    $ git clone https://github.com/llvm/llvm-project.git --depth 1 --single-branch --branch release/17.x
    ```

### Build LLVM/MLIR
- It's recommended to built with the `libc++` to make sure it's compatible with
  Zig's `libc++`:
    ```bash
    # `LLVM_SRC` is the LLVM source folder you just cloned
    $ cp build_llvm_mlir.sh $LLVM_SRC/

    $ cd $LLVM_SRC
    $ ./build_llvm_mlir.sh

    # MLIR will be installed in $LLVM_SRC/out/mlir.
    ```
    > [!NOTE]  
    > There is a test case excluded while building MLIR because we have no idea
    > how to pass linker flags properly to make it link against the just-built
    > `libc++`.
    > If you want to make sure it can run successfully, try run the script
    > `run_mlir_standalone_test.sh` in this folder after LLVM and MLIR is built.
- You can just build MLIR if:
    1. You have a pre-installed LLVM with `libc++` in system paths.
    2. You want to try building MLIR and all chapters in this repository with
        GCC, which would link against `libstdc++`. (but you need to change
        related settings in `build.zig` and `build_dialect.sh` in all chapters.
        See also [this thread][1] for potential workaround to make Zig link to
        `libstdc++` before the official support is landed.)
    ```bash
    $ cp build_mlir.sh $LLVM_SRC/

    $ cd $LLVM_SRC

    # Note that `LLVM_ENABLE_LIBCXX=ON` is not set in this script
    $ ./build_mlir.sh

    # MLIR will be installed in $LLVM_SRC/out/mlir.
    ```

[1]: https://github.com/ziglang/zig/issues/3936

