# mlir_toy
Learn MLIR the hard way (probably) with Zig.

- Reimplement things in Zig if it's possible.
- Use Zig build system to replace CMake if it's possible.
    - Currently it's hard to figure out a way to build dialect and it's C-API
    without using MLIR's CMake module (`AddMLIR.cmake`), so we wrote a script
    `build_dialect.sh` and integrate it into `build.zig` as a workaround. For
    further details related to this work, please check out `src/sample`.
- Minimize dependencies of LLVM internal libraries/tools.

I believe it's a way to make me learn more and gain a solid understanding of
LLVM/MLIR internals.


## Prerequisites
- Zig 0.13.0
- LLVM 17 (with `libc++`)


## Build
### LLVM/MLIR
Please check out [utils/llvm/README.md](./utils/llvm/README.md).

### Zig
```bash
# Basic usage: build with specific chapter
$ zig build -Dllvm_dir=${LLVM_DIR} -Dmlir_dir=${MLIR_DIR} -Dchapters=${CHAPTER}
```
Please check out [build.sh](./build.sh) for more details.


## Run and compare the output
```bash
# - Run the output with data provided in MLIR toy example
$ ./zig-out/bin/toyc-chX ./toy_example/ChX/XXX [OPTIONS}

# - Or compare the output with the binary compiled from official toy example
#   (remember to update the variables in this script based on your case)
$ ./compare_output.sh
```

> [!NOTE]  
> It's recommended to create symlinks of compiled binaries of MLIR toy example
> and data to this directory. Or you might need to change the default paths set
> in `compare_output.sh` accordingly.
```bash
# Assume that you cloned LLVM source repo to "~/workspace/tool/llvm-17", and
# ran the script "./utils/llvm/build_mlir.sh" to build MLIR.
$ LLVM_ROOT_DIR=~/workspace/tool/llvm-17
$ MLIR_BUILD_DIR=${LLVM_SRC_DIR}/build_reldeb_rtti

$ ln -s ${MLIR_BUILD_DIR}/bin toy_bin
$ ln -s ${LLVM_ROOT_DIR}/mlir/test/Examples/Toy toy_examples
```


## Tests
### Port of tests for MLIR C-API
```bash
$ cd tests/mlir

# Remember to update the variables `LLVM_DIR`, `MLIR_DIR`, `FILECHECK_BIN`
$ ./run_tests.sh
```

### Compare the output with C++ implementation for a chapter
```bash
# Usage: ./tests/compare_toyc_output.sh <ChN>
# - ChN: chapter number (0 ~ 7)
#
# Environment variables:
# - VERBOSE: (0 or 1) show details when running test
# - ENABLE_DEBUG: (0 or 1) add debug options to CLI when running toyc binaries

# Example: compare the outputs of C++ and our Zig implementation for Ch7, and
# show the details while running.
$ VERBOSE=1 ./tests/compare_toyc_output.sh 7
```


## Tips
- If you ran into problems while compiling with C libraries, try adding flags
  `--verbose-cc` and `--verbose-cimport` to get details for debugging.
- Test cases in `tests/mlir/CAPI` are also resources to learn how to use MLIR
  C-API in Zig.

