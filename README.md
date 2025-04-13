# mlir_toy
Learn MLIR the hard way (probably) with Zig.

- Reimplement things in Zig if it's possible.
- Use Zig build system to replace CMake if it's possible.
- Minimize dependencies of LLVM internal libraries/tools.

I believe it's a way to make me learn more and gain a solid understanding of
LLVM/MLIR internals.


## Prerequisites
- Zig 0.13.0
- LLVM 17


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
# ran the script "./utils/llvm/build_llvm_mlir.sh" to build MLIR.
$ LLVM_ROOT_DIR=~/workspace/tool/llvm-17
$ MLIR_INST_DIR=${LLVM_SRC_DIR}/out/mlir

$ ln -s ${MLIR_INST_DIR}/examples toy_bin
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

> [!NOTE]  
> When comparing the output for Ch7, the test would fail when `ENABLE_DEBUG=1`
> because there is an extra canonicalizer pass added to the pipeline (see
> [here][mlir_17_extra_pass]). It's considered as a redundant pass but it would
> show the IR before & after processing because `--mlir-print-ir-before-all`
> and `--mlir-print-ir-after-all` is enabled. So you can safely comment these
> 2 options out if you want to run the script with `ENABLE_DEBUG=1` for Ch7.


## Current Limitations and Workarounds
Here are things that cannot be done currently as initially planned, I would
come back to them if possible.

### 1. Build MLIR dialect library with Zig
The build of a MLIR dialect library heavily depends on things defined in CMake
modules like [`AddMLIR.cmake`][gh_addmlir] and [`AddLLVM.cmake`][gh_addllvm].
Though I think it's possible to replicate those macros without using CMake,
it's really time-consuming and it requires to maintain if I want to switch to
different version of LLVM/MLIR.

But the major barrier stops me working on this for now is the lack of supports
for some compiler & linker features in `zig cc/c++`, e.g., [unsupported
linker arg: -rpath-link][zig-issue-18713]. So my current workaround for this is
to use CMake within a build script `build_dialect.sh` and integrate this step
in `build.zig`. You can check this out in each chapter folder to know how it
works, and see also ["src/sample/"](./src/sample) for some notes and code I've
done while digging into this topic.

### 2. Implement dialect Ops with current MLIR C-API
If I understand correctly, currently it's not possible to implement Ops of
custom dialect without directly writing C++. Because they rely on C++'s CRTP
for type traits and supports of some internal features like IR verification.
So that's why we still keep some C++ implementations in "ChX/toy/cpp" folder.

Regarding passes, it's possible to work with zig and MLIR C-API directly. You
can check out how it work in ["tests/mlir/CAPI/pass.zig"](./tests/mlir/CAPI/pass.zig).

As for other things that're not support in current MLIR C-API, we have to extend
it by ourselves. See also "ChX/toy/c" folder for details.


## Tips
- If you ran into problems while compiling with C libraries, try adding flags
  `--verbose-cc` and `--verbose-cimport` to get details for debugging.
- Documents of C-APIs are mostly available in `$LLVM_SRC/mlir/include/mlir-c`,
  but some C-APIs are defined in "Passes.td" files and they will be generated
  as "*.capi.h" and "*.capi.cpp" files after being built.
- Test cases in `tests/mlir/CAPI` are also resources to learn how to use MLIR
  C-API in Zig.


## Postscripts
Honestly, I didn't even know it's possible to complete this tutorial using Zig
and the MLIR C-API at first. But I just want to push my limits and see what I
could achieve, even though it's my first time building something related to a
compiler.

Using Zig forced me to learn more low-level details. It made me struggle at
times and exposed things I hadn't considered before, such as polymorphism via
vtable/static dispatch and memory management while handling errors.

Working with the MLIR C-API was also challenging, it pushed me to understand
how specific steps work behind the scenes in MLIR when I couldn't directly
manipulate data in C++.

And issues related to the build system (like [issue #1][this-issue-1]) also
forced me to dive deeper into understanding how compilers, linkers and CMake
work when building ELFs.

But I really enjoyed it because I did learn a lot in this process.

Now this project is almost finished, though there are still plenty things to
learn to build the thing I want with MLIR. Working on this project has sparked
my interest in exploring other techs for building compilers without relying on
LLVM/MLIR. For instance, I really want to see how Zig compiler would evolve
after the [announcement][zig-issue-16270] about removing LLVM-related
dependencies.

So, if you're looking to learn MLIR in an unconventional way, hope this project
would be helpful to you as well.

And here are more things that might interest you about Zig:
- [Andrew Kelley's talk about Data Oriented Design and Zig compiler][yt_andrew_dod]
- [Mitchell Hashimoto's blog series about Zig compiler][mitchell_blog_zig]
- [Andrew Kelley's talk about Zig build system][yt_andrew_build_system]
- [Validark's project to make Zig parser faster][gh_acc_zig_parser]


<!-- links -->
[gh_addmlir]: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/cmake/modules/AddMLIR.cmake
[gh_addllvm]: https://github.com/llvm/llvm-project/blob/release/17.x/llvm/cmake/modules/AddLLVM.cmake
[zig-issue-18713]: https://github.com/ziglang/zig/issues/18713
[zig-issue-16270]: https://github.com/ziglang/zig/issues/16270
[mlir_17_extra_pass]: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/examples/toy/Ch7/toyc.cpp#L154
[this-issue-1]: https://github.com/NaleRaphael/mlir_toy/issues/1
[yt_andrew_dod]: https://www.youtube.com/watch?v=IroPQ150F6c
[mitchell_blog_zig]: https://mitchellh.com/zig
[yt_andrew_build_system]: https://www.youtube.com/watch?v=wFlyUzUVFhw
[gh_acc_zig_parser]: https://github.com/Validark/Accelerated-Zig-Parser
