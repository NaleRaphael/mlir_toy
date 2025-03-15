## An example of building C bindings for MLIR dialect
- This example is not included as a chapter of the root repository. To build
    this example, please add `sample` to the `Chapter` enum and
    update the branch of `sample` in `getChapterBuildFn()` accordingly.

## CMake related notes
- In the official MLIR example "standalone", it would fail to build as a shared
    library (by adding `-DBUILD_SHARED_LIBS=ON`). Related discussions:
    - https://discourse.llvm.org/t/shared-library-support/381
    - https://github.com/llvm/llvm-project/issues/108253
    - https://github.com/j2kun/mlir-tutorial/issues/33

### `AddMLIR.cmake`
- `add_mlir_library`: core function to create a MLIR library
- `add_mlir_dialect_library`: to build a dialect
- `add_mlir_public_c_api_library`: to build C bindings of a dialect
    - Call `add_mlir_library` with custom args:
        - `ENABLE_AGGREGATION`: force generation of an object library, export metadata, install additional object files to include given target as a part of an aggregated shared library.
        - `EXCLUDE_FROM_LIBMLIR`: don't include given lib in libMLIR.so. (see also the docstring of `add_mlir_library`)
        - `ADDITIONAL_HEADER_DIRS`: `${MLIR_MAIN_INCLUDE_DIR}/mlir-c` (MLIR C-API header dirs)
    - Set property:
        - `CXX_VISIBILITY_PRESET`: hidden (not to export all symbols by default)
            - https://gcc.gnu.org/wiki/Visibility
    - Add definitions:
        - `-DMLIR_CAPI_BUILDING_LIBRARY=1`
- `add_mlir_aggregate`
    - Call `add_mlir_library` with custom args:
        - `PARTIAL_SOURCES_INTENDED`
        - `EXCLUDE_FROM_LIBMLIR`
            - Private link libs
    - Call `target_sources`: Specifies sources to use when building a target and/or its dependents
    - Disallow undefined symbols (call `target_link_option` with "LINKER:-z,defs")

