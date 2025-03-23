# We don't have all required files as below to call `mlir_add_dialect()` (see
# also AddMLIR.cmake), so we do what it does manually:
# - ${dialect}{.h.inc|.cpp.inc}
# - ${dialect}Type{.h.inc|.cpp.inc}
# - ${dialect}Dialect{.h.inc|.cpp.inc}
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyCh3OpsIncGen)
add_dependencies(mlir-headers ToyCh3OpsIncGen)

set(LLVM_TARGET_DEFINITIONS ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh3CombineIncGen)
add_dependencies(mlir-headers ToyCh3CombineIncGen)

add_mlir_dialect_library(MLIRToy
    Dialect.cpp
    ToyCombine.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/toy/cpp

    DEPENDS
    ToyCh3OpsIncGen
    ToyCh3CombineIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
