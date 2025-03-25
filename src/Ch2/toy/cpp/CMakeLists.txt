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
add_public_tablegen_target(ToyCh2OpsIncGen)
add_dependencies(mlir-headers ToyCh2OpsIncGen)

add_mlir_dialect_library(MLIRToy
    Dialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/toy/cpp

    DEPENDS
    ToyCh2OpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRParser
)
