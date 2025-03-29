# We don't have all required files as below to call `add_mlir_dialect()` (see
# also AddMLIR.cmake), so we do what it does manually:
# - ${dialect}{.h.inc|.cpp.inc}
# - ${dialect}Type{.h.inc|.cpp.inc}
# - ${dialect}Dialect{.h.inc|.cpp.inc}
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyCh4OpsIncGen)
add_dependencies(mlir-headers ToyCh4OpsIncGen)

set(LLVM_TARGET_DEFINITIONS ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh4CombineIncGen)
add_dependencies(mlir-headers ToyCh4CombineIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyCh4ShapeInferenceInterfaceIncGen)
add_dependencies(mlir-headers ToyCh4ShapeInferenceInterfaceIncGen)

add_mlir_dialect_library(MLIRToy
    Dialect.cpp
    ToyCombine.cpp
    ShapeInferencePass.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/toy/cpp

    DEPENDS
    ToyCh4OpsIncGen
    ToyCh4CombineIncGen
    ToyCh4ShapeInferenceInterfaceIncGen

    LINK_LIBS PUBLIC
    MLIRCastInterfaces
    MLIRIR
    MLIRPass
)
