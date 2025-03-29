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
add_public_tablegen_target(ToyCh5OpsIncGen)
add_dependencies(mlir-headers ToyCh5OpsIncGen)

set(LLVM_TARGET_DEFINITIONS ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh5CombineIncGen)
add_dependencies(mlir-headers ToyCh5CombineIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyCh5ShapeInferenceInterfaceIncGen)
add_dependencies(mlir-headers ToyCh5ShapeInferenceInterfaceIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_dialect_library(MLIRToy
    Dialect.cpp
    ToyCombine.cpp
    ShapeInferencePass.cpp
    LowerToAffineLoops.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/toy/cpp

    DEPENDS
    ToyCh5OpsIncGen
    ToyCh5CombineIncGen
    ToyCh5ShapeInferenceInterfaceIncGen

    LINK_LIBS PUBLIC
    MLIRCastInterfaces
    MLIRIR
    MLIRPass
    ${dialect_libs}
    ${extension_libs}
)
