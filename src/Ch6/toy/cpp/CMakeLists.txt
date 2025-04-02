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
add_public_tablegen_target(ToyCh6OpsIncGen)
add_dependencies(mlir-headers ToyCh6OpsIncGen)

set(LLVM_TARGET_DEFINITIONS ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh6CombineIncGen)
add_dependencies(mlir-headers ToyCh6CombineIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyCh6ShapeInferenceInterfaceIncGen)
add_dependencies(mlir-headers ToyCh6ShapeInferenceInterfaceIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
    Core
    Support
    nativecodegen
    OrcJIT
    )

add_mlir_dialect_library(MLIRToy
    Dialect.cpp
    ToyCombine.cpp
    ShapeInferencePass.cpp
    LowerToAffineLoops.cpp
    LowerToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/toy/cpp

    DEPENDS
    ToyCh6OpsIncGen
    ToyCh6CombineIncGen
    ToyCh6ShapeInferenceInterfaceIncGen

    LINK_LIBS PUBLIC
    MLIRCastInterfaces
    MLIRIR
    MLIRPass
    ${dialect_libs}
    ${extension_libs}
    ${conversion_libs}
)
