#ifndef TOY_DIALECT_C_H
#define TOY_DIALECT_C_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Toy, toy);

// Same as the marco `DEFINE_C_API_STRUCT` defined in "mlir-c/IR.h"
#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirOpBuilder, void);
DEFINE_C_API_STRUCT(MlirToyFuncOp, const void);
DEFINE_C_API_STRUCT(MlirToyReturnOp, const void);
#undef DEFINE_C_API_STRUCT

//===----------------------------------------------------------------------===//
// mlir::OpBuilder API
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED MlirOpBuilder mlirOpBuilderCreate(MlirContext context);
MLIR_CAPI_EXPORTED MlirLocation mlirOpBuilderGetUnknownLoc(MlirOpBuilder builder);
MLIR_CAPI_EXPORTED void mlirOpBuilderSetInsertionPointToStart(
    MlirOpBuilder builder, MlirBlock block
);
MLIR_CAPI_EXPORTED void mlirOpBuilderSetInsertionPointToEnd(
    MlirOpBuilder builder, MlirBlock block
);

//===----------------------------------------------------------------------===//
// ToyFuncOp API
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED MlirOperation mlirToyFuncOpToMlirOperation(MlirToyFuncOp toy_func_op);

// Create a `toy::FuncOp` as **generic function**. Here the generic function
// is defined as:
// - Inputs: uniformly unranked tensors
// - Output: null (will be inferred later)
//
// See also: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp#L109-L115
MLIR_CAPI_EXPORTED MlirToyFuncOp mlirToyFuncOpCreateAsGeneric(
    MlirOpBuilder op_builder, MlirLocation loc, MlirStringRef name,
    size_t numArgs
);

MLIR_CAPI_EXPORTED MlirToyFuncOp mlirToyFuncOpCreateFromFunctionType(
    MlirOpBuilder op_builder, MlirLocation loc, MlirStringRef name,
    MlirType func_t
);

MLIR_CAPI_EXPORTED MlirType mlirToyFuncOpGetFunctionType(MlirToyFuncOp toy_func_op);

// Change the type of `toy:FuncOp` inplace.
MLIR_CAPI_EXPORTED void mlirToyFuncOpSetType(MlirToyFuncOp toy_func_op, MlirType func_t);

//===----------------------------------------------------------------------===//
// ToyReturnOp API
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED MlirToyReturnOp mlirToyReturnOpFromMlirOperation(MlirOperation op);
MLIR_CAPI_EXPORTED bool mlirToyReturnOpIsNull(MlirToyReturnOp toy_return_op);
MLIR_CAPI_EXPORTED bool mlirToyReturnOpHasOperand(MlirToyReturnOp toy_return_op);

//===----------------------------------------------------------------------===//
// Other ToyOp API
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED MlirValue mlirToyConstantOpCreateFromDouble(
    MlirOpBuilder op_builder, MlirLocation loc, double value
);
MLIR_CAPI_EXPORTED MlirValue mlirToyConstantOpCreateFromTensor(
    MlirOpBuilder op_builder, MlirLocation loc, MlirType shape_t,
    MlirAttribute data_attr
);

MLIR_CAPI_EXPORTED MlirValue mlirToyAddOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirValue lhs, MlirValue rhs
);

MLIR_CAPI_EXPORTED MlirValue mlirToyMulOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirValue lhs, MlirValue rhs
);

MLIR_CAPI_EXPORTED MlirValue mlirToyReshapeOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirType shape_t,
    MlirValue value
);

MLIR_CAPI_EXPORTED MlirValue mlirToyTransposeOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirValue operand
);

MLIR_CAPI_EXPORTED MlirValue mlirToyGenericCallOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirStringRef callee,
    intptr_t n, MlirValue const *operands
);

MLIR_CAPI_EXPORTED void mlirToyReturnOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, intptr_t n,
    MlirValue const *operands
);

MLIR_CAPI_EXPORTED void mlirToyPrintOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirValue value
);

//===----------------------------------------------------------------------===//
// Some other helper functions
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED void mlirToyFuncOpErase(MlirToyFuncOp func_op);
MLIR_CAPI_EXPORTED void mlirToyOperationDump(MlirToyFuncOp op);
MLIR_CAPI_EXPORTED void mlirModuleDump(MlirModule module);

//===----------------------------------------------------------------------===//
// Extensions
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED void mlirExtContextPrintOpOnDiagnostic(MlirContext ctx, bool enable);
MLIR_CAPI_EXPORTED void mlirExtContextSetPrintStackTraceOnDiagnostic(MlirContext ctx, bool enable);

MLIR_CAPI_EXPORTED void mlirExtOpPrintingFlagsPrintValueUsers(MlirOpPrintingFlags flags);

MLIR_CAPI_EXPORTED void mlirExtOperationEmitError(MlirOperation op, const char *message);
MLIR_CAPI_EXPORTED void mlirExtModuleDump(MlirModule module);
MLIR_CAPI_EXPORTED bool mlirExtBlockIsEmpty(MlirBlock block);
MLIR_CAPI_EXPORTED MlirOperation mlirExtBlockGetLastOperation(MlirBlock block);
MLIR_CAPI_EXPORTED MlirLogicalResult mlirVerify(MlirOperation op);

MLIR_CAPI_EXPORTED MlirOperation mlirExtParseSourceFileAsModuleOp(
    MlirContext ctx, MlirStringRef file_path
);

#ifdef __cplusplus
}
#endif


#endif // TOY_DIALECT_C_H
