#ifndef TOY_DIALECT_C_H
#define TOY_DIALECT_C_H

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "llvm-c/Types.h"

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
DEFINE_C_API_STRUCT(MlirToyStructAccessOp, const void);
DEFINE_C_API_STRUCT(MlirToyStructConstantOp, const void);
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

// Set the visibility of this op to private.
MLIR_CAPI_EXPORTED void mlirToyFuncOpSetPrivate(MlirToyFuncOp toy_func_op);

//===----------------------------------------------------------------------===//
// ToyReturnOp API
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED MlirToyReturnOp mlirToyReturnOpFromMlirOperation(MlirOperation op);
MLIR_CAPI_EXPORTED bool mlirToyReturnOpIsNull(MlirToyReturnOp toy_return_op);
MLIR_CAPI_EXPORTED bool mlirToyReturnOpHasOperand(MlirToyReturnOp toy_return_op);

//===----------------------------------------------------------------------===//
// ToyStruct API
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED MlirType mlirToyStructTypeGet(intptr_t num_elt, const MlirType* el_types);

//===----------------------------------------------------------------------===//
// ToyStructAccessOp API
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED MlirValue mlirToyStructAccessOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirValue input,
    intptr_t index
);

//===----------------------------------------------------------------------===//
// ToyStructConstantOp API
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED MlirValue mlirToyStructConstantOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirType data_type,
    MlirAttribute data_attr
);

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
// Toy passes API
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED MlirPass mlirToyCreateShapeInferencePass();
MLIR_CAPI_EXPORTED MlirPass mlirToyCreateLowerToAffinePass();
MLIR_CAPI_EXPORTED MlirPass mlirToyCreateLowerToLLVMPass();

//===----------------------------------------------------------------------===//
// Some other helper functions
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED void mlirToyFuncOpErase(MlirToyFuncOp func_op);
MLIR_CAPI_EXPORTED void mlirToyOperationDump(MlirToyFuncOp op);

//===----------------------------------------------------------------------===//
// Extensions
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED void mlirExtModuleDump(MlirModule module);
MLIR_CAPI_EXPORTED void mlirExtOperationEmitError(MlirOperation op, const char *message);
MLIR_CAPI_EXPORTED bool mlirExtBlockIsEmpty(MlirBlock block);
MLIR_CAPI_EXPORTED MlirOperation mlirExtBlockGetLastOperation(MlirBlock block);
MLIR_CAPI_EXPORTED MlirLogicalResult mlirExtVerify(MlirOperation op);

MLIR_CAPI_EXPORTED MlirOperation mlirExtParseSourceFileAsModuleOp(
    MlirContext ctx, MlirStringRef file_path
);

//===----------------------------------------------------------------------===//
// Extensions for affine passes
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED MlirPass mlirExtAffineCreateLoopFusionPass();
MLIR_CAPI_EXPORTED MlirPass mlirExtAffineCreateAffineScalarReplacementPass();

//===----------------------------------------------------------------------===//
// Extensions for LLVM Dialect things
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED MlirPass mlirExtLLVMCreateDIScopeForLLVMFuncOpPass();
MLIR_CAPI_EXPORTED MlirPass mlirExtLLVMCreateRequestCWrappersPass();

//===----------------------------------------------------------------------===//
// Extensions for LLVM IR translation
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED void mlirExtRegisterBuiltinDialectTranslation(MlirContext ctx);
MLIR_CAPI_EXPORTED void mlirExtRegisterLLVMDialectTranslation(MlirContext ctx);
MLIR_CAPI_EXPORTED LLVMModuleRef mlirExtTranslateModuleToLLVMIR(
    MlirOperation module_op, LLVMContextRef llvm_ctx
);

//===----------------------------------------------------------------------===//
// Extensions for LLVM types
//===----------------------------------------------------------------------===//
// XXX: Here we keep using `MLIR_CAPI_EXPORTED` for LLVM extensions to make it
// easier to manage custom APIs. Otherwise, we have to follow the LLVM C-API
// convention to export these symbols.
MLIR_CAPI_EXPORTED bool llvmExtModuleIsNull(LLVMModuleRef llvm_module);

//===----------------------------------------------------------------------===//
// Extensions for CLI options setting
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED void mlirExtContextPrintOpOnDiagnostic(MlirContext ctx, bool enable);
MLIR_CAPI_EXPORTED void mlirExtContextSetPrintStackTraceOnDiagnostic(MlirContext ctx, bool enable);
MLIR_CAPI_EXPORTED void mlirExtOpPrintingFlagsPrintValueUsers(MlirOpPrintingFlags flags);

MLIR_CAPI_EXPORTED void mlirExtPassManagerEnableIRPrinting(
    MlirPassManager pm, bool print_before, bool print_after,
    bool print_module_scope, bool print_after_only_on_change,
    bool print_after_only_on_failure, MlirOpPrintingFlags flags
);

//===----------------------------------------------------------------------===//
// Helper functions (for things cannot be done with pure C-APIs)
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED int mlirExtOptimizeLLVMModule(LLVMModuleRef llvm_module, bool enable_opt);

#ifdef __cplusplus
}
#endif


#endif // TOY_DIALECT_C_H
