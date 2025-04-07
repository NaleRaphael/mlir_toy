#ifndef TOY_HELPER_H
#define TOY_HELPER_H

#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Builders.h"

// Define `wrap()` and `unwarp()` for type (see also "mlir/CAPI/IR.h")
DEFINE_C_API_PTR_METHODS(MlirOpBuilder, mlir::OpBuilder)

DEFINE_C_API_METHODS(MlirToyFuncOp, mlir::toy::FuncOp)
DEFINE_C_API_METHODS(MlirToyReturnOp, mlir::toy::ReturnOp)
DEFINE_C_API_METHODS(MlirToyStructAccessOp, mlir::toy::StructAccessOp)
DEFINE_C_API_METHODS(MlirToyStructConstantOp, mlir::toy::StructConstantOp)

#endif // TOY_HELPER_H
