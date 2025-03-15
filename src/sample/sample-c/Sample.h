#ifndef C_DIALECT_SAMPLE_H
#define C_DIALECT_SAMPLE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// C bindings
// ref: mlir/include/mlir-c/Dialect/Func.h

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Sample, sample);

// TODO: check whether we indeed need the marco `MLIR_CAPI_EXPORTED`
// (when it's omitted, it's still compiled)
// MLIR_CAPI_EXPORTED void mlirRegisterSampleDialect(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // C_DIALECT_SAMPLE_H
