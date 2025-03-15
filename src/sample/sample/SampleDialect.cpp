// // #include "mlir/IR/Dialect.h"
//
// // // Definitions of C-API
// // #include "mlir/CAPI/IR.h"
//
// // Call `MLIR_DEFINE_CAPI_DIALECT_REGISTRATION` to register dialect, and
// // generate `mlirGetDialectHandle__##Namespace##__()` function.
// // See also: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/include/mlir/CAPI/Registration.h#L36-L52
// #include "mlir/CAPI/Registration.h"
//
// // For `wrap()`, `unwrap()` macros (actually defined in "mlir/CAPI/Wrap.h")
// // #include "mlir/CAPI/Support.h"
//
// // The actual C-API for user to use
// #include "mlir-c/IR.h"
// #include "mlir-c/Support.h"

#include "SampleDialect.h"

using namespace mlir;
using namespace mlir::sample;

// ---------------------------------------------------------------------------
// Dialect definitions (usually generated by tablegen: XXXDialect.cpp.inc)
// ---------------------------------------------------------------------------
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sample::SampleDialect)
namespace mlir {
namespace sample {

SampleDialect::SampleDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<SampleDialect>()) {
  initialize();
}

SampleDialect::~SampleDialect() = default;

} // namespace sample
} // namespace mlir

// ---------------------------------------------------------------------------
// Dialect definitions (usually written by our own: XXXDialect.cpp)
// ---------------------------------------------------------------------------
void mlir::sample::SampleDialect::initialize() {
    // This is a sample, we do nothing here for now.
}
