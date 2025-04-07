#include "toy/c/toy.h"
#include "toy/cpp/Dialect.h"
#include "toy/cpp/Passes.h"
#include "toy/cpp/Helper.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

// wrap() and unwrap() is defined in "mlir/CAPI/Wrap.h", and it's included by
// "mlir/CAPI/Support.h" -> "mlir/CAPI/Registration.h".

// From "mlir/CAPI/Registration.h"
// name, namespace, class_name
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Toy, toy, mlir::toy::ToyDialect)

// NOTE:
// MlirXXX: Types defined to be used in C-API
// mlirXXX: C-APIs
// MLIRXXX: MLIR C++ impls
//
// - In headers of C-API ("mlir-c/xxx.h"), we should not include C++ headers
// - In C++ impls for C-API ("mlir/lib/CAPI/xxx.cpp"), we can include headers of
//   C-API and C++.

//===----------------------------------------------------------------------===//
// mlir::OpBuilder API
//===----------------------------------------------------------------------===//
MlirOpBuilder mlirOpBuilderCreate(MlirContext context) {
    auto *builder = new mlir::OpBuilder(unwrap(context));
    return wrap(builder);
}

MlirLocation mlirOpBuilderGetUnknownLoc(MlirOpBuilder builder) {
    return wrap(unwrap(builder)->getUnknownLoc());
}

void mlirOpBuilderSetInsertionPointToStart(MlirOpBuilder builder, MlirBlock block) {
    unwrap(builder)->setInsertionPointToStart(unwrap(block));
}

void mlirOpBuilderSetInsertionPointToEnd(MlirOpBuilder builder, MlirBlock block) {
    unwrap(builder)->setInsertionPointToEnd(unwrap(block));
}

//===----------------------------------------------------------------------===//
// ToyFuncOp API
//===----------------------------------------------------------------------===//
MlirOperation mlirToyFuncOpToMlirOperation(MlirToyFuncOp toy_func_op) {
    return wrap(unwrap(toy_func_op).getOperation());
}

MlirToyFuncOp mlirToyFuncOpCreateAsGeneric(
    MlirOpBuilder op_builder, MlirLocation loc, MlirStringRef name,
    size_t num_args
) {
    auto *builder = unwrap(op_builder);

    mlir::Location _loc = unwrap(loc);
    llvm::StringRef _name = unwrap(name);

    // For prototype, arguments types are uniformly unranked tensors.
    // https://github.com/llvm/llvm-project/blob/6009708b/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp#L109-L110
    llvm::SmallVector<mlir::Type, 4> _arg_types(
        num_args,
        mlir::UnrankedTensorType::get(builder->getF64Type())
    );
    auto func_type = builder->getFunctionType(_arg_types, std::nullopt);

    auto op = builder->create<mlir::toy::FuncOp>(_loc, _name, func_type);
    return wrap(op);
}

MlirToyFuncOp mlirToyFuncOpCreateFromFunctionType(
    MlirOpBuilder op_builder, MlirLocation loc, MlirStringRef name,
    MlirType func_t
) {
    auto *builder = unwrap(op_builder);

    mlir::Location _loc = unwrap(loc);
    llvm::StringRef _name = unwrap(name);
    mlir::FunctionType _func_t = unwrap(func_t).dyn_cast<mlir::FunctionType>();

    auto op = builder->create<mlir::toy::FuncOp>(_loc, _name, _func_t);
    return wrap(op);
}

MlirType mlirToyFuncOpGetFunctionType(MlirToyFuncOp toy_func_op) {
    mlir::toy::FuncOp _toy_func_op = unwrap(toy_func_op);
    mlir::FunctionType _func_t = _toy_func_op.getFunctionType();
    return wrap(_func_t);
}

void mlirToyFuncOpSetType(MlirToyFuncOp toy_func_op, MlirType func_t) {
    mlir::toy::FuncOp _toy_func_op = unwrap(toy_func_op);
    mlir::FunctionType _func_t = unwrap(func_t).dyn_cast<mlir::FunctionType>();
    _toy_func_op.setType(_func_t);
}

void mlirToyFuncOpSetPrivate(MlirToyFuncOp toy_func_op) {
    unwrap(toy_func_op).setPrivate();
}

//===----------------------------------------------------------------------===//
// ToyReturnOp API
//===----------------------------------------------------------------------===//
MlirToyReturnOp mlirToyReturnOpFromMlirOperation(MlirOperation op) {
    mlir::toy::ReturnOp _op = llvm::dyn_cast<mlir::toy::ReturnOp>(unwrap(op));
    return wrap(_op);
}

bool mlirToyReturnOpIsNull(MlirToyReturnOp toy_return_op) {
    return !toy_return_op.ptr;
}

bool mlirToyReturnOpHasOperand(MlirToyReturnOp toy_return_op) {
    return unwrap(toy_return_op).hasOperand();
}

//===----------------------------------------------------------------------===//
// ToyStructType API
//===----------------------------------------------------------------------===//
MlirType mlirToyStructTypeGet(intptr_t num_elt, const MlirType* el_types) {
    std::vector<mlir::Type> unwrapped_el_types;
    for (int i = 0; i < num_elt; i++) {
        unwrapped_el_types.push_back(unwrap(el_types[i]));
    }
    mlir::toy::StructType struct_type = mlir::toy::StructType::get(
        llvm::ArrayRef(unwrapped_el_types)
    );
    return wrap(struct_type);
}

//===----------------------------------------------------------------------===//
// ToyStructAccessOp API
//===----------------------------------------------------------------------===//
MlirValue mlirToyStructAccessOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirValue input,
    intptr_t index
) {
    auto *builder = unwrap(op_builder);

    mlir::Location _loc = unwrap(loc);
    mlir::Value _input = unwrap(input);
    auto _index = static_cast<size_t>(index);

    // TODO: check whehter we should cast index from `intptr_t` to `size_t`
    auto op = builder->create<mlir::toy::StructAccessOp>(_loc, _input, _index);
    return wrap(op.getOutput());
}

//===----------------------------------------------------------------------===//
// ToyStructConstantOp API
//===----------------------------------------------------------------------===//
MlirValue mlirToyStructConstantOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirType data_type,
    MlirAttribute data_attr
) {
    auto *builder = unwrap(op_builder);

    mlir::Location _loc = unwrap(loc);
    mlir::Type _type = unwrap(data_type);
    mlir::ArrayAttr _attr = llvm::cast<mlir::ArrayAttr>(unwrap(data_attr));

    auto op = builder->create<mlir::toy::StructConstantOp>(_loc, _type, _attr);
    return wrap(op.getOutput());
}

//===----------------------------------------------------------------------===//
// Other ToyOp API
//===----------------------------------------------------------------------===//
// NOTE: The returned types of these functions is determined by the definitions
// in "Ops.td". If the field `results` exists in definition, we return a
// `MlirValue`. Otherwise, we return nothing. Also, note that the output of
// builder `op` would have a method `getResult()` only when the field `results`
// is defined.

// TODO: remove this. Try to create constant from Zig side directly like:
// https://github.com/llvm/llvm-project/blob/main/mlir/test/CAPI/ir.c#L1862-L1873
MlirValue mlirToyConstantOpCreateFromDouble(
    MlirOpBuilder op_builder, MlirLocation loc, double value
) {
    auto *builder = unwrap(op_builder);
    mlir::Location _loc = unwrap(loc);

    auto op = builder->create<mlir::toy::ConstantOp>(_loc, value);
    return wrap(op.getResult());
}

MlirValue mlirToyConstantOpCreateFromTensor(
    MlirOpBuilder op_builder, MlirLocation loc, MlirType shape_t,
    MlirAttribute data_attr
) {
    auto *builder = unwrap(op_builder);
    mlir::Location _loc = unwrap(loc);
    mlir::Type _shape_t = unwrap(shape_t);
    mlir::DenseElementsAttr _data_attr = unwrap(data_attr).dyn_cast<mlir::DenseElementsAttr>();

    auto op = builder->create<mlir::toy::ConstantOp>(_loc, _shape_t, _data_attr);
    return wrap(op.getResult());
}

MlirValue mlirToyAddOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirValue lhs, MlirValue rhs
) {
    auto *builder = unwrap(op_builder);
    mlir::Location _loc = unwrap(loc);
    mlir::Value _lhs = unwrap(lhs);
    mlir::Value _rhs = unwrap(rhs);

    auto op = builder->create<mlir::toy::AddOp>(_loc, _lhs, _rhs);
    return wrap(op.getResult());
}

MlirValue mlirToyMulOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirValue lhs, MlirValue rhs
) {
    auto *builder = unwrap(op_builder);
    mlir::Location _loc = unwrap(loc);
    mlir::Value _lhs = unwrap(lhs);
    mlir::Value _rhs = unwrap(rhs);

    auto op = builder->create<mlir::toy::MulOp>(_loc, _lhs, _rhs);
    return wrap(op.getResult());
}

MlirValue mlirToyReshapeOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirType shape_t,
    MlirValue value
) {
    auto *builder = unwrap(op_builder);
    mlir::Location _loc = unwrap(loc);
    mlir::Type _shape_t = unwrap(shape_t);
    mlir::Value _value = unwrap(value);

    auto op = builder->create<mlir::toy::ReshapeOp>(_loc, _shape_t, _value);
    return wrap(op.getResult());
}

MlirValue mlirToyTransposeOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirValue operand
) {
    auto *builder = unwrap(op_builder);
    mlir::Location _loc = unwrap(loc);
    mlir::Value _operand = unwrap(operand);

    auto op = builder->create<mlir::toy::TransposeOp>(_loc, _operand);
    return wrap(op.getResult());
}

// NOTE: `mlirOperationStateAddOperands()` can be an example if we need to
// manipulate the underlying data directly.
// https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/CAPI/IR/IR.cpp#L321-L324
MlirValue mlirToyGenericCallOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirStringRef callee,
    intptr_t n, MlirValue const *operands
) {
    auto *builder = unwrap(op_builder);
    mlir::Location _loc = unwrap(loc);
    llvm::StringRef _callee = unwrap(callee);

    llvm::SmallVector<mlir::Value, 4> _operands;
    for (auto i = 0; i < n; i++) {
        _operands.push_back(unwrap(operands[i]));
    }

    // XXX: we don't have ResultTypes, so we cannot use this `build()` function.
    // auto op = builder->create<mlir::toy::GenericCallOp>(_loc, callee, /* result_t, */ n, operands);
    auto op = builder->create<mlir::toy::GenericCallOp>(_loc, _callee, _operands);
    return wrap(op.getResult());
}

void mlirToyReturnOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, intptr_t n,
    MlirValue const *operands
) {
    auto *builder = unwrap(op_builder);
    mlir::Location _loc = unwrap(loc);
    llvm::SmallVector<mlir::Value, 4> _operands;
    for (auto i = 0; i < n; i++) {
        _operands.push_back(unwrap(operands[i]));
    }
    mlir::ValueRange values(_operands);

    builder->create<mlir::toy::ReturnOp>(_loc, values);
}

void mlirToyPrintOpCreate(
    MlirOpBuilder op_builder, MlirLocation loc, MlirValue value
) {
    auto *builder = unwrap(op_builder);
    mlir::Location _loc = unwrap(loc);
    mlir::Value _value = unwrap(value);

    builder->create<mlir::toy::PrintOp>(_loc, _value);
}

//===----------------------------------------------------------------------===//
// Toy passes API
//===----------------------------------------------------------------------===//
MlirPass mlirToyCreateShapeInferencePass() {
    return wrap(mlir::toy::createShapeInferencePass().release());
}

MlirPass mlirToyCreateLowerToAffinePass() {
    return wrap(mlir::toy::createLowerToAffinePass().release());
}

MlirPass mlirToyCreateLowerToLLVMPass() {
    return wrap(mlir::toy::createLowerToLLVMPass().release());
}

//===----------------------------------------------------------------------===//
// Some other helper functions
//===----------------------------------------------------------------------===//
void mlirToyFuncOpErase(MlirToyFuncOp func_op) {
    unwrap(func_op)->erase();
}

void mlirToyOperationDump(MlirToyFuncOp op) {
    unwrap(op)->dump();
}

//===----------------------------------------------------------------------===//
// Extensions
//===----------------------------------------------------------------------===//
void mlirExtModuleDump(MlirModule module) {
    unwrap(module)->dump();
}

void mlirExtOperationEmitError(MlirOperation op, const char *message) {
    unwrap(op)->emitError() << message;
}

bool mlirExtBlockIsEmpty(MlirBlock block) {
    return unwrap(block)->getOperations().empty();
}

MlirOperation mlirExtBlockGetLastOperation(MlirBlock block) {
    mlir::Operation &back = unwrap(block)->back();
    return wrap(&back);
}

MlirLogicalResult mlirExtVerify(MlirOperation op) {
    return wrap(mlir::verify(unwrap(op)));
}

MlirOperation mlirExtParseSourceFileAsModuleOp(
    MlirContext ctx, MlirStringRef file_path
) {
    mlir::MLIRContext *_ctx = unwrap(ctx);
    llvm::StringRef _file_path = unwrap(file_path);
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(_file_path, _ctx).release();
    return wrap(module.getOperation());
}

//===----------------------------------------------------------------------===//
// Extensions for affine passes
//===----------------------------------------------------------------------===//
MlirPass mlirExtAffineCreateLoopFusionPass() {
    return wrap(mlir::affine::createLoopFusionPass().release());
}

MlirPass mlirExtAffineCreateAffineScalarReplacementPass() {
    return wrap(mlir::affine::createAffineScalarReplacementPass().release());
}

//===----------------------------------------------------------------------===//
// Extensions for LLVM Dialect things
//===----------------------------------------------------------------------===//
MlirPass mlirExtLLVMCreateDIScopeForLLVMFuncOpPass() {
    return wrap(mlir::LLVM::createDIScopeForLLVMFuncOpPass().release());
}

// Attach a `llvm.emit_c_interface` attribute to each function. This makes it
// possible to invoke a function via `ExecutionEngine` from C-API side.
// (In `mlirExecutionEngineInvokePacked()`, it will prefixed the input function
// name with "_mlir_ciface_" to look up functions respecting calling convention
// of C.)
MlirPass mlirExtLLVMCreateRequestCWrappersPass() {
    return wrap(mlir::LLVM::createRequestCWrappersPass().release());
}

//===----------------------------------------------------------------------===//
// Extensions for LLVM IR translation
//===----------------------------------------------------------------------===//
void mlirExtRegisterBuiltinDialectTranslation(MlirContext ctx) {
    mlir::MLIRContext *_ctx = unwrap(ctx);
    mlir::registerBuiltinDialectTranslation(*_ctx);
}

void mlirExtRegisterLLVMDialectTranslation(MlirContext ctx) {
    mlir::MLIRContext *_ctx = unwrap(ctx);
    mlir::registerLLVMDialectTranslation(*_ctx);
}

LLVMModuleRef mlirExtTranslateModuleToLLVMIR(
    MlirOperation module_op, LLVMContextRef llvm_ctx
) {
    mlir::Operation *_module_op = unwrap(module_op);

    llvm::LLVMContext *_llvm_ctx = llvm::unwrap(llvm_ctx);
    auto llvm_module = mlir::translateModuleToLLVMIR(_module_op, *_llvm_ctx);

    if (!llvm_module) {
        llvm::errs() << "failed to emit LLVM IR\n";
    }

    // NOTE: Convert it to `LLVMModuleRef` anyway, but we must check whether
    // it's a nullptr before processing it.
    return llvm::wrap(llvm_module.release());
}

//===----------------------------------------------------------------------===//
// Extensions for LLVM types
//===----------------------------------------------------------------------===//
bool llvmExtModuleIsNull(LLVMModuleRef llvm_module) {
    return llvm::unwrap(llvm_module) == NULL;
}

//===----------------------------------------------------------------------===//
// Extensions for CLI options setting
//===----------------------------------------------------------------------===//
void mlirExtContextPrintOpOnDiagnostic(MlirContext ctx, bool enable) {
    unwrap(ctx)->printOpOnDiagnostic(enable);
}

void mlirExtContextSetPrintStackTraceOnDiagnostic(MlirContext ctx, bool enable) {
    unwrap(ctx)->printStackTraceOnDiagnostic(enable);
}

void mlirExtOpPrintingFlagsPrintValueUsers(MlirOpPrintingFlags flags) {
    // XXX: In LLVM 17, this setter does not take a `enable` flag. Once it's
    // called, `printValueUsersFlag` will be enabled.
    // https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/IR/AsmPrinter.cpp#L247-L251
    unwrap(flags)->printValueUsers();
}

// The builtin API `mlirPassManagerEnableIRPrinting` doesn't support
// fine-grained control of those features via C-API until LLVM 20. So we made
// this extension for it.
// https://github.com/llvm/llvm-project/blob/release/20.x/mlir/lib/CAPI/IR/Pass.cpp#L47-L72
void mlirExtPassManagerEnableIRPrinting(
    MlirPassManager pm, bool print_before, bool print_after,
    bool print_module_scope, bool print_after_only_on_change,
    bool print_after_only_on_failure, MlirOpPrintingFlags flags
) {
    typedef std::function<bool(mlir::Pass *, mlir::Operation *)> cfg_t;
    cfg_t shouldPrintBeforePass = [print_before](mlir::Pass *, mlir::Operation *) {
        return print_before;
    };
    cfg_t shouldPrintAfterPass = [print_after](mlir::Pass *, mlir::Operation *) {
        return print_after;
    };

    unwrap(pm)->enableIRPrinting(
        shouldPrintBeforePass,
        shouldPrintAfterPass,
        print_module_scope,
        print_after_only_on_change,
        print_after_only_on_failure,
        llvm::errs(),
        *unwrap(flags)
    );
}

//===----------------------------------------------------------------------===//
// Helper functions (for things cannot be done with pure C-APIs)
//===----------------------------------------------------------------------===//
int mlirExtOptimizeLLVMModule(LLVMModuleRef llvm_module, bool enable_opt) {
    llvm::Module *_llvm_module = llvm::unwrap(llvm_module);

    auto opt_pipeline = mlir::makeOptimizingTransformer(
        enable_opt ? 3 : 0, 0, nullptr
    );
    if (auto err = opt_pipeline(_llvm_module)) {
        llvm::errs() << "Failed to optimize LLVM IR" << err << "\n";
        return -1;
    }
    return 0;
}
