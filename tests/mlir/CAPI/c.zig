pub const stdio = @cImport(@cInclude("stdio.h"));
pub const unistd = @cImport(@cInclude("unistd.h"));

// Header files will be deduplicated by compiler, so we just list them all for
// each file here.
pub const mlir = @cImport({
    // ir.zig
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/Support.h");
    @cInclude("mlir-c/ExecutionEngine.h");
    @cInclude("mlir-c/RegisterEverything.h");
    @cInclude("mlir-c/BuiltinAttributes.h");
    @cInclude("mlir-c/BuiltinTypes.h");
    @cInclude("mlir-c/Conversion.h");
    @cInclude("mlir-c/IntegerSet.h");
    @cInclude("mlir-c/Dialect/Func.h");
    @cInclude("mlir-c/Diagnostics.h");

    // quant.zig
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/BuiltinTypes.h");
    @cInclude("mlir-c/Dialect/Quant.h");

    // pass.zig
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/Pass.h");
    @cInclude("mlir-c/RegisterEverything.h");
    @cInclude("mlir-c/Transforms.h");
    @cInclude("mlir-c/Dialect/Func.h");

    // execution_engine.zig
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/Support.h");
    @cInclude("mlir-c/ExecutionEngine.h");
    @cInclude("mlir-c/RegisterEverything.h");
    @cInclude("mlir-c/BuiltinTypes.h");
    @cInclude("mlir-c/Conversion.h");

    // transform.zig
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/Support.h");
    @cInclude("mlir-c/Dialect/Transform.h");

    // sparse_tensor.zig
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/RegisterEverything.h");
    @cInclude("mlir-c/Dialect/SparseTensor.h");

    // pdl.zig
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/BuiltinTypes.h");
    @cInclude("mlir-c/Dialect/PDL.h");

    // _llvm.zig
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/BuiltinTypes.h");
    @cInclude("mlir-c/Dialect/LLVM.h");
});
