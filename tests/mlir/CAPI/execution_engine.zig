const std = @import("std");
const test_options = @import("test_options");
const c = @import("c.zig");

// NOTE: these are the headers required for tests in this file, but it
// aggregate to "c.zig" to avoid multiple invocations of `@cImport`.
// const mlir = @cImport({
//     @cInclude("mlir-c/IR.h");
//     @cInclude("mlir-c/Support.h");
//     @cInclude("mlir-c/ExecutionEngine.h");
//     @cInclude("mlir-c/RegisterEverything.h");
//     @cInclude("mlir-c/BuiltinTypes.h");
//     @cInclude("mlir-c/Conversion.h");
// });
const mlir = c.mlir;

const test_allocator = std.testing.allocator;
const expect = std.testing.expect;
const strref = mlir.mlirStringRefCreateFromCString;

fn registerAllUpstreamDialects(ctx: mlir.MlirContext) void {
    const registry = mlir.mlirDialectRegistryCreate();
    mlir.mlirRegisterAllDialects(registry);
    mlir.mlirContextAppendDialectRegistry(ctx, registry);
    mlir.mlirDialectRegistryDestroy(registry);
}

fn lowerModuleToLLVM(ctx: mlir.MlirContext, module: mlir.MlirModule) void {
    const pm = mlir.mlirPassManagerCreate(ctx);
    defer mlir.mlirPassManagerDestroy(pm);

    const opm = mlir.mlirPassManagerGetNestedUnder(pm, strref("func.func"));

    mlir.mlirPassManagerAddOwnedPass(pm, mlir.mlirCreateConversionConvertFuncToLLVMPass());
    mlir.mlirOpPassManagerAddOwnedPass(opm, mlir.mlirCreateConversionArithToLLVMConversionPass());

    const status = mlir.mlirPassManagerRunOnOp(pm, mlir.mlirModuleGetOperation(module));
    std.debug.assert(!mlir.mlirLogicalResultIsFailure(status));
}

fn checkObjectFile(jit: mlir.MlirExecutionEngine) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const curr_cwd = std.fs.cwd();

    const obj_name = "omp_creation.o";
    var dir_to_dump = try tmp.dir.makeOpenPath("mlir_capi_test", .{});

    // Create an empty file first
    const out_obj = try dir_to_dump.createFile(obj_name, .{});
    out_obj.close();

    const file_size_before = blk: {
        const f = try dir_to_dump.openFile(obj_name, .{});
        defer f.close();
        break :blk (try f.stat()).size;
    };

    // XXX: `mlirExecutionEngineDumpToObjectFile` would complain about
    // the long file path. To avoid this problem, we change the working
    // directory while running it.
    {
        try dir_to_dump.setAsCwd();
        mlir.mlirExecutionEngineDumpToObjectFile(jit, strref(obj_name));

        const file_size_after = blk: {
            const f = try dir_to_dump.openFile(obj_name, .{});
            defer f.close();
            break :blk (try f.stat()).size;
        };

        // Make sure the object file is dumped successfully
        try expect(file_size_after > file_size_before);

        // Get symbol table
        var proc = std.process.Child.init(
            &.{ "nm", "-j", obj_name },
            test_allocator,
        );
        proc.stdout_behavior = .Pipe;
        proc.stderr_behavior = .Pipe;

        var child_stdout = std.ArrayList(u8).init(test_allocator);
        var child_stderr = std.ArrayList(u8).init(test_allocator);
        defer child_stdout.deinit();
        defer child_stderr.deinit();

        try proc.spawn();
        try proc.collectOutput(&child_stdout, &child_stderr, 4096);

        const term = try proc.wait();
        try expect(term.Exited == 0);

        // Check it contains symbols prefixed with __kmpc_
        try expect(std.mem.containsAtLeast(u8, child_stdout.items, 1, "__kmpc_"));

        // Reset cwd
        try curr_cwd.setAsCwd();
    }
}

test "testSimpleExecution" {
    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    registerAllUpstreamDialects(ctx);

    // NOTE: Here we defined a function takes 2 input arguments to perform
    // "add", which is different to the one in original test case. But it's
    // just intended to make sure this would work as well.
    const module_str =
        \\module {
        \\  func.func @add(%arg0 : i32, %arg1 : i32) -> i32 attributes { llvm.emit_c_interface } {
        \\    %res = arith.addi %arg0, %arg1 : i32
        \\    return %res : i32
        \\  }
        \\}
    ;
    const module = mlir.mlirModuleCreateParse(ctx, strref(module_str));
    defer mlir.mlirModuleDestroy(module);

    lowerModuleToLLVM(ctx, module);
    mlir.mlirRegisterAllLLVMTranslations(ctx);

    const jit = mlir.mlirExecutionEngineCreate(module, 2, 0, null, false);
    defer mlir.mlirExecutionEngineDestroy(jit);

    std.debug.assert(!mlir.mlirExecutionEngineIsNull(jit));

    var in1: i32 = 42;
    var in2: i32 = 11;
    var output: i32 = -1;
    var args = [3]?*anyopaque{ @ptrCast(&in1), @ptrCast(&in2), @ptrCast(&output) };

    const result = mlir.mlirExecutionEngineInvokePacked(jit, strref("add"), &args);
    std.debug.assert(!mlir.mlirLogicalResultIsFailure(result));

    std.debug.assert(output == in1 + in2);
}

test "testOmpCreation" {
    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    registerAllUpstreamDialects(ctx);

    const module_str =
        \\module {
        \\  func.func @main() attributes { llvm.emit_c_interface } {
        \\    %0 = arith.constant 0 : i32
        \\    %1 = arith.constant 1 : i32
        \\    %2 = arith.constant 2 : i32
        \\    omp.parallel {
        \\      omp.wsloop for (%3) : i32 = (%0) to (%2) step (%1) {
        \\        omp.yield
        \\      }
        \\      omp.terminator
        \\    }
        \\    llvm.return
        \\  }
        \\}
    ;
    const module = mlir.mlirModuleCreateParse(ctx, strref(module_str));
    defer mlir.mlirModuleDestroy(module);

    lowerModuleToLLVM(ctx, module);

    // As it's stated in the original test case, enabling object dump would
    // require linking to OpenMP library.
    // But here we want to check whether it can work normally if we really
    // want to dump the object file, so we would enable it when linking to
    // OpenMP is requested.
    const dump_obj = test_options.link_openmp;
    const jit = mlir.mlirExecutionEngineCreate(module, 2, 0, null, dump_obj);
    defer mlir.mlirExecutionEngineDestroy(jit);

    std.debug.assert(!mlir.mlirExecutionEngineIsNull(jit));

    if (dump_obj) {
        std.debug.print("checking object file\n", .{});
        try checkObjectFile(jit);
    }
}
