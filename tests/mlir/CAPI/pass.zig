const std = @import("std");
const c = @import("c.zig");
const helper = @import("helper.zig");

// NOTE: these are the headers required for tests in this file, but it
// aggregate to "c.zig" to avoid multiple invocations of `@cImport`.
// const mlir = @cImport({
//     @cInclude("mlir-c/IR.h");
//     @cInclude("mlir-c/Pass.h");
//     @cInclude("mlir-c/RegisterEverything.h");
//     @cInclude("mlir-c/Transforms.h");
//     @cInclude("mlir-c/Dialect/Func.h");
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

// Same helper function as it's defined in "tests/ir.zig".
fn printToStderr(str: mlir.MlirStringRef, user_data: ?*anyopaque) callconv(.C) void {
    _ = user_data;
    _ = c.stdio.fwrite(str.data, 1, str.length, c.stdio.stderr);
}

fn dontPrint(str: mlir.MlirStringRef, user_data: ?*anyopaque) callconv(.C) void {
    _ = str;
    _ = user_data;
}

test "testRunPassOnModule" {
    var fc_runner = try helper.FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);
    registerAllUpstreamDialects(ctx);

    const func_asm =
        \\func.func @foo(%arg0 : i32) -> i32 {
        \\  %res = arith.addi %arg0, %arg0 : i32
        \\  return %res : i32
        \\}
    ;
    const func = mlir.mlirOperationCreateParse(
        ctx,
        strref(func_asm),
        strref("func_asm"),
    );
    defer mlir.mlirOperationDestroy(func);
    try expect(!mlir.mlirOperationIsNull(func));

    const pm = mlir.mlirPassManagerCreate(ctx);
    defer mlir.mlirPassManagerDestroy(pm);

    // Run the print-op-stats pass on the top-level module
    const print_op_stat_pass = mlir.mlirCreateTransformsPrintOpStats();
    mlir.mlirPassManagerAddOwnedPass(pm, print_op_stat_pass);

    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-01");
        const success = mlir.mlirPassManagerRunOnOp(pm, func);
        try expect(!mlir.mlirLogicalResultIsFailure(success));

        // CASE-01: arith.addi , 1
        // CASE-01: func.func , 1
        // CASE-01: func.return , 1

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }
}

test "testRunpassOnNestedModule" {
    var fc_runner = try helper.FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);
    registerAllUpstreamDialects(ctx);

    const module_asm =
        \\module {
        \\  func.func @foo(%arg0 : i32) -> i32 {
        \\    %res = arith.addi %arg0, %arg0 : i32
        \\    return %res : i32
        \\  }
        \\  module {
        \\    func.func @bar(%arg0 : f32) -> f32 {
        \\      %res = arith.addf %arg0, %arg0 : f32
        \\      return %res : f32
        \\    }
        \\  }
        \\}
    ;
    const module = mlir.mlirOperationCreateParse(
        ctx,
        strref(module_asm),
        strref("module_asm"),
    );
    defer mlir.mlirOperationDestroy(module);
    try expect(!mlir.mlirOperationIsNull(module));

    // Run the print-op-stats pass on functions under the top-level module
    {
        const pm = mlir.mlirPassManagerCreate(ctx);
        defer mlir.mlirPassManagerDestroy(pm);
        const nested_func_pm = mlir.mlirPassManagerGetNestedUnder(
            pm,
            strref("func.func"),
        );

        const print_op_stat_pass = mlir.mlirCreateTransformsPrintOpStats();
        mlir.mlirOpPassManagerAddOwnedPass(nested_func_pm, print_op_stat_pass);

        if (fc_runner.canRun()) {
            try fc_runner.runAndWaitForInput("CASE-02-a");
            const success = mlir.mlirPassManagerRunOnOp(pm, module);
            try expect(!mlir.mlirLogicalResultIsFailure(success));

            // CASE-02-a: arith.addi , 1
            // CASE-02-a: func.func , 1
            // CASE-02-a: func.return , 1

            const term = try fc_runner.cleanup();
            try expect(term != null and term.?.Exited == 0);
        }
    }

    // Run the print-op-stats pass on functions under the nested module
    {
        const pm = mlir.mlirPassManagerCreate(ctx);
        defer mlir.mlirPassManagerDestroy(pm);
        const nested_module_pm = mlir.mlirPassManagerGetNestedUnder(
            pm,
            strref("builtin.module"),
        );
        const nested_func_pm = mlir.mlirOpPassManagerGetNestedUnder(
            nested_module_pm,
            strref("func.func"),
        );

        const print_op_stat_pass = mlir.mlirCreateTransformsPrintOpStats();
        mlir.mlirOpPassManagerAddOwnedPass(nested_func_pm, print_op_stat_pass);

        if (fc_runner.canRun()) {
            try fc_runner.runAndWaitForInput("CASE-02-b");
            const success = mlir.mlirPassManagerRunOnOp(pm, module);
            try expect(!mlir.mlirLogicalResultIsFailure(success));

            // CASE-02-b: arith.addf , 1
            // CASE-02-b: func.func , 1
            // CASE-02-b: func.return , 1

            const term = try fc_runner.cleanup();
            try expect(term != null and term.?.Exited == 0);
        }
    }
}

test "testPrintPassPipeline" {
    var fc_runner = try helper.FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    const pm = mlir.mlirPassManagerCreateOnOperation(ctx, strref("any"));
    defer mlir.mlirPassManagerDestroy(pm);

    // Populate the pass-manager
    const nested_module_pm = mlir.mlirPassManagerGetNestedUnder(
        pm,
        strref("builtin.module"),
    );
    const nested_func_pm = mlir.mlirOpPassManagerGetNestedUnder(
        nested_module_pm,
        strref("func.func"),
    );
    const print_op_stat_pass = mlir.mlirCreateTransformsPrintOpStats();
    mlir.mlirOpPassManagerAddOwnedPass(nested_func_pm, print_op_stat_pass);

    if (fc_runner.canRun()) {
        // Print the top level pass manager
        {
            try fc_runner.runAndWaitForInput("CASE-03-a");
            const opm = mlir.mlirPassManagerGetAsOpPassManager(pm);
            mlir.mlirPrintPassPipeline(opm, printToStderr, null);

            // CASE-03-a: any(builtin.module(func.func(print-op-stats{json=false})))

            const term = try fc_runner.cleanup();
            try expect(term != null and term.?.Exited == 0);
        }

        // Print the pipeline nested one level down
        {
            try fc_runner.runAndWaitForInput("CASE-03-b");
            mlir.mlirPrintPassPipeline(nested_module_pm, printToStderr, null);

            // CASE-03-b: builtin.module(func.func(print-op-stats{json=false}))

            const term = try fc_runner.cleanup();
            try expect(term != null and term.?.Exited == 0);
        }

        // Print the pipeline nested two level down
        {
            try fc_runner.runAndWaitForInput("CASE-03-c");
            mlir.mlirPrintPassPipeline(nested_func_pm, printToStderr, null);

            // CASE-03-c: func.func(print-op-stats{json=false})

            const term = try fc_runner.cleanup();
            try expect(term != null and term.?.Exited == 0);
        }
    }
}

test "testParsePassPipeline" {
    var fc_runner = try helper.FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    const pm = mlir.mlirPassManagerCreate(ctx);
    defer mlir.mlirPassManagerDestroy(pm);

    const opm = mlir.mlirPassManagerGetAsOpPassManager(pm);

    // Try parse a pipeline
    // NOTE: Here we replace the callback `printToStderr` with `dontPrint` to
    // suppress the error message since it won't affect the test:
    // > MLIR Textual PassPipeline Parser:1:11: error:
    // > 'print-op-stats' does not refer to a registered pass or pass pipeline
    var status = mlir.mlirParsePassPipeline(
        opm,
        strref("builtin.module(func.func(print-op-stats{json=false}))"),
        dontPrint,
        null,
    );
    // Expect a failure, we haven't registered the print-op-stats pass yet.
    try expect(mlir.mlirLogicalResultIsFailure(status));

    // Try again after registrating the pass
    mlir.mlirRegisterTransformsPrintOpStats();
    status = mlir.mlirParsePassPipeline(
        opm,
        strref("builtin.module(func.func(print-op-stats{json=false}))"),
        printToStderr,
        null,
    );
    try expect(mlir.mlirLogicalResultIsSuccess(status));

    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-04-a");
        // const opm = mlir.mlirPassManagerGetasOpPassManager(pm);
        mlir.mlirPrintPassPipeline(opm, printToStderr, null);

        // CASE-04-a: builtin.module(func.func(print-op-stats{json=false}))

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    // Try appending a pass:
    status = mlir.mlirOpPassManagerAddPipeline(
        opm,
        strref("func.func(print-op-stats{json=false})"),
        printToStderr,
        null,
    );
    try expect(mlir.mlirLogicalResultIsSuccess(status));

    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-04-b");
        mlir.mlirPrintPassPipeline(opm, printToStderr, null);

        // CASE-04-b: builtin.module(
        // CHECK-SAME: func.func(print-op-stats{json=false}),
        // CHECK-SAME: func.func(print-op-stats{json=false}))

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }
}

test "testParseErrorCapture" {
    var fc_runner = try helper.FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    const pm = mlir.mlirPassManagerCreate(ctx);
    defer mlir.mlirPassManagerDestroy(pm);

    const opm = mlir.mlirPassManagerGetAsOpPassManager(pm);
    const invalid_pipeline = strref("invalid");
    var status: mlir.MlirLogicalResult = undefined;

    const checkResult = struct {
        const fcr_t = helper.FileCheckRunner;
        const ret_t = mlir.MlirLogicalResult;

        fn func(
            fcr: *fcr_t,
            prefix: []const u8,
            comptime should_print: bool,
            args: anytype,
            head: ?[]const u8,
            tail: ?[]const u8,
        ) !ret_t {
            var ret: ret_t = undefined;
            const can_run = fcr.canRun();
            const print_fn = if (should_print) printToStderr else dontPrint;

            if (can_run) {
                try fcr.runAndWaitForInput(prefix);
            }

            if (head) |v| std.debug.print("{s}", .{v});
            ret = @call(.auto, args.func, .{
                args.pm,
                args.pipeline,
                print_fn,
                args.user_data,
            });
            if (tail) |v| std.debug.print("{s}", .{v});

            if (can_run) {
                const term = try fcr.cleanup();
                try expect(term != null and term.?.Exited == 0);
            }
            return ret;
        }
    }.func;

    // CASE-05-a: expected pass pipeline to be wrapped with the anchor operation type
    status = try checkResult(&fc_runner, "CASE-05-a", true, &.{
        .func = mlir.mlirParsePassPipeline,
        .pm = opm,
        .pipeline = invalid_pipeline,
        .user_data = null,
    }, null, null);
    try expect(mlir.mlirLogicalResultIsFailure(status));

    // CASE-05-b: 'invalid' does not refer to a registered pass or pass pipeline
    status = try checkResult(&fc_runner, "CASE-05-b", true, &.{
        .func = mlir.mlirOpPassManagerAddPipeline,
        .pm = opm,
        .pipeline = invalid_pipeline,
        .user_data = null,
    }, null, null);
    try expect(mlir.mlirLogicalResultIsFailure(status));

    // CASE-05-c: dontPrint: <>
    status = try checkResult(&fc_runner, "CASE-05-c", false, &.{
        .func = mlir.mlirParsePassPipeline,
        .pm = opm,
        .pipeline = invalid_pipeline,
        .user_data = null,
    }, "dontPrint: <", ">\n");
    try expect(mlir.mlirLogicalResultIsFailure(status));

    // CASE-05-d: dontPrint: <>
    status = try checkResult(&fc_runner, "CASE-05-d", false, &.{
        .func = mlir.mlirOpPassManagerAddPipeline,
        .pm = opm,
        .pipeline = invalid_pipeline,
        .user_data = null,
    }, "dontPrint: <", ">\n");
    try expect(mlir.mlirLogicalResultIsFailure(status));
}

const TestExternalPassUserData = extern struct {
    construct_call_count: c_int = 0,
    destruct_call_count: c_int = 0,
    initialize_call_count: c_int = 0,
    clone_call_count: c_int = 0,
    run_call_count: c_int = 0,

    const Self = @This();

    pub fn castFrom(ptr: *anyopaque) *Self {
        return @as(*Self, @alignCast(@ptrCast(ptr)));
    }
};

const InitializePassProto = *const fn (
    ctx: mlir.MlirContext,
    user_data: ?*anyopaque,
) callconv(.C) mlir.MlirLogicalResult;

const RunPassProto = *const fn (
    op: mlir.MlirOperation,
    pass: mlir.MlirExternalPass,
    user_data: ?*anyopaque,
) callconv(.C) void;

fn testConstructExternalPass(user_data: ?*anyopaque) callconv(.C) void {
    var ud = TestExternalPassUserData.castFrom(user_data.?);
    ud.construct_call_count += 1;
}

fn testDestructExternalPass(user_data: ?*anyopaque) callconv(.C) void {
    var ud = TestExternalPassUserData.castFrom(user_data.?);
    ud.destruct_call_count += 1;
}

fn testInitializeExternalPass(
    _: mlir.MlirContext,
    user_data: ?*anyopaque,
) callconv(.C) mlir.MlirLogicalResult {
    var ud = TestExternalPassUserData.castFrom(user_data.?);
    ud.initialize_call_count += 1;
    return mlir.mlirLogicalResultSuccess();
}

fn testInitializeFailingExternalPass(
    _: mlir.MlirContext,
    user_data: ?*anyopaque,
) callconv(.C) mlir.MlirLogicalResult {
    var ud = TestExternalPassUserData.castFrom(user_data.?);
    ud.initialize_call_count += 1;
    return mlir.mlirLogicalResultFailure();
}

fn testCloneExternalPass(user_data: ?*anyopaque) callconv(.C) ?*anyopaque {
    var ud = TestExternalPassUserData.castFrom(user_data.?);
    ud.clone_call_count += 1;
    return user_data;
}

fn testRunExternalPass(
    _: mlir.MlirOperation,
    _: mlir.MlirExternalPass,
    user_data: ?*anyopaque,
) callconv(.C) void {
    var ud = TestExternalPassUserData.castFrom(user_data.?);
    ud.run_call_count += 1;
}

fn testRunExternalFuncPass(
    op: mlir.MlirOperation,
    pass: mlir.MlirExternalPass,
    user_data: ?*anyopaque,
) callconv(.C) void {
    var ud = TestExternalPassUserData.castFrom(user_data.?);
    ud.run_call_count += 1;

    const op_name = mlir.mlirIdentifierStr(mlir.mlirOperationGetName(op));
    if (!mlir.mlirStringRefEqual(op_name, strref("func.func"))) {
        mlir.mlirExternalPassSignalFailure(pass);
    }
}

fn testRunFailingExternalPass(
    _: mlir.MlirOperation,
    pass: mlir.MlirExternalPass,
    user_data: ?*anyopaque,
) callconv(.C) void {
    var ud = TestExternalPassUserData.castFrom(user_data.?);
    ud.run_call_count += 1;
    mlir.mlirExternalPassSignalFailure(pass);
}

fn makeTestExternalPassCallbacks(
    initialize_pass: ?InitializePassProto,
    run_pass: ?RunPassProto,
) callconv(.C) mlir.MlirExternalPassCallbacks {
    return mlir.MlirExternalPassCallbacks{
        .construct = testConstructExternalPass,
        .destruct = testDestructExternalPass,
        .initialize = initialize_pass,
        .clone = testCloneExternalPass,
        .run = run_pass,
    };
}

test "testExternalPass" {
    var fc_runner = try helper.FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);
    registerAllUpstreamDialects(ctx);

    const module_asm =
        \\module {
        \\  func.func @foo(%arg0 : i32) -> i32 {
        \\    %res = arith.addi %arg0, %arg0 : i32
        \\    return %res : i32
        \\  }
        \\}
    ;
    const module = mlir.mlirOperationCreateParse(ctx, strref(module_asm), strref("module_asm"));
    defer mlir.mlirOperationDestroy(module);
    try expect(!mlir.mlirOperationIsNull(module));

    const description = strref("");
    const empty_op_name = strref("");

    const type_id_allocator = mlir.mlirTypeIDAllocatorCreate();
    defer mlir.mlirTypeIDAllocatorDestroy(type_id_allocator);

    // Run a generic pass
    {
        const pass_id = mlir.mlirTypeIDAllocatorAllocateTypeID(type_id_allocator);
        const name = strref("TestExternalPass");
        const argument = strref("test-external-pass");
        var user_data = TestExternalPassUserData{};

        const external_pass = mlir.mlirCreateExternalPass(
            pass_id,
            name,
            argument,
            description,
            empty_op_name,
            0,
            null,
            makeTestExternalPassCallbacks(null, testRunExternalPass),
            &user_data,
        );
        try expect(user_data.construct_call_count == 1);

        const pm = mlir.mlirPassManagerCreate(ctx);
        mlir.mlirPassManagerAddOwnedPass(pm, external_pass);
        const success = mlir.mlirPassManagerRunOnOp(pm, module);
        try expect(mlir.mlirLogicalResultIsSuccess(success));
        try expect(user_data.run_call_count == 1);

        mlir.mlirPassManagerDestroy(pm);
        try expect(user_data.destruct_call_count == user_data.construct_call_count);
    }

    {
        const pass_id = mlir.mlirTypeIDAllocatorAllocateTypeID(type_id_allocator);
        const name = strref("TestExternalFuncPass");
        const argument = strref("test-external-func-pass");
        var user_data = TestExternalPassUserData{};
        var func_handle = mlir.mlirGetDialectHandle__func__();
        const func_op_name = strref("func.func");

        const external_pass = mlir.mlirCreateExternalPass(
            pass_id,
            name,
            argument,
            description,
            func_op_name,
            1,
            &func_handle,
            makeTestExternalPassCallbacks(null, testRunExternalFuncPass),
            &user_data,
        );
        try expect(user_data.construct_call_count == 1);

        const pm = mlir.mlirPassManagerCreate(ctx);
        const nested_func_pm = mlir.mlirPassManagerGetNestedUnder(pm, func_op_name);
        mlir.mlirOpPassManagerAddOwnedPass(nested_func_pm, external_pass);
        const success = mlir.mlirPassManagerRunOnOp(pm, module);
        try expect(mlir.mlirLogicalResultIsSuccess(success));

        // Since this is a nested pass, it can be clone and run in parallel
        try expect(user_data.clone_call_count == user_data.construct_call_count - 1);

        // The pass should only be run once since there is only one func op
        try expect(user_data.run_call_count == 1);

        mlir.mlirPassManagerDestroy(pm);
        try expect(user_data.destruct_call_count == user_data.construct_call_count);
    }

    // Run a pass with `initialize` set
    {
        const pass_id = mlir.mlirTypeIDAllocatorAllocateTypeID(type_id_allocator);
        const name = strref("TestExternalPass");
        const argument = strref("test-external-pass");
        var user_data = TestExternalPassUserData{};

        const external_pass = mlir.mlirCreateExternalPass(
            pass_id,
            name,
            argument,
            description,
            empty_op_name,
            0,
            null,
            makeTestExternalPassCallbacks(
                testInitializeExternalPass,
                testRunExternalPass,
            ),
            &user_data,
        );
        try expect(user_data.construct_call_count == 1);

        const pm = mlir.mlirPassManagerCreate(ctx);
        mlir.mlirPassManagerAddOwnedPass(pm, external_pass);
        const success = mlir.mlirPassManagerRunOnOp(pm, module);
        try expect(mlir.mlirLogicalResultIsSuccess(success));
        try expect(user_data.initialize_call_count == 1);
        try expect(user_data.run_call_count == 1);

        mlir.mlirPassManagerDestroy(pm);
        try expect(user_data.destruct_call_count == user_data.construct_call_count);
    }

    // Run a pass that fails during `initialize`
    {
        const pass_id = mlir.mlirTypeIDAllocatorAllocateTypeID(type_id_allocator);
        const name = strref("TestExternalFailingPass");
        const argument = strref("test-external-failing-pass");
        var user_data = TestExternalPassUserData{};

        const external_pass = mlir.mlirCreateExternalPass(
            pass_id,
            name,
            argument,
            description,
            empty_op_name,
            0,
            null,
            makeTestExternalPassCallbacks(
                testInitializeFailingExternalPass,
                testRunExternalPass,
            ),
            &user_data,
        );
        try expect(user_data.construct_call_count == 1);

        const pm = mlir.mlirPassManagerCreate(ctx);
        mlir.mlirPassManagerAddOwnedPass(pm, external_pass);
        const success = mlir.mlirPassManagerRunOnOp(pm, module);
        try expect(mlir.mlirLogicalResultIsFailure(success));
        try expect(user_data.initialize_call_count == 1);
        try expect(user_data.run_call_count == 0);

        mlir.mlirPassManagerDestroy(pm);
        try expect(user_data.destruct_call_count == user_data.construct_call_count);
    }

    // Run a pass that fails during `run`
    {
        const pass_id = mlir.mlirTypeIDAllocatorAllocateTypeID(type_id_allocator);
        const name = strref("TestExternalFailingPass");
        const argument = strref("test-external-failing-pass");
        var user_data = TestExternalPassUserData{};

        const external_pass = mlir.mlirCreateExternalPass(
            pass_id,
            name,
            argument,
            description,
            empty_op_name,
            0,
            null,
            makeTestExternalPassCallbacks(null, testRunFailingExternalPass),
            &user_data,
        );
        try expect(user_data.construct_call_count == 1);

        const pm = mlir.mlirPassManagerCreate(ctx);
        mlir.mlirPassManagerAddOwnedPass(pm, external_pass);
        const success = mlir.mlirPassManagerRunOnOp(pm, module);
        try expect(mlir.mlirLogicalResultIsFailure(success));
        try expect(user_data.run_call_count == 1);

        mlir.mlirPassManagerDestroy(pm);
        try expect(user_data.destruct_call_count == user_data.construct_call_count);
    }
}
