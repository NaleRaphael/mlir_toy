const std = @import("std");
const c = @import("c.zig");
const helper = @import("helper.zig");

// NOTE: these are the headers required for tests in this file, but it
// aggregate to "c.zig" to avoid multiple invocations of `@cImport`.
// const mlir = @cImport({
//     @cInclude("mlir-c/IR.h");
//     @cInclude("mlir-c/Support.h");
//     @cInclude("mlir-c/ExecutionEngine.h");
//     @cInclude("mlir-c/RegisterEverything.h");
//     @cInclude("mlir-c/BuiltinAttributes.h");
//     @cInclude("mlir-c/BuiltinTypes.h");
//     @cInclude("mlir-c/Conversion.h");
//     @cInclude("mlir-c/IntegerSet.h");
//     @cInclude("mlir-c/Dialect/Func.h");
//     @cInclude("mlir-c/Diagnostics.h");
// });
const mlir = c.mlir;

const test_allocator = std.testing.allocator;
const expect = std.testing.expect;
const FileCheckRunner = helper.FileCheckRunner;

const strref = mlir.mlirStringRefCreateFromCString;

fn registerAllUpstreamDialects(ctx: mlir.MlirContext) void {
    const registry = mlir.mlirDialectRegistryCreate();
    mlir.mlirRegisterAllDialects(registry);
    mlir.mlirContextAppendDialectRegistry(ctx, registry);
    mlir.mlirDialectRegistryDestroy(registry);
}

fn createAndInitContext() mlir.MlirContext {
    const ctx = mlir.mlirContextCreate();
    registerAllUpstreamDialects(ctx);

    _ = mlir.mlirContextGetOrLoadDialect(ctx, strref("func"));
    _ = mlir.mlirContextGetOrLoadDialect(ctx, strref("memref"));
    _ = mlir.mlirContextGetOrLoadDialect(ctx, strref("shape"));
    _ = mlir.mlirContextGetOrLoadDialect(ctx, strref("scf"));

    return ctx;
}

fn populateLoopBody(
    ctx: mlir.MlirContext,
    loop_body: mlir.MlirBlock,
    location: mlir.MlirLocation,
    func_body: mlir.MlirBlock,
) void {
    const iv = mlir.mlirBlockGetArgument(loop_body, 0);
    const func_arg0 = mlir.mlirBlockGetArgument(func_body, 0);
    const func_arg1 = mlir.mlirBlockGetArgument(func_body, 1);
    const f32_type = mlir.mlirTypeParseGet(ctx, strref("f32"));

    var load_LHS_state = mlir.mlirOperationStateGet(strref("memref.load"), location);
    const load_LHS_operands = [_]mlir.MlirValue{ func_arg0, iv };
    mlir.mlirOperationStateAddOperands(&load_LHS_state, 2, &load_LHS_operands);
    mlir.mlirOperationStateAddResults(&load_LHS_state, 1, &f32_type);
    const load_LHS = mlir.mlirOperationCreate(&load_LHS_state);
    mlir.mlirBlockAppendOwnedOperation(loop_body, load_LHS);

    var load_RHS_state = mlir.mlirOperationStateGet(strref("memref.load"), location);
    const load_RHS_operands = [_]mlir.MlirValue{ func_arg1, iv };
    mlir.mlirOperationStateAddOperands(&load_RHS_state, 2, &load_RHS_operands);
    mlir.mlirOperationStateAddResults(&load_RHS_state, 1, &f32_type);
    const load_RHS = mlir.mlirOperationCreate(&load_RHS_state);
    mlir.mlirBlockAppendOwnedOperation(loop_body, load_RHS);

    var add_state = mlir.mlirOperationStateGet(strref("arith.addf"), location);
    const add_operands = [_]mlir.MlirValue{
        mlir.mlirOperationGetResult(load_LHS, 0),
        mlir.mlirOperationGetResult(load_RHS, 0),
    };
    mlir.mlirOperationStateAddOperands(&add_state, 2, &add_operands);
    mlir.mlirOperationStateAddResults(&add_state, 1, &f32_type);
    const add = mlir.mlirOperationCreate(&add_state);
    mlir.mlirBlockAppendOwnedOperation(loop_body, add);

    var store_state = mlir.mlirOperationStateGet(strref("memref.store"), location);
    const store_operands = [_]mlir.MlirValue{ mlir.mlirOperationGetResult(add, 0), func_arg0, iv };
    mlir.mlirOperationStateAddOperands(&store_state, 3, &store_operands);
    const store = mlir.mlirOperationCreate(&store_state);
    mlir.mlirBlockAppendOwnedOperation(loop_body, store);

    var yield_state = mlir.mlirOperationStateGet(strref("scf.yield"), location);
    const yield = mlir.mlirOperationCreate(&yield_state);
    mlir.mlirBlockAppendOwnedOperation(loop_body, yield);
}

// src: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/test/CAPI/ir.c#L84
// NOTE: we moved the dump call outside this function.
fn makeAndDumpAdd(ctx: mlir.MlirContext, location: mlir.MlirLocation) mlir.MlirModule {
    const module_op = mlir.mlirModuleCreateEmpty(location);
    const module_body = mlir.mlirModuleGetBody(module_op);

    const memref_type = mlir.mlirTypeParseGet(ctx, strref("memref<?xf32>"));
    const func_body_arg_types = [_]mlir.MlirType{ memref_type, memref_type };
    const func_body_arg_locs = [_]mlir.MlirLocation{ location, location };
    const func_body_region = mlir.mlirRegionCreate();
    const func_body = mlir.mlirBlockCreate(
        func_body_arg_types.len,
        &func_body_arg_types,
        &func_body_arg_locs,
    );
    mlir.mlirRegionAppendOwnedBlock(func_body_region, func_body);

    const func_type_attr = mlir.mlirAttributeParseGet(
        ctx,
        strref("(memref<?xf32>, memref<?xf32>) -> ()"),
    );
    const func_name_attr = mlir.mlirAttributeParseGet(ctx, strref("\"add\""));
    const func_attrs = [_]mlir.MlirNamedAttribute{
        mlir.mlirNamedAttributeGet(
            mlir.mlirIdentifierGet(ctx, strref("function_type")),
            func_type_attr,
        ),
        mlir.mlirNamedAttributeGet(
            mlir.mlirIdentifierGet(ctx, strref("sym_name")),
            func_name_attr,
        ),
    };
    var func_state = mlir.mlirOperationStateGet(strref("func.func"), location);
    mlir.mlirOperationStateAddAttributes(&func_state, 2, &func_attrs);
    mlir.mlirOperationStateAddOwnedRegions(&func_state, 1, &func_body_region);
    const func = mlir.mlirOperationCreate(&func_state);
    mlir.mlirBlockInsertOwnedOperation(module_body, 0, func);

    const index_type = mlir.mlirTypeParseGet(
        ctx,
        strref("index"),
    );
    const index_zero_literal = mlir.mlirAttributeParseGet(ctx, strref("0 : index"));
    const index_zero_value_attr = mlir.mlirNamedAttributeGet(
        mlir.mlirIdentifierGet(ctx, strref("value")),
        index_zero_literal,
    );
    var const_zero_state = mlir.mlirOperationStateGet(strref("arith.constant"), location);
    mlir.mlirOperationStateAddResults(&const_zero_state, 1, &index_type);
    mlir.mlirOperationStateAddAttributes(&const_zero_state, 1, &index_zero_value_attr);
    const const_zero = mlir.mlirOperationCreate(&const_zero_state);
    mlir.mlirBlockAppendOwnedOperation(func_body, const_zero);

    const func_arg0 = mlir.mlirBlockGetArgument(func_body, 0);
    const const_zero_value = mlir.mlirOperationGetResult(const_zero, 0);
    const dim_operands = [_]mlir.MlirValue{ func_arg0, const_zero_value };
    var dim_state = mlir.mlirOperationStateGet(strref("memref.dim"), location);
    mlir.mlirOperationStateAddOperands(&dim_state, 2, &dim_operands);
    mlir.mlirOperationStateAddResults(&dim_state, 1, &index_type);
    const dim = mlir.mlirOperationCreate(&dim_state);
    mlir.mlirBlockAppendOwnedOperation(func_body, dim);

    const loop_body_region = mlir.mlirRegionCreate();
    const loop_body = mlir.mlirBlockCreate(0, null, null);
    _ = mlir.mlirBlockAddArgument(loop_body, index_type, location);
    mlir.mlirRegionAppendOwnedBlock(loop_body_region, loop_body);

    const index_one_literal = mlir.mlirAttributeParseGet(ctx, strref("1 : index"));
    const index_one_value_attr = mlir.mlirNamedAttributeGet(
        mlir.mlirIdentifierGet(ctx, strref("value")),
        index_one_literal,
    );
    var const_one_state = mlir.mlirOperationStateGet(strref("arith.constant"), location);
    mlir.mlirOperationStateAddResults(&const_one_state, 1, &index_type);
    mlir.mlirOperationStateAddAttributes(&const_one_state, 1, &index_one_value_attr);
    const const_one = mlir.mlirOperationCreate(&const_one_state);
    mlir.mlirBlockAppendOwnedOperation(func_body, const_one);

    const dim_value = mlir.mlirOperationGetResult(dim, 0);
    const const_one_value = mlir.mlirOperationGetResult(const_one, 0);
    const loop_operands = [_]mlir.MlirValue{ const_zero_value, dim_value, const_one_value };
    var loop_state = mlir.mlirOperationStateGet(strref("scf.for"), location);
    mlir.mlirOperationStateAddOperands(&loop_state, 3, &loop_operands);
    mlir.mlirOperationStateAddOwnedRegions(&loop_state, 1, &loop_body_region);
    const loop = mlir.mlirOperationCreate(&loop_state);
    mlir.mlirBlockAppendOwnedOperation(func_body, loop);

    populateLoopBody(ctx, loop_body, location, func_body);

    var ret_state = mlir.mlirOperationStateGet(strref("func.return"), location);
    const ret = mlir.mlirOperationCreate(&ret_state);
    mlir.mlirBlockAppendOwnedOperation(func_body, ret);

    return module_op;
}

const OpListNode = struct {
    op: mlir.MlirOperation,
    next: ?*OpListNode,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        op: mlir.MlirOperation,
        next: ?*OpListNode,
    ) !*Self {
        var node = try allocator.create(Self);
        node.op = op;
        node.next = next;
        node.allocator = allocator;
        return node;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }
};

const ModuleStats = struct {
    num_operations: isize,
    num_attributes: isize,
    num_blocks: isize,
    num_regions: isize,
    num_values: isize,
    num_block_arguments: isize,
    num_op_results: isize,

    pub fn init() ModuleStats {
        return .{
            .num_operations = 0,
            .num_attributes = 0,
            .num_blocks = 0,
            .num_regions = 0,
            .num_values = 0,
            .num_block_arguments = 0,
            .num_op_results = 0,
        };
    }
};

const StatError = error{
    ValueIsAOpResult,
    ValueIsABlockArgument,
    ValueIsNotAOpResult,
    ValueisNotABlockArgument,
    OperationNotEqual,
    ResultNumberMismatched,
    BlockNotEqual,
    BlockArgumentNumberMismatched,
    ValueNumberMismatched,
};

fn collectStatsSingle(
    head: *OpListNode,
    stats: *ModuleStats,
    allocator: std.mem.Allocator,
) !void {
    const operation = head.op;
    stats.num_operations += 1;
    stats.num_values += mlir.mlirOperationGetNumResults(operation);
    stats.num_attributes += mlir.mlirOperationGetNumAttributes(operation);

    const num_regions = mlir.mlirOperationGetNumRegions(operation);
    stats.num_regions += num_regions;

    const num_results = mlir.mlirOperationGetNumResults(operation);
    for (0..@intCast(num_results)) |i| {
        const result = mlir.mlirOperationGetResult(operation, @intCast(i));
        if (!mlir.mlirValueIsAOpResult(result))
            return StatError.ValueIsNotAOpResult;
        if (mlir.mlirValueIsABlockArgument(result))
            return StatError.ValueIsABlockArgument;
        if (!mlir.mlirOperationEqual(operation, mlir.mlirOpResultGetOwner(result)))
            return StatError.OperationNotEqual;
        if (i != mlir.mlirOpResultGetResultNumber(result))
            return StatError.ResultNumberMismatched;
        stats.num_op_results += 1;
    }

    var region = mlir.mlirOperationGetFirstRegion(operation);
    while (!mlir.mlirRegionIsNull(region)) : (region = mlir.mlirRegionGetNextInOperation(region)) {
        var block = mlir.mlirRegionGetFirstBlock(region);
        while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
            stats.num_blocks += 1;
            const num_args = mlir.mlirBlockGetNumArguments(block);
            stats.num_values += num_args;

            for (0..@intCast(num_args)) |j| {
                const arg = mlir.mlirBlockGetArgument(block, @as(isize, @intCast(j)));
                if (!mlir.mlirValueIsABlockArgument(arg))
                    return StatError.ValueisNotABlockArgument;
                if (mlir.mlirValueIsAOpResult(arg))
                    return StatError.ValueIsAOpResult;
                if (!mlir.mlirBlockEqual(block, mlir.mlirBlockArgumentGetOwner(arg)))
                    return StatError.BlockNotEqual;
                if (j != mlir.mlirBlockArgumentGetArgNumber(arg))
                    return StatError.BlockArgumentNumberMismatched;
                stats.num_block_arguments += 1;
            }

            var child = mlir.mlirBlockGetFirstOperation(block);
            while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
                const node = try OpListNode.init(allocator, child, head.next);
                head.next = node;
            }
        }
    }
}

fn collectStats(operation: mlir.MlirOperation, allocator: std.mem.Allocator) !void {
    var head: ?*OpListNode = try OpListNode.init(allocator, operation, null);

    var stats = ModuleStats.init();

    while (head) |h| {
        collectStatsSingle(h, &stats, allocator) catch |err| {
            h.deinit();
            return err;
        };

        const next = h.next;
        h.deinit();
        head = next;
    }

    if (stats.num_values != stats.num_block_arguments + stats.num_op_results) {
        return StatError.ValueNumberMismatched;
    }

    std.debug.print("@stats\n", .{});
    std.debug.print("Number of operations: {d}\n", .{stats.num_operations});
    std.debug.print("Number of attributes: {d}\n", .{stats.num_attributes});
    std.debug.print("Number of blocks: {d}\n", .{stats.num_blocks});
    std.debug.print("Number of regions: {d}\n", .{stats.num_regions});
    std.debug.print("Number of values: {d}\n", .{stats.num_values});
    std.debug.print("Number of block arguments: {d}\n", .{stats.num_block_arguments});
    std.debug.print("Number of op results: {d}\n", .{stats.num_op_results});

    // CHECK-LABEL: @stats
    // CASE-01-b: Number of operations: 12
    // CASE-01-b: Number of attributes: 5
    // CASE-01-b: Number of blocks: 3
    // CASE-01-b: Number of regions: 3
    // CASE-01-b: Number of values: 9
    // CASE-01-b: Number of block arguments: 3
    // CASE-01-b: Number of op results: 6
}

fn printToStderr(str: mlir.MlirStringRef, user_data: ?*anyopaque) callconv(.C) void {
    // XXX: While this function is used as a callback to print MLIR objects,
    // the following approach would contain gibberish text (cannot confirm the
    // root cause of it currently):
    // ```zig
    // std.debug.print("{s}", .{str.data});
    // ```
    //
    // And we cannot use `std.c.fwrite()` since stderr is not exposed in `std.c`.
    // ```zig
    // // The last argument requires to be a `*std.c.FILE`, and it's a opaque
    // // struct. And there is only `STDERR_FILENO` existing in `std.c`.
    // std.c.fwrite(std.data, 1, str.length, ...);
    // ```
    //
    // So we have to make this function all from C library as follows.
    _ = user_data;
    _ = c.stdio.fwrite(str.data, 1, str.length, c.stdio.stderr);
}

fn printFirstOfEach(ctx: mlir.MlirContext, operation: mlir.MlirOperation) !void {
    var op = operation;

    var region = mlir.mlirOperationGetRegion(op, 0);
    var block = mlir.mlirRegionGetFirstBlock(region);
    op = mlir.mlirBlockGetFirstOperation(block);
    region = mlir.mlirOperationGetRegion(op, 0);
    const parent_operation = op;
    block = mlir.mlirRegionGetFirstBlock(region);
    op = mlir.mlirBlockGetFirstOperation(block);

    try expect(mlir.mlirModuleIsNull(mlir.mlirModuleFromOperation(op)));
    try expect(mlir.mlirOperationEqual(
        mlir.mlirOperationGetParentOperation(op),
        parent_operation,
    ));
    try expect(mlir.mlirBlockEqual(
        mlir.mlirOperationGetBlock(op),
        block,
    ));
    try expect(mlir.mlirOperationEqual(
        mlir.mlirBlockGetParentOperation(block),
        parent_operation,
    ));
    try expect(mlir.mlirRegionEqual(
        mlir.mlirBlockGetParentRegion(block),
        region,
    ));

    mlir.mlirBlockPrint(block, printToStderr, null);
    std.debug.print("First operation: ", .{});
    mlir.mlirOperationPrint(op, printToStderr, null);
    std.debug.print("\n", .{});

    // CASE-01-c:   %[[C0:.*]] = arith.constant 0 : index
    // CASE-01-c:   %[[DIM:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xf32>
    // CASE-01-c:   %[[C1:.*]] = arith.constant 1 : index
    // CASE-01-c:   scf.for %[[I:.*]] = %[[C0]] to %[[DIM]] step %[[C1]] {
    // CASE-01-c:     %[[LHS:.*]] = memref.load %{{.*}}[%[[I]]] : memref<?xf32>
    // CASE-01-c:     %[[RHS:.*]] = memref.load %{{.*}}[%[[I]]] : memref<?xf32>
    // CASE-01-c:     %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
    // CASE-01-c:     memref.store %[[SUM]], %{{.*}}[%[[I]]] : memref<?xf32>
    // CASE-01-c:   }
    // CASE-01-c: return
    // CASE-01-c: First operation: {{.*}} = arith.constant 0 : index

    const ident = mlir.mlirOperationGetName(op);
    const ident_str = mlir.mlirIdentifierStr(ident);
    const z_ident_str = std.mem.sliceTo(ident_str.data, 0);
    try expect(std.mem.eql(u8, z_ident_str, "arith.constant"));

    const ident_again = mlir.mlirIdentifierGet(ctx, ident_str);
    try expect(mlir.mlirIdentifierEqual(ident, ident_again));

    const terminator = mlir.mlirBlockGetTerminator(block);
    std.debug.print("Terminator: ", .{});
    mlir.mlirOperationPrint(terminator, printToStderr, null);
    std.debug.print("\n", .{});
    // CASE-01-c: Terminator: func.return

    // Get the attribute by index.
    const named_attr0 = mlir.mlirOperationGetAttribute(op, 0);
    std.debug.print("Get attr 0: ", .{});
    mlir.mlirAttributePrint(named_attr0.attribute, printToStderr, null);
    std.debug.print("\n", .{});
    // CASE-01-c: Get attr 0: 0 : index

    // Re-get the attribute by name.
    const attr0_by_name = mlir.mlirOperationGetAttributeByName(
        op,
        mlir.mlirIdentifierStr(named_attr0.name),
    );
    std.debug.print("Get attr 0 by name: ", .{});
    mlir.mlirAttributePrint(attr0_by_name, printToStderr, null);
    std.debug.print("\n", .{});
    // CASE-01-c: Get attr 0 by name: 0 : index

    // Get a non-existing attribute and assert that it is null (sanity).
    const does_not_exist = mlir.mlirOperationGetAttributeByName(op, strref("does_not_exist"));
    try expect(mlir.mlirAttributeIsNull(does_not_exist));

    // Get result 0 and its type.
    const value = mlir.mlirOperationGetResult(op, 0);
    std.debug.print("Result 0: ", .{});
    mlir.mlirValuePrint(value, printToStderr, null);
    std.debug.print("\n", .{});
    try expect(!mlir.mlirValueIsNull(value));
    // CASE-01-c: Result 0: {{.*}} = arith.constant 0 : index

    const value_type = mlir.mlirValueGetType(value);
    std.debug.print("Result 0 type: ", .{});
    mlir.mlirTypePrint(value_type, printToStderr, null);
    std.debug.print("\n", .{});
    // CASE-01-c: Result 0 type: index

    // Set a custom attribute
    mlir.mlirOperationSetAttributeByName(op, strref("custom_attr"), mlir.mlirBoolAttrGet(ctx, 1));
    std.debug.print("Op with set attr: ", .{});
    mlir.mlirOperationPrint(op, printToStderr, null);
    std.debug.print("\n", .{});
    // CASE-01-c: Op with set attr: {{.*}} {custom_attr = true}

    // Remove the attribute.
    try expect(mlir.mlirOperationRemoveAttributeByName(op, strref("custom_attr")));
    // Remove again, and it should fail.
    try expect(!mlir.mlirOperationRemoveAttributeByName(op, strref("custom_attr")));
    // The removed attr should be null.
    try expect(mlir.mlirAttributeIsNull(mlir.mlirOperationGetAttributeByName(op, strref("custom_attr"))));

    // Add a large attribute to verify printing flags.
    const elts_shape = [_]i64{4};
    const elts_data = [_]i32{ 1, 2, 3, 4 };
    mlir.mlirOperationSetAttributeByName(
        op,
        strref("elts"),
        mlir.mlirDenseElementsAttrInt32Get(
            mlir.mlirRankedTensorTypeGet(
                1,
                &elts_shape,
                mlir.mlirIntegerTypeGet(ctx, 32),
                mlir.mlirAttributeGetNull(),
            ),
            4,
            &elts_data,
        ),
    );
    const flags = mlir.mlirOpPrintingFlagsCreate();
    mlir.mlirOpPrintingFlagsElideLargeElementsAttrs(flags, 2);
    mlir.mlirOpPrintingFlagsPrintGenericOpForm(flags);
    mlir.mlirOpPrintingFlagsEnableDebugInfo(flags, true, false);
    mlir.mlirOpPrintingFlagsUseLocalScope(flags);
    std.debug.print("Op print with all flags: ", .{});
    mlir.mlirOperationPrintWithFlags(op, flags, printToStderr, null);
    std.debug.print("\n", .{});
    // CASE-01-c: Op print with all flags: %{{.*}} = "arith.constant"() <{value = 0 : index}> {elts = dense_resource<__elided__> : tensor<4xi32>} : () -> index loc(unknown)

    mlir.mlirOpPrintingFlagsDestroy(flags);
}

// src: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/test/CAPI/ir.c#482
test "constructAndTraverseIr" {
    const ctx = createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const location = mlir.mlirLocationUnknownGet(ctx);

    // NOTE: module dump is moved below to run with FileCheck
    const module_op = makeAndDumpAdd(ctx, location);
    defer mlir.mlirModuleDestroy(module_op);

    const module = mlir.mlirModuleGetOperation(module_op);
    try expect(!mlir.mlirModuleIsNull(mlir.mlirModuleFromOperation(module)));

    var fc_runner = try FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-01-a");
        mlir.mlirOperationDump(module);

        // CASE-01-a: module {
        // CASE-01-a:   func @add(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>) {
        // CASE-01-a:     %[[C0:.*]] = arith.constant 0 : index
        // CASE-01-a:     %[[DIM:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xf32>
        // CASE-01-a:     %[[C1:.*]] = arith.constant 1 : index
        // CASE-01-a:     scf.for %[[I:.*]] = %[[C0]] to %[[DIM]] step %[[C1]] {
        // CASE-01-a:       %[[LHS:.*]] = memref.load %[[ARG0]][%[[I]]] : memref<?xf32>
        // CASE-01-a:       %[[RHS:.*]] = memref.load %[[ARG1]][%[[I]]] : memref<?xf32>
        // CASE-01-a:       %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
        // CASE-01-a:       memref.store %[[SUM]], %[[ARG0]][%[[I]]] : memref<?xf32>
        // CASE-01-a:     }
        // CASE-01-a:     return
        // CASE-01-a:   }
        // CASE-01-a: }

        const term_a = try fc_runner.cleanup();
        try expect(term_a != null and term_a.?.Exited == 0);

        try fc_runner.runAndWaitForInput("CASE-01-b");
        try collectStats(module, test_allocator);
        const term_b = try fc_runner.cleanup();
        try expect(term_b != null and term_b.?.Exited == 0);

        try fc_runner.runAndWaitForInput("CASE-01-c");
        try printFirstOfEach(ctx, module);
        const term_c = try fc_runner.cleanup();
        try expect(term_c != null and term_c.?.Exited == 0);
    }
}

// src: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/test/CAPI/ir.c#L502
test "buildWithInsertionsAndPrint" {
    const ctx = createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const loc = mlir.mlirLocationUnknownGet(ctx);
    mlir.mlirContextSetAllowUnregisteredDialects(ctx, true);

    const owning_region = mlir.mlirRegionCreate();
    const null_block = mlir.mlirRegionGetFirstBlock(owning_region);
    var state = mlir.mlirOperationStateGet(strref("insertion.order.test"), loc);
    mlir.mlirOperationStateAddOwnedRegions(&state, 1, &owning_region);
    const op = mlir.mlirOperationCreate(&state);
    const region = mlir.mlirOperationGetRegion(op, 0);

    // Use integer types of different bidwidth as block arguments in order to
    // differentiates blocks.
    const i1_t = mlir.mlirIntegerTypeGet(ctx, 1);
    const i2_t = mlir.mlirIntegerTypeGet(ctx, 2);
    const i3_t = mlir.mlirIntegerTypeGet(ctx, 3);
    const i4_t = mlir.mlirIntegerTypeGet(ctx, 4);
    const i5_t = mlir.mlirIntegerTypeGet(ctx, 5);
    const block1 = mlir.mlirBlockCreate(1, &i1_t, &loc);
    const block2 = mlir.mlirBlockCreate(1, &i2_t, &loc);
    const block3 = mlir.mlirBlockCreate(1, &i3_t, &loc);
    const block4 = mlir.mlirBlockCreate(1, &i4_t, &loc);
    const block5 = mlir.mlirBlockCreate(1, &i5_t, &loc);
    // Insert blocks so as to obtain the 1-2-3-4 order,
    mlir.mlirRegionInsertOwnedBlockBefore(region, null_block, block3);
    mlir.mlirRegionInsertOwnedBlockBefore(region, block3, block2);
    mlir.mlirRegionInsertOwnedBlockAfter(region, null_block, block1);
    mlir.mlirRegionInsertOwnedBlockAfter(region, block3, block4);
    mlir.mlirRegionInsertOwnedBlockBefore(region, block3, block5);

    var op1_state = mlir.mlirOperationStateGet(strref("dummy.op1"), loc);
    var op2_state = mlir.mlirOperationStateGet(strref("dummy.op2"), loc);
    var op3_state = mlir.mlirOperationStateGet(strref("dummy.op3"), loc);
    var op4_state = mlir.mlirOperationStateGet(strref("dummy.op4"), loc);
    var op5_state = mlir.mlirOperationStateGet(strref("dummy.op5"), loc);
    var op6_state = mlir.mlirOperationStateGet(strref("dummy.op6"), loc);
    var op7_state = mlir.mlirOperationStateGet(strref("dummy.op7"), loc);
    var op8_state = mlir.mlirOperationStateGet(strref("dummy.op8"), loc);
    const op1 = mlir.mlirOperationCreate(&op1_state);
    const op2 = mlir.mlirOperationCreate(&op2_state);
    const op3 = mlir.mlirOperationCreate(&op3_state);
    const op4 = mlir.mlirOperationCreate(&op4_state);
    const op5 = mlir.mlirOperationCreate(&op5_state);
    const op6 = mlir.mlirOperationCreate(&op6_state);
    const op7 = mlir.mlirOperationCreate(&op7_state);
    const op8 = mlir.mlirOperationCreate(&op8_state);

    // Insert operations in the first block so as to obtain the 1-2-3-4 order.
    const null_operation = mlir.mlirBlockGetFirstOperation(block1);
    try expect(mlir.mlirOperationIsNull(null_operation));
    mlir.mlirBlockInsertOwnedOperationBefore(block1, null_operation, op3);
    mlir.mlirBlockInsertOwnedOperationBefore(block1, op3, op2);
    mlir.mlirBlockInsertOwnedOperationAfter(block1, null_operation, op1);
    mlir.mlirBlockInsertOwnedOperationAfter(block1, op3, op4);

    // Append operations to the rest of blocks to make them non-empty and thus
    // printable.
    mlir.mlirBlockAppendOwnedOperation(block2, op5);
    mlir.mlirBlockAppendOwnedOperation(block3, op6);
    mlir.mlirBlockAppendOwnedOperation(block4, op7);
    mlir.mlirBlockAppendOwnedOperation(block5, op8);

    // Remove block5.
    mlir.mlirBlockDetach(block5);
    mlir.mlirBlockDestroy(block5);

    var fc_runner = try FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-02");
        mlir.mlirOperationDump(op);

        // CHECK-LABEL:  "insertion.order.test"
        // CASE-02:    ^{{.*}}(%{{.*}}: i1
        // CASE-02:      "dummy.op1"
        // CHECK-NEXT:   "dummy.op2"
        // CHECK-NEXT:   "dummy.op3"
        // CHECK-NEXT:   "dummy.op4"
        // CASE-02:    ^{{.*}}(%{{.*}}: i2
        // CASE-02:      "dummy.op5"
        // CHECK-NOT:  ^{{.*}}(%{{.*}}: i5
        // CHECK-NOT:    "dummy.op8"
        // CASE-02:    ^{{.*}}(%{{.*}}: i3
        // CASE-02:      "dummy.op6"
        // CASE-02:    ^{{.*}}(%{{.*}}: i4
        // CASE-02:      "dummy.op7"

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    mlir.mlirOperationDestroy(op);
    mlir.mlirContextSetAllowUnregisteredDialects(ctx, false);
}

test "createOperationWithTypeInference" {
    const ctx = createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const loc = mlir.mlirLocationUnknownGet(ctx);
    const i_attr = mlir.mlirIntegerAttrGet(mlir.mlirIntegerTypeGet(ctx, 32), 4);

    // The shape.const_size op implements result type inference and is only used
    // for that reason.
    var state = mlir.mlirOperationStateGet(strref("shape.const_size"), loc);
    var value_attr = mlir.mlirNamedAttributeGet(
        mlir.mlirIdentifierGet(ctx, strref("value")),
        i_attr,
    );
    mlir.mlirOperationStateAddAttributes(&state, 1, &value_attr);
    mlir.mlirOperationStateEnableResultTypeInference(&state);

    // Expect result type inference to succeed.
    const op = mlir.mlirOperationCreate(&state);
    try expect(!mlir.mlirOperationIsNull(op));

    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    try session.start();
    mlir.mlirTypeDump(mlir.mlirValueGetType(mlir.mlirOperationGetResult(op, 0)));
    try session.stop();
    try expect(session.contentEql("!shape.size\n"));

    mlir.mlirOperationDestroy(op);
}

test "printBuiltinTypes" {
    const ctx = createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const i32_t = mlir.mlirIntegerTypeGet(ctx, 32);
    const si32_t = mlir.mlirIntegerTypeSignedGet(ctx, 32);
    const ui32_t = mlir.mlirIntegerTypeUnsignedGet(ctx, 32);
    try expect(mlir.mlirTypeIsAInteger(i32_t) and !mlir.mlirTypeIsAF32(i32_t));
    try expect(mlir.mlirTypeIsAInteger(si32_t) and mlir.mlirIntegerTypeIsSigned(si32_t));
    try expect(mlir.mlirTypeIsAInteger(ui32_t) and mlir.mlirIntegerTypeIsUnsigned(ui32_t));
    try expect(!mlir.mlirTypeEqual(i32_t, ui32_t) and !mlir.mlirTypeEqual(i32_t, si32_t));
    try expect(mlir.mlirIntegerTypeGetWidth(i32_t) == mlir.mlirIntegerTypeGetWidth(si32_t));
    try session.start();
    mlir.mlirTypeDump(i32_t);
    mlir.mlirTypeDump(si32_t);
    mlir.mlirTypeDump(ui32_t);
    try session.stop();
    try expect(session.contentEql("i32\nsi32\nui32\n"));

    const index_t = mlir.mlirIndexTypeGet(ctx);
    try expect(mlir.mlirTypeIsAIndex(index_t));
    try session.start();
    mlir.mlirTypeDump(index_t);
    try session.stop();
    try expect(session.contentEql("index\n"));

    const bf16_t = mlir.mlirBF16TypeGet(ctx);
    const f16_t = mlir.mlirF16TypeGet(ctx);
    const f32_t = mlir.mlirF32TypeGet(ctx);
    const f64_t = mlir.mlirF64TypeGet(ctx);
    try expect(mlir.mlirTypeIsABF16(bf16_t));
    try expect(mlir.mlirTypeIsAF16(f16_t));
    try expect(mlir.mlirTypeIsAF32(f32_t));
    try expect(mlir.mlirTypeIsAF64(f64_t));
    try session.start();
    mlir.mlirTypeDump(bf16_t);
    mlir.mlirTypeDump(f16_t);
    mlir.mlirTypeDump(f32_t);
    mlir.mlirTypeDump(f64_t);
    try session.stop();
    try expect(session.contentEql("bf16\nf16\nf32\nf64\n"));

    const none_t = mlir.mlirNoneTypeGet(ctx);
    try expect(mlir.mlirTypeIsANone(none_t));
    try session.start();
    mlir.mlirTypeDump(none_t);
    try session.stop();
    try expect(session.contentEql("none\n"));

    const cplx_t = mlir.mlirComplexTypeGet(f32_t);
    try expect(mlir.mlirTypeIsAComplex(cplx_t) and
        mlir.mlirTypeEqual(mlir.mlirComplexTypeGetElementType(cplx_t), f32_t));
    try session.start();
    mlir.mlirTypeDump(cplx_t);
    try session.stop();
    try expect(session.contentEql("complex<f32>\n"));

    const shape = [_]i64{ 2, 3 };
    const vector_t = mlir.mlirVectorTypeGet(shape.len, &shape, f32_t);
    try expect(mlir.mlirTypeIsAVector(vector_t) and mlir.mlirTypeIsAShaped(vector_t));
    try expect(mlir.mlirTypeEqual(mlir.mlirShapedTypeGetElementType(vector_t), f32_t) and
        mlir.mlirShapedTypeHasRank(vector_t) and
        mlir.mlirShapedTypeGetRank(vector_t) == 2 and
        mlir.mlirShapedTypeGetDimSize(vector_t, 0) == 2 and
        !mlir.mlirShapedTypeIsDynamicDim(vector_t, 0) and
        mlir.mlirShapedTypeGetDimSize(vector_t, 1) == 3 and
        mlir.mlirShapedTypeHasStaticShape(vector_t));
    try session.start();
    mlir.mlirTypeDump(vector_t);
    try session.stop();
    try expect(session.contentEql("vector<2x3xf32>\n"));

    const empty_attr = mlir.mlirAttributeGetNull();
    const ranked_tensor_t = mlir.mlirRankedTensorTypeGet(shape.len, &shape, f32_t, empty_attr);
    try expect(mlir.mlirTypeIsATensor(ranked_tensor_t) and
        mlir.mlirTypeIsARankedTensor(ranked_tensor_t) and
        mlir.mlirAttributeIsNull(mlir.mlirRankedTensorTypeGetEncoding(ranked_tensor_t)));
    try session.start();
    mlir.mlirTypeDump(ranked_tensor_t);
    try session.stop();
    try expect(session.contentEql("tensor<2x3xf32>\n"));

    const unranked_tensor_t = mlir.mlirUnrankedTensorTypeGet(f32_t);
    try expect(mlir.mlirTypeIsATensor(unranked_tensor_t) and
        mlir.mlirTypeIsAUnrankedTensor(unranked_tensor_t) and
        !mlir.mlirShapedTypeHasRank(unranked_tensor_t));
    try session.start();
    mlir.mlirTypeDump(unranked_tensor_t);
    try session.stop();
    try expect(session.contentEql("tensor<*xf32>\n"));

    const i64_t = mlir.mlirIntegerTypeGet(ctx, 64);
    const mem_space2 = mlir.mlirIntegerAttrGet(i64_t, 2);
    const mem_ref_t = mlir.mlirMemRefTypeContiguousGet(f32_t, shape.len, &shape, mem_space2);
    try expect(mlir.mlirTypeIsAMemRef(mem_ref_t) and
        mlir.mlirAttributeEqual(mlir.mlirMemRefTypeGetMemorySpace(mem_ref_t), mem_space2));

    const mem_space4 = mlir.mlirIntegerAttrGet(i64_t, 4);
    const unranked_mem_ref_t = mlir.mlirUnrankedMemRefTypeGet(f32_t, mem_space4);
    try expect(mlir.mlirTypeIsAUnrankedMemRef(unranked_mem_ref_t) and
        !mlir.mlirTypeIsAMemRef(unranked_mem_ref_t) and
        mlir.mlirAttributeEqual(mlir.mlirUnrankedMemrefGetMemorySpace(unranked_mem_ref_t), mem_space4));
    try session.start();
    mlir.mlirTypeDump(unranked_mem_ref_t);
    try session.stop();
    try expect(session.contentEql("memref<*xf32, 4>\n"));

    const tuple_element_types = [_]mlir.MlirType{ unranked_mem_ref_t, f32_t };
    const tuple_t = mlir.mlirTupleTypeGet(ctx, 2, &tuple_element_types);
    try expect(mlir.mlirTypeIsATuple(tuple_t) and
        mlir.mlirTupleTypeGetNumTypes(tuple_t) == 2 and
        mlir.mlirTypeEqual(mlir.mlirTupleTypeGetType(tuple_t, 0), unranked_mem_ref_t) and
        mlir.mlirTypeEqual(mlir.mlirTupleTypeGetType(tuple_t, 1), f32_t));

    const i1_t = mlir.mlirIntegerTypeGet(ctx, 1);
    const i16_t = mlir.mlirIntegerTypeGet(ctx, 16);
    const func_inputs_t = [2]mlir.MlirType{ index_t, i1_t };
    const func_results_t = [3]mlir.MlirType{ i16_t, i32_t, i64_t };
    const func_t = mlir.mlirFunctionTypeGet(ctx, 2, &func_inputs_t, 3, &func_results_t);
    try expect(mlir.mlirFunctionTypeGetNumInputs(func_t) == 2);
    try expect(mlir.mlirFunctionTypeGetNumResults(func_t) == 3);
    try expect(mlir.mlirTypeEqual(func_inputs_t[0], mlir.mlirFunctionTypeGetInput(func_t, 0)) and
        mlir.mlirTypeEqual(func_inputs_t[1], mlir.mlirFunctionTypeGetInput(func_t, 1)));
    try expect(mlir.mlirTypeEqual(func_results_t[0], mlir.mlirFunctionTypeGetResult(func_t, 0)) and
        mlir.mlirTypeEqual(func_results_t[1], mlir.mlirFunctionTypeGetResult(func_t, 1)) and
        mlir.mlirTypeEqual(func_results_t[2], mlir.mlirFunctionTypeGetResult(func_t, 2)));
    try session.start();
    mlir.mlirTypeDump(func_t);
    try session.stop();
    try expect(session.contentEql("(index, i1) -> (i16, i32, i64)\n"));

    const namespace = mlir.mlirStringRefCreate("dialect", 7);
    const data = mlir.mlirStringRefCreate("type", 4);
    mlir.mlirContextSetAllowUnregisteredDialects(ctx, true);
    const opaque_t = mlir.mlirOpaqueTypeGet(ctx, namespace, data);
    mlir.mlirContextSetAllowUnregisteredDialects(ctx, false);
    try expect(mlir.mlirTypeIsAOpaque(opaque_t) and
        mlir.mlirStringRefEqual(mlir.mlirOpaqueTypeGetDialectNamespace(opaque_t), namespace) and
        mlir.mlirStringRefEqual(mlir.mlirOpaqueTypeGetData(opaque_t), data));
    try session.start();
    mlir.mlirTypeDump(opaque_t);
    try session.stop();
    try expect(session.contentEql("!dialect.type\n"));
}

fn strRefEqlTo(str_ref: mlir.MlirStringRef, expected: []const u8) bool {
    return std.mem.eql(u8, std.mem.sliceTo(str_ref.data, 0), expected);
}

const EType = enum { Bool, I8, U8, I16, U16, I32, U32, I64, U64, F16, Float, Double, BF16 };

fn DenseElementsAttr(
    comptime etype: EType,
    shape_t: mlir.MlirType,
    n_elements: isize,
    elements: anytype,
) mlir.MlirAttribute {
    return switch (etype) {
        .Bool => mlir.mlirDenseElementsAttrBoolGet(shape_t, n_elements, elements),
        .I8 => mlir.mlirDenseElementsAttrInt8Get(shape_t, n_elements, elements),
        .U8 => mlir.mlirDenseElementsAttrUInt8Get(shape_t, n_elements, elements),
        .I16 => mlir.mlirDenseElementsAttrInt16Get(shape_t, n_elements, elements),
        .U16 => mlir.mlirDenseElementsAttrUInt16Get(shape_t, n_elements, elements),
        .I32 => mlir.mlirDenseElementsAttrInt32Get(shape_t, n_elements, elements),
        .U32 => mlir.mlirDenseElementsAttrUInt32Get(shape_t, n_elements, elements),
        .I64 => mlir.mlirDenseElementsAttrInt64Get(shape_t, n_elements, elements),
        .U64 => mlir.mlirDenseElementsAttrUInt64Get(shape_t, n_elements, elements),
        .F16 => mlir.mlirDenseElementsAttrFloat16Get(shape_t, n_elements, elements),
        .Float => mlir.mlirDenseElementsAttrFloatGet(shape_t, n_elements, elements),
        .Double => mlir.mlirDenseElementsAttrDoubleGet(shape_t, n_elements, elements),
        .BF16 => mlir.mlirDenseElementsAttrBFloat16Get(shape_t, n_elements, elements),
    };
}

fn SplatDenseElementsAttr(comptime etype: EType, shape_t: mlir.MlirType, element: anytype) mlir.MlirAttribute {
    return switch (etype) {
        .Bool => mlir.mlirDenseElementsAttrBoolSplatGet(shape_t, element),
        .I8 => mlir.mlirDenseElementsAttrInt8SplatGet(shape_t, element),
        .U8 => mlir.mlirDenseElementsAttrUInt8SplatGet(shape_t, element),
        .I16 => mlir.mlirDenseElementsAttrInt16SplatGet(shape_t, element),
        .U16 => mlir.mlirDenseElementsAttrUInt16SplatGet(shape_t, element),
        .I32 => mlir.mlirDenseElementsAttrInt32SplatGet(shape_t, element),
        .U32 => mlir.mlirDenseElementsAttrUInt32SplatGet(shape_t, element),
        .I64 => mlir.mlirDenseElementsAttrInt64SplatGet(shape_t, element),
        .U64 => mlir.mlirDenseElementsAttrUInt64SplatGet(shape_t, element),
        .F16 => mlir.mlirDenseElementsAttrFloat16SplatGet(shape_t, element),
        .Float => mlir.mlirDenseElementsAttrFloatSplatGet(shape_t, element),
        .Double => mlir.mlirDenseElementsAttrDoubleSplatGet(shape_t, element),
        .BF16 => mlir.mlirDenseElementsAttrBFloat16SplatGet(shape_t, element),
    };
}

fn DenseBlobElementsAttr(
    comptime etype: EType,
    shape_t: mlir.MlirType,
    name: []const u8,
    n_elements: isize,
    elements: anytype,
) mlir.MlirAttribute {
    const _name = strref(@ptrCast(name));
    return switch (etype) {
        .Bool => @compileError("not supported"),
        .I8 => mlir.mlirUnmanagedDenseInt8ResourceElementsAttrGet(shape_t, _name, n_elements, elements),
        .U8 => mlir.mlirUnmanagedDenseUInt8ResourceElementsAttrGet(shape_t, _name, n_elements, elements),
        .I16 => mlir.mlirUnmanagedDenseInt16ResourceElementsAttrGet(shape_t, _name, n_elements, elements),
        .U16 => mlir.mlirUnmanagedDenseUInt16ResourceElementsAttrGet(shape_t, _name, n_elements, elements),
        .I32 => mlir.mlirUnmanagedDenseInt32ResourceElementsAttrGet(shape_t, _name, n_elements, elements),
        .U32 => mlir.mlirUnmanagedDenseUInt32ResourceElementsAttrGet(shape_t, _name, n_elements, elements),
        .I64 => mlir.mlirUnmanagedDenseInt64ResourceElementsAttrGet(shape_t, _name, n_elements, elements),
        .U64 => mlir.mlirUnmanagedDenseUInt64ResourceElementsAttrGet(shape_t, _name, n_elements, elements),
        .F16 => @compileError("not supported"),
        .Float => mlir.mlirUnmanagedDenseFloatResourceElementsAttrGet(shape_t, _name, n_elements, elements),
        .Double => mlir.mlirUnmanagedDenseDoubleResourceElementsAttrGet(shape_t, _name, n_elements, elements),
        .BF16 => @compileError("not supported"),
    };
}

test "printBuiltinAttributes" {
    const ctx = createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const u1_t = mlir.mlirIntegerTypeGet(ctx, 1);
    const i8_t = mlir.mlirIntegerTypeGet(ctx, 8);
    const u8_t = mlir.mlirIntegerTypeUnsignedGet(ctx, 8);
    const i16_t = mlir.mlirIntegerTypeGet(ctx, 16);
    const u16_t = mlir.mlirIntegerTypeUnsignedGet(ctx, 16);
    const i32_t = mlir.mlirIntegerTypeGet(ctx, 32);
    const u32_t = mlir.mlirIntegerTypeUnsignedGet(ctx, 32);
    const i64_t = mlir.mlirIntegerTypeGet(ctx, 64);
    const u64_t = mlir.mlirIntegerTypeUnsignedGet(ctx, 64);
    const f16_t = mlir.mlirF16TypeGet(ctx);
    const f32_t = mlir.mlirF32TypeGet(ctx);
    const f64_t = mlir.mlirF64TypeGet(ctx);
    const bf16_t = mlir.mlirBF16TypeGet(ctx);

    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const float_attr = mlir.mlirFloatAttrDoubleGet(ctx, mlir.mlirF64TypeGet(ctx), 2.0);
    try expect(mlir.mlirAttributeIsAFloat(float_attr) and
        @abs(mlir.mlirFloatAttrGetValueDouble(float_attr) - 2.0) < 1e-6);
    try session.start();
    mlir.mlirAttributeDump(float_attr);
    try session.stop();
    try expect(session.contentEql("2.000000e+00 : f64\n"));

    const floating_t = mlir.mlirAttributeGetType(float_attr);
    try session.start();
    mlir.mlirTypeDump(floating_t);
    try session.stop();
    try expect(session.contentEql("f64\n"));

    const int_attr = mlir.mlirIntegerAttrGet(mlir.mlirIntegerTypeGet(ctx, 32), 42);
    const sint_attr = mlir.mlirIntegerAttrGet(mlir.mlirIntegerTypeSignedGet(ctx, 8), -1);
    const uint_attr = mlir.mlirIntegerAttrGet(mlir.mlirIntegerTypeUnsignedGet(ctx, 8), 255);
    try expect(mlir.mlirAttributeIsAInteger(int_attr) and
        mlir.mlirIntegerAttrGetValueInt(int_attr) == 42 and
        mlir.mlirIntegerAttrGetValueSInt(sint_attr) == -1 and
        mlir.mlirIntegerAttrGetValueUInt(uint_attr) == 255);
    try session.start();
    mlir.mlirAttributeDump(int_attr);
    mlir.mlirAttributeDump(sint_attr);
    mlir.mlirAttributeDump(uint_attr);
    try session.stop();
    try expect(session.contentEql("42 : i32\n-1 : si8\n255 : ui8\n"));

    const bool_attr = mlir.mlirBoolAttrGet(ctx, 1);
    try expect(mlir.mlirAttributeIsABool(bool_attr) and
        mlir.mlirBoolAttrGetValue(bool_attr));
    try session.start();
    mlir.mlirAttributeDump(bool_attr);
    try session.stop();
    try expect(session.contentEql("true\n"));

    const data = "abcdefghijklmnopqestuvwxyz";
    const none_t = mlir.mlirNoneTypeGet(ctx);
    const opaque_attr = mlir.mlirOpaqueAttrGet(ctx, strref("func"), 3, data, none_t);
    try expect(mlir.mlirAttributeIsAOpaque(opaque_attr) and
        strRefEqlTo(mlir.mlirOpaqueAttrGetDialectNamespace(opaque_attr), "func"));

    const opaque_data: mlir.MlirStringRef = mlir.mlirOpaqueAttrGetData(opaque_attr);
    try expect(opaque_data.length == 3 and
        strRefEqlTo(opaque_data, data[0..opaque_data.length]));
    try session.start();
    mlir.mlirAttributeDump(opaque_attr);
    try session.stop();
    try expect(session.contentEql("#func.abc\n"));

    const str_attr = mlir.mlirStringAttrGet(ctx, mlir.mlirStringRefCreate(data[3..], 2));
    try expect(mlir.mlirAttributeIsAString(str_attr));

    const str_val: mlir.MlirStringRef = mlir.mlirStringAttrGetValue(str_attr);
    try expect(str_val.length == 2 and
        strRefEqlTo(str_val, data[3..][0..str_val.length]));
    try session.start();
    mlir.mlirAttributeDump(str_attr);
    try session.stop();
    try expect(session.contentEql("\"de\"\n"));

    const flat_sym_ref = mlir.mlirFlatSymbolRefAttrGet(
        ctx,
        mlir.mlirStringRefCreate(data[5..], 3),
    );
    try expect(mlir.mlirAttributeIsAFlatSymbolRef(flat_sym_ref));

    const flat_sym_ref_val: mlir.MlirStringRef = mlir.mlirFlatSymbolRefAttrGetValue(flat_sym_ref);
    try expect(flat_sym_ref_val.length == 3 and
        strRefEqlTo(flat_sym_ref_val, data[5..][0..flat_sym_ref_val.length]));
    try session.start();
    mlir.mlirAttributeDump(flat_sym_ref);
    try session.stop();
    try expect(session.contentEql("@fgh\n"));

    const symbols = [_]mlir.MlirAttribute{ flat_sym_ref, flat_sym_ref };
    const sym_ref = mlir.mlirSymbolRefAttrGet(
        ctx,
        mlir.mlirStringRefCreate(data[8..], 2),
        2,
        &symbols,
    );
    const ref_0 = mlir.mlirSymbolRefAttrGetNestedReference(sym_ref, 0);
    const ref_1 = mlir.mlirSymbolRefAttrGetNestedReference(sym_ref, 1);
    try expect(mlir.mlirAttributeIsASymbolRef(sym_ref) and
        mlir.mlirSymbolRefAttrGetNumNestedReferences(sym_ref) == 2 and
        mlir.mlirAttributeEqual(ref_0, flat_sym_ref) and
        mlir.mlirAttributeEqual(ref_1, flat_sym_ref));

    const sym_ref_leaf: mlir.MlirStringRef = mlir.mlirSymbolRefAttrGetLeafReference(sym_ref);
    const sym_ref_root: mlir.MlirStringRef = mlir.mlirSymbolRefAttrGetRootReference(sym_ref);
    try expect(sym_ref_leaf.length == 3 and
        strRefEqlTo(sym_ref_leaf, data[5..][0..sym_ref_leaf.length]) and
        sym_ref_root.length == 2 and
        strRefEqlTo(sym_ref_root, data[8..][0..sym_ref_root.length]));
    try session.start();
    mlir.mlirAttributeDump(sym_ref);
    try session.stop();
    try expect(session.contentEql("@ij::@fgh::@fgh\n"));

    const type_attr = mlir.mlirTypeAttrGet(f32_t);
    try expect(mlir.mlirAttributeIsAType(type_attr) and
        mlir.mlirTypeEqual(f32_t, mlir.mlirTypeAttrGetValue(type_attr)));
    try session.start();
    mlir.mlirAttributeDump(type_attr);
    try session.stop();
    try expect(session.contentEql("f32\n"));

    const unit_attr = mlir.mlirUnitAttrGet(ctx);
    try expect(mlir.mlirAttributeIsAUnit(unit_attr));
    try session.start();
    mlir.mlirAttributeDump(unit_attr);
    try session.stop();
    try expect(session.contentEql("unit\n"));

    const shape = [_]i64{ 1, 2 };
    const bools = [_]c_int{ 0, 1 }; // cannot use `u1` here
    const ints8 = [_]i8{ 0, 1 };
    const uints8 = [_]u8{ 0, 1 };
    const ints16 = [_]i16{ 0, 1 };
    const uints16 = [_]u16{ 0, 1 };
    const ints32 = [_]i32{ 0, 1 };
    const uints32 = [_]u32{ 0, 1 };
    const ints64 = [_]i64{ 0, 1 };
    const uints64 = [_]u64{ 0, 1 };
    const floats = [_]f32{ 0.0, 1.0 };
    const doubles = [_]f64{ 0.0, 1.0 };
    const bf16s = [_]u16{ 0x0, 0x3f80 };
    const f16s = [_]u16{ 0x0, 0x3c00 };

    const encoding = mlir.mlirAttributeGetNull();
    const RTensor = mlir.mlirRankedTensorTypeGet;
    const bool_shape = RTensor(2, &shape, u1_t, encoding);
    const i8_shape = RTensor(2, &shape, i8_t, encoding);
    const u8_shape = RTensor(2, &shape, u8_t, encoding);
    const i16_shape = RTensor(2, &shape, i16_t, encoding);
    const u16_shape = RTensor(2, &shape, u16_t, encoding);
    const i32_shape = RTensor(2, &shape, i32_t, encoding);
    const u32_shape = RTensor(2, &shape, u32_t, encoding);
    const i64_shape = RTensor(2, &shape, i64_t, encoding);
    const u64_shape = RTensor(2, &shape, u64_t, encoding);
    const float_shape = RTensor(2, &shape, f32_t, encoding);
    const double_shape = RTensor(2, &shape, f64_t, encoding);
    const bf16_shape = RTensor(2, &shape, bf16_t, encoding);
    const f16_shape = RTensor(2, &shape, f16_t, encoding);

    const bool_els = DenseElementsAttr(.Bool, bool_shape, 2, &bools);
    const i8_els = DenseElementsAttr(.I8, i8_shape, 2, &ints8);
    const u8_els = DenseElementsAttr(.U8, u8_shape, 2, &uints8);
    const i16_els = DenseElementsAttr(.I16, i16_shape, 2, &ints16);
    const u16_els = DenseElementsAttr(.U16, u16_shape, 2, &uints16);
    const i32_els = DenseElementsAttr(.I32, i32_shape, 2, &ints32);
    const u32_els = DenseElementsAttr(.U32, u32_shape, 2, &uints32);
    const i64_els = DenseElementsAttr(.I64, i64_shape, 2, &ints64);
    const u64_els = DenseElementsAttr(.U64, u64_shape, 2, &uints64);
    const float_els = DenseElementsAttr(.Float, float_shape, 2, &floats);
    const double_els = DenseElementsAttr(.Double, double_shape, 2, &doubles);
    const bf16_els = DenseElementsAttr(.BF16, bf16_shape, 2, &bf16s);
    const f16_els = DenseElementsAttr(.F16, f16_shape, 2, &f16s);

    try expect(mlir.mlirAttributeIsADenseElements(bool_els));
    try expect(mlir.mlirAttributeIsADenseElements(i8_els));
    try expect(mlir.mlirAttributeIsADenseElements(u8_els));
    try expect(mlir.mlirAttributeIsADenseElements(i16_els));
    try expect(mlir.mlirAttributeIsADenseElements(u16_els));
    try expect(mlir.mlirAttributeIsADenseElements(i32_els));
    try expect(mlir.mlirAttributeIsADenseElements(u32_els));
    try expect(mlir.mlirAttributeIsADenseElements(i64_els));
    try expect(mlir.mlirAttributeIsADenseElements(u64_els));
    try expect(mlir.mlirAttributeIsADenseElements(float_els));
    try expect(mlir.mlirAttributeIsADenseElements(double_els));
    try expect(mlir.mlirAttributeIsADenseElements(bf16_els));
    try expect(mlir.mlirAttributeIsADenseElements(f16_els));

    try expect(mlir.mlirDenseElementsAttrGetBoolValue(bool_els, 1));
    try expect(mlir.mlirDenseElementsAttrGetInt8Value(i8_els, 1) == 1);
    try expect(mlir.mlirDenseElementsAttrGetUInt8Value(u8_els, 1) == 1);
    try expect(mlir.mlirDenseElementsAttrGetInt16Value(i16_els, 1) == 1);
    try expect(mlir.mlirDenseElementsAttrGetUInt16Value(u16_els, 1) == 1);
    try expect(mlir.mlirDenseElementsAttrGetInt32Value(i32_els, 1) == 1);
    try expect(mlir.mlirDenseElementsAttrGetUInt32Value(u32_els, 1) == 1);
    try expect(mlir.mlirDenseElementsAttrGetInt64Value(i64_els, 1) == 1);
    try expect(mlir.mlirDenseElementsAttrGetUInt64Value(u64_els, 1) == 1);
    try expect(@abs(mlir.mlirDenseElementsAttrGetFloatValue(float_els, 1) - 1.0) <= 1e-6);
    try expect(@abs(mlir.mlirDenseElementsAttrGetDoubleValue(double_els, 1) - 1.0) <= 1e-6);

    try session.start();
    mlir.mlirAttributeDump(bool_els);
    mlir.mlirAttributeDump(i8_els);
    mlir.mlirAttributeDump(u8_els);
    mlir.mlirAttributeDump(i16_els);
    mlir.mlirAttributeDump(u16_els);
    mlir.mlirAttributeDump(i32_els);
    mlir.mlirAttributeDump(u32_els);
    mlir.mlirAttributeDump(i64_els);
    mlir.mlirAttributeDump(u64_els);
    mlir.mlirAttributeDump(float_els);
    mlir.mlirAttributeDump(double_els);
    mlir.mlirAttributeDump(bf16_els);
    mlir.mlirAttributeDump(f16_els);
    try session.stop();

    // Note that there is a newline character at the end of each line
    try expect(session.contentEql(
        \\dense<[[false, true]]> : tensor<1x2xi1>
        \\dense<[[0, 1]]> : tensor<1x2xi8>
        \\dense<[[0, 1]]> : tensor<1x2xui8>
        \\dense<[[0, 1]]> : tensor<1x2xi16>
        \\dense<[[0, 1]]> : tensor<1x2xui16>
        \\dense<[[0, 1]]> : tensor<1x2xi32>
        \\dense<[[0, 1]]> : tensor<1x2xui32>
        \\dense<[[0, 1]]> : tensor<1x2xi64>
        \\dense<[[0, 1]]> : tensor<1x2xui64>
        \\dense<[[0.000000e+00, 1.000000e+00]]> : tensor<1x2xf32>
        \\dense<[[0.000000e+00, 1.000000e+00]]> : tensor<1x2xf64>
        \\dense<[[0.000000e+00, 1.000000e+00]]> : tensor<1x2xbf16>
        \\dense<[[0.000000e+00, 1.000000e+00]]> : tensor<1x2xf16>
        \\
    ));

    const splat_bool = SplatDenseElementsAttr(.Bool, bool_shape, true);
    const splat_int8 = SplatDenseElementsAttr(.I8, i8_shape, 1);
    const splat_uint8 = SplatDenseElementsAttr(.U8, u8_shape, 1);
    const splat_int32 = SplatDenseElementsAttr(.I32, i32_shape, 1);
    const splat_uint32 = SplatDenseElementsAttr(.U32, u32_shape, 1);
    const splat_int64 = SplatDenseElementsAttr(.I64, i64_shape, 1);
    const splat_uint64 = SplatDenseElementsAttr(.U64, u64_shape, 1);
    const splat_float = SplatDenseElementsAttr(.Float, float_shape, 1.0);
    const splat_double = SplatDenseElementsAttr(.Double, double_shape, 1.0);

    const isDenseElementsAndSplat = struct {
        fn func(attr: mlir.MlirAttribute) bool {
            return mlir.mlirAttributeIsADenseElements(attr) and
                mlir.mlirDenseElementsAttrIsSplat(attr);
        }
    }.func;

    try expect(isDenseElementsAndSplat(splat_bool));
    try expect(isDenseElementsAndSplat(splat_int8));
    try expect(isDenseElementsAndSplat(splat_uint8));
    try expect(isDenseElementsAndSplat(splat_int32));
    try expect(isDenseElementsAndSplat(splat_uint32));
    try expect(isDenseElementsAndSplat(splat_int64));
    try expect(isDenseElementsAndSplat(splat_uint64));
    try expect(isDenseElementsAndSplat(splat_float));
    try expect(isDenseElementsAndSplat(splat_double));

    try expect(mlir.mlirDenseElementsAttrGetBoolSplatValue(splat_bool) == 1);
    try expect(mlir.mlirDenseElementsAttrGetInt8SplatValue(splat_int8) == 1);
    try expect(mlir.mlirDenseElementsAttrGetUInt8SplatValue(splat_uint8) == 1);
    try expect(mlir.mlirDenseElementsAttrGetInt32SplatValue(splat_int32) == 1);
    try expect(mlir.mlirDenseElementsAttrGetUInt32SplatValue(splat_uint32) == 1);
    try expect(mlir.mlirDenseElementsAttrGetInt64SplatValue(splat_int64) == 1);
    try expect(mlir.mlirDenseElementsAttrGetUInt64SplatValue(splat_uint64) == 1);
    try expect(@abs(mlir.mlirDenseElementsAttrGetFloatSplatValue(splat_float) - 1.0) < 1e-6);
    try expect(@abs(mlir.mlirDenseElementsAttrGetDoubleSplatValue(splat_double) - 1.0) < 1e-6);

    const getRawData = struct {
        const GetRawDataError = error{NotExists};
        fn func(comptime T: type, attr: mlir.MlirAttribute) ![*]const T {
            if (mlir.mlirDenseElementsAttrGetRawData(attr)) |raw| {
                return @alignCast(@ptrCast(raw));
            } else {
                return GetRawDataError.NotExists;
            }
        }
    }.func;

    const checkRawData = struct {
        fn func(comptime T: type, raw_data: [*]const T, expected: []const T) bool {
            const ti = @typeInfo(T);
            for (raw_data, expected) |a, b| {
                switch (ti) {
                    .Int => if (a != b) return false,
                    .Float => if (@abs(a - b) > 1e-6) return false,
                    else => @compileError("not supported type"),
                }
            }
            return true;
        }
    }.func;

    const u8_raw = try getRawData(u8, u8_els);
    const i8_raw = try getRawData(i8, i8_els);
    const u32_raw = try getRawData(u32, u32_els);
    const i32_raw = try getRawData(i32, i32_els);
    const u64_raw = try getRawData(u64, u64_els);
    const i64_raw = try getRawData(i64, i64_els);
    const float_raw = try getRawData(f32, float_els);
    const double_raw = try getRawData(f64, double_els);
    const bf16_raw = try getRawData(u16, bf16_els);
    const f16_raw = try getRawData(u16, f16_els);

    try expect(checkRawData(u8, u8_raw, &.{ 0, 1 }));
    try expect(checkRawData(i8, i8_raw, &.{ 0, 1 }));
    try expect(checkRawData(u32, u32_raw, &.{ 0, 1 }));
    try expect(checkRawData(i32, i32_raw, &.{ 0, 1 }));
    try expect(checkRawData(u64, u64_raw, &.{ 0, 1 }));
    try expect(checkRawData(i64, i64_raw, &.{ 0, 1 }));
    try expect(checkRawData(f32, float_raw, &.{ 0, 1 }));
    try expect(checkRawData(f64, double_raw, &.{ 0, 1 }));
    try expect(checkRawData(u16, bf16_raw, &.{ 0, 0x3f80 }));
    try expect(checkRawData(u16, f16_raw, &.{ 0, 0x3c00 }));

    try session.start();
    mlir.mlirAttributeDump(splat_bool);
    mlir.mlirAttributeDump(splat_uint8);
    mlir.mlirAttributeDump(splat_int8);
    mlir.mlirAttributeDump(splat_uint32);
    mlir.mlirAttributeDump(splat_int32);
    mlir.mlirAttributeDump(splat_uint64);
    mlir.mlirAttributeDump(splat_int64);
    mlir.mlirAttributeDump(splat_float);
    mlir.mlirAttributeDump(splat_double);
    try session.stop();
    try expect(session.contentEql(
        \\dense<true> : tensor<1x2xi1>
        \\dense<1> : tensor<1x2xui8>
        \\dense<1> : tensor<1x2xi8>
        \\dense<1> : tensor<1x2xui32>
        \\dense<1> : tensor<1x2xi32>
        \\dense<1> : tensor<1x2xui64>
        \\dense<1> : tensor<1x2xi64>
        \\dense<1.000000e+00> : tensor<1x2xf32>
        \\dense<1.000000e+00> : tensor<1x2xf64>
        \\
    ));

    const getValueFromElements = struct {
        fn func(attr: mlir.MlirAttribute, rank: isize, idxs: []const u64) mlir.MlirAttribute {
            return mlir.mlirElementsAttrGetValue(attr, rank, @constCast(@ptrCast(idxs)));
        }
    }.func;

    try session.start();
    mlir.mlirAttributeDump(getValueFromElements(float_els, 2, &uints64));
    mlir.mlirAttributeDump(getValueFromElements(double_els, 2, &uints64));
    mlir.mlirAttributeDump(getValueFromElements(bf16_els, 2, &uints64));
    mlir.mlirAttributeDump(getValueFromElements(f16_els, 2, &uints64));
    try session.stop();
    try expect(session.contentEql(
        \\1.000000e+00 : f32
        \\1.000000e+00 : f64
        \\1.000000e+00 : bf16
        \\1.000000e+00 : f16
        \\
    ));

    const indices = [_]i64{ 0, 1 };
    const one: i64 = 1;
    const indices_attr = mlir.mlirDenseElementsAttrInt64Get(
        RTensor(2, &shape, i64_t, encoding),
        2,
        &indices,
    );
    const values_attr = mlir.mlirDenseElementsAttrFloatGet(
        RTensor(1, &one, f32_t, encoding),
        1,
        &floats,
    );
    const sparse_attr = mlir.mlirSparseElementsAttribute(
        RTensor(2, &shape, f32_t, encoding),
        indices_attr,
        values_attr,
    );
    try session.start();
    mlir.mlirAttributeDump(sparse_attr);
    try session.stop();
    try expect(session.contentEql("sparse<[[0, 1]], 0.000000e+00> : tensor<1x2xf32>\n"));

    const bool_array = mlir.mlirDenseBoolArrayGet(ctx, 2, &bools);
    const i8_array = mlir.mlirDenseI8ArrayGet(ctx, 2, &ints8);
    const i16_array = mlir.mlirDenseI16ArrayGet(ctx, 2, &ints16);
    const i32_array = mlir.mlirDenseI32ArrayGet(ctx, 2, &ints32);
    const i64_array = mlir.mlirDenseI64ArrayGet(ctx, 2, &ints64);
    const float_array = mlir.mlirDenseF32ArrayGet(ctx, 2, &floats);
    const double_array = mlir.mlirDenseF64ArrayGet(ctx, 2, &doubles);

    try expect(mlir.mlirAttributeIsADenseBoolArray(bool_array));
    try expect(mlir.mlirAttributeIsADenseI8Array(i8_array));
    try expect(mlir.mlirAttributeIsADenseI16Array(i16_array));
    try expect(mlir.mlirAttributeIsADenseI32Array(i32_array));
    try expect(mlir.mlirAttributeIsADenseI64Array(i64_array));
    try expect(mlir.mlirAttributeIsADenseF32Array(float_array));
    try expect(mlir.mlirAttributeIsADenseF64Array(double_array));

    try expect(mlir.mlirDenseArrayGetNumElements(bool_array) == 2);
    try expect(mlir.mlirDenseArrayGetNumElements(i8_array) == 2);
    try expect(mlir.mlirDenseArrayGetNumElements(i16_array) == 2);
    try expect(mlir.mlirDenseArrayGetNumElements(i32_array) == 2);
    try expect(mlir.mlirDenseArrayGetNumElements(i64_array) == 2);
    try expect(mlir.mlirDenseArrayGetNumElements(float_array) == 2);
    try expect(mlir.mlirDenseArrayGetNumElements(double_array) == 2);

    try expect(mlir.mlirDenseBoolArrayGetElement(bool_array, 1));
    try expect(mlir.mlirDenseI8ArrayGetElement(i8_array, 1) == 1);
    try expect(mlir.mlirDenseI16ArrayGetElement(i16_array, 1) == 1);
    try expect(mlir.mlirDenseI32ArrayGetElement(i32_array, 1) == 1);
    try expect(mlir.mlirDenseI64ArrayGetElement(i64_array, 1) == 1);
    try expect(@abs(mlir.mlirDenseF32ArrayGetElement(float_array, 1) - 1.0) < 1e-6);
    try expect(@abs(mlir.mlirDenseF64ArrayGetElement(double_array, 1) - 1.0) < 1e-6);

    const layout_strides = [_]i64{ 5, 7, 13 };
    const strided_layout_attr = mlir.mlirStridedLayoutAttrGet(ctx, 42, 3, &layout_strides[0]);
    try session.start();
    mlir.mlirAttributeDump(strided_layout_attr);
    try session.stop();
    try expect(session.contentEql("strided<[5, 7, 13], offset: 42>\n"));

    try expect(mlir.mlirStridedLayoutAttrGetOffset(strided_layout_attr) == 42);
    try expect(mlir.mlirStridedLayoutAttrGetNumStrides(strided_layout_attr) == 3);
    try expect(mlir.mlirStridedLayoutAttrGetStride(strided_layout_attr, 0) == 5);
    try expect(mlir.mlirStridedLayoutAttrGetStride(strided_layout_attr, 1) == 7);
    try expect(mlir.mlirStridedLayoutAttrGetStride(strided_layout_attr, 2) == 13);

    const u8_blob = DenseBlobElementsAttr(.U8, u8_shape, "resource_ui8", 2, &uints8);
    const i8_blob = DenseBlobElementsAttr(.I8, i8_shape, "resource_i8", 2, &ints8);
    const u16_blob = DenseBlobElementsAttr(.U16, u16_shape, "resource_ui16", 2, &uints16);
    const i16_blob = DenseBlobElementsAttr(.I16, i16_shape, "resource_i16", 2, &ints16);
    const u32_blob = DenseBlobElementsAttr(.U32, u32_shape, "resource_ui32", 2, &uints32);
    const i32_blob = DenseBlobElementsAttr(.I32, i32_shape, "resource_i32", 2, &ints32);
    const u64_blob = DenseBlobElementsAttr(.U64, u64_shape, "resource_ui64", 2, &uints64);
    const i64_blob = DenseBlobElementsAttr(.I64, i64_shape, "resource_i64", 2, &ints64);
    const float_blob = DenseBlobElementsAttr(.Float, float_shape, "resource_f32", 2, &floats);
    const double_blob = DenseBlobElementsAttr(.Double, double_shape, "resource_f64", 2, &doubles);

    try session.start();
    mlir.mlirAttributeDump(u8_blob);
    mlir.mlirAttributeDump(i8_blob);
    mlir.mlirAttributeDump(u16_blob);
    mlir.mlirAttributeDump(i16_blob);
    mlir.mlirAttributeDump(u32_blob);
    mlir.mlirAttributeDump(i32_blob);
    mlir.mlirAttributeDump(u64_blob);
    mlir.mlirAttributeDump(i64_blob);
    mlir.mlirAttributeDump(float_blob);
    mlir.mlirAttributeDump(double_blob);
    try session.stop();
    try expect(session.contentEql(
        \\dense_resource<resource_ui8> : tensor<1x2xui8>
        \\dense_resource<resource_i8> : tensor<1x2xi8>
        \\dense_resource<resource_ui16> : tensor<1x2xui16>
        \\dense_resource<resource_i16> : tensor<1x2xi16>
        \\dense_resource<resource_ui32> : tensor<1x2xui32>
        \\dense_resource<resource_i32> : tensor<1x2xi32>
        \\dense_resource<resource_ui64> : tensor<1x2xui64>
        \\dense_resource<resource_i64> : tensor<1x2xi64>
        \\dense_resource<resource_f32> : tensor<1x2xf32>
        \\dense_resource<resource_f64> : tensor<1x2xf64>
        \\
    ));

    try expect(mlir.mlirDenseUInt8ResourceElementsAttrGetValue(u8_blob, 1) == 1);
    try expect(mlir.mlirDenseInt8ResourceElementsAttrGetValue(i8_blob, 1) == 1);
    try expect(mlir.mlirDenseUInt16ResourceElementsAttrGetValue(u16_blob, 1) == 1);
    try expect(mlir.mlirDenseInt16ResourceElementsAttrGetValue(i16_blob, 1) == 1);
    try expect(mlir.mlirDenseUInt32ResourceElementsAttrGetValue(u32_blob, 1) == 1);
    try expect(mlir.mlirDenseInt32ResourceElementsAttrGetValue(i32_blob, 1) == 1);
    try expect(mlir.mlirDenseUInt64ResourceElementsAttrGetValue(u64_blob, 1) == 1);
    try expect(mlir.mlirDenseInt64ResourceElementsAttrGetValue(i64_blob, 1) == 1);
    try expect(@abs(mlir.mlirDenseFloatResourceElementsAttrGetValue(float_blob, 1) - 1.0) < 1e-6);
    try expect(@abs(mlir.mlirDenseDoubleResourceElementsAttrGetValue(double_blob, 1) - 1.0) < 1e-6);

    const loc = mlir.mlirLocationUnknownGet(ctx);
    const loc_attr = mlir.mlirLocationGetAttribute(loc);
    try expect(mlir.mlirAttributeIsALocation(loc_attr));
}

test "printAffineMap" {
    const ctx = createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const empty_affine_map = mlir.mlirAffineMapEmptyGet(ctx);
    const affine_map = mlir.mlirAffineMapZeroResultGet(ctx, 3, 2);
    const const_affine_map = mlir.mlirAffineMapConstantGet(ctx, 2);
    const multi_dim_identity_affine_map = mlir.mlirAffineMapMultiDimIdentityGet(ctx, 3);
    const minor_identity_affine_map = mlir.mlirAffineMapMinorIdentityGet(ctx, 3, 2);
    var permutation = [_]u32{ 1, 2, 0 };
    const permutation_affine_map = mlir.mlirAffineMapPermutationGet(ctx, permutation.len, &permutation);

    try session.start();
    mlir.mlirAffineMapDump(empty_affine_map);
    mlir.mlirAffineMapDump(affine_map);
    mlir.mlirAffineMapDump(const_affine_map);
    mlir.mlirAffineMapDump(multi_dim_identity_affine_map);
    mlir.mlirAffineMapDump(minor_identity_affine_map);
    mlir.mlirAffineMapDump(permutation_affine_map);
    try session.stop();
    try expect(session.contentEql(
        \\() -> ()
        \\(d0, d1, d2)[s0, s1] -> ()
        \\() -> (2)
        \\(d0, d1, d2) -> (d0, d1, d2)
        \\(d0, d1, d2) -> (d1, d2)
        \\(d0, d1, d2) -> (d1, d2, d0)
        \\
    ));

    const isIdentity = mlir.mlirAffineMapIsIdentity;
    try expect(isIdentity(empty_affine_map));
    try expect(!isIdentity(affine_map));
    try expect(!isIdentity(const_affine_map));
    try expect(isIdentity(multi_dim_identity_affine_map));
    try expect(!isIdentity(minor_identity_affine_map));
    try expect(!isIdentity(permutation_affine_map));

    const isMinorIdentity = mlir.mlirAffineMapIsMinorIdentity;
    try expect(isMinorIdentity(empty_affine_map));
    try expect(!isMinorIdentity(affine_map));
    try expect(isMinorIdentity(multi_dim_identity_affine_map));
    try expect(isMinorIdentity(minor_identity_affine_map));
    try expect(!isMinorIdentity(permutation_affine_map));

    const isEmpty = mlir.mlirAffineMapIsEmpty;
    try expect(isEmpty(empty_affine_map));
    try expect(!isEmpty(affine_map));
    try expect(!isEmpty(const_affine_map));
    try expect(!isEmpty(multi_dim_identity_affine_map));
    try expect(!isEmpty(minor_identity_affine_map));
    try expect(!isEmpty(permutation_affine_map));

    const isSingleConstant = mlir.mlirAffineMapIsSingleConstant;
    try expect(!isSingleConstant(empty_affine_map));
    try expect(!isSingleConstant(affine_map));
    try expect(isSingleConstant(const_affine_map));
    try expect(!isSingleConstant(multi_dim_identity_affine_map));
    try expect(!isSingleConstant(minor_identity_affine_map));
    try expect(!isSingleConstant(permutation_affine_map));

    try expect(mlir.mlirAffineMapGetSingleConstantResult(const_affine_map) == 2);

    const getNumDims = mlir.mlirAffineMapGetNumDims;
    try expect(getNumDims(empty_affine_map) == 0);
    try expect(getNumDims(affine_map) == 3);
    try expect(getNumDims(const_affine_map) == 0);
    try expect(getNumDims(multi_dim_identity_affine_map) == 3);
    try expect(getNumDims(minor_identity_affine_map) == 3);
    try expect(getNumDims(permutation_affine_map) == 3);

    const getNumSymbols = mlir.mlirAffineMapGetNumSymbols;
    try expect(getNumSymbols(empty_affine_map) == 0);
    try expect(getNumSymbols(affine_map) == 2);
    try expect(getNumSymbols(const_affine_map) == 0);
    try expect(getNumSymbols(multi_dim_identity_affine_map) == 0);
    try expect(getNumSymbols(minor_identity_affine_map) == 0);
    try expect(getNumSymbols(permutation_affine_map) == 0);

    const getNumResults = mlir.mlirAffineMapGetNumResults;
    try expect(getNumResults(empty_affine_map) == 0);
    try expect(getNumResults(affine_map) == 0);
    try expect(getNumResults(const_affine_map) == 1);
    try expect(getNumResults(multi_dim_identity_affine_map) == 3);
    try expect(getNumResults(minor_identity_affine_map) == 2);
    try expect(getNumResults(permutation_affine_map) == 3);

    const getNumInputs = mlir.mlirAffineMapGetNumInputs;
    try expect(getNumInputs(empty_affine_map) == 0);
    try expect(getNumInputs(affine_map) == 5);
    try expect(getNumInputs(const_affine_map) == 0);
    try expect(getNumInputs(multi_dim_identity_affine_map) == 3);
    try expect(getNumInputs(minor_identity_affine_map) == 3);
    try expect(getNumInputs(permutation_affine_map) == 3);

    const isProjPermOrPerm = struct {
        fn func(affine: mlir.MlirAffineMap, is_proj_perm: bool, is_perm: bool) bool {
            return mlir.mlirAffineMapIsProjectedPermutation(affine) == is_proj_perm and
                mlir.mlirAffineMapIsPermutation(affine) == is_perm;
        }
    }.func;
    try expect(isProjPermOrPerm(empty_affine_map, true, true));
    try expect(isProjPermOrPerm(affine_map, false, false));
    try expect(isProjPermOrPerm(const_affine_map, false, false));
    try expect(isProjPermOrPerm(multi_dim_identity_affine_map, true, true));
    try expect(isProjPermOrPerm(minor_identity_affine_map, true, false));
    try expect(isProjPermOrPerm(permutation_affine_map, true, true));

    var sub = [_]isize{1};
    const sub_map = mlir.mlirAffineMapGetSubMap(multi_dim_identity_affine_map, sub.len, &sub);
    const major_sub_map = mlir.mlirAffineMapGetMajorSubMap(multi_dim_identity_affine_map, 1);
    const minor_sub_map = mlir.mlirAffineMapGetMinorSubMap(multi_dim_identity_affine_map, 1);

    try session.start();
    mlir.mlirAffineMapDump(sub_map);
    mlir.mlirAffineMapDump(major_sub_map);
    mlir.mlirAffineMapDump(minor_sub_map);
    try session.stop();
    try expect(session.contentEql(
        \\(d0, d1, d2) -> (d1)
        \\(d0, d1, d2) -> (d0)
        \\(d0, d1, d2) -> (d2)
        \\
    ));
}

test "printAffineExpr" {
    const ctx = createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const affine_dim_expr = mlir.mlirAffineDimExprGet(ctx, 5);
    const affine_symbol_expr = mlir.mlirAffineSymbolExprGet(ctx, 5);
    const affine_constant_expr = mlir.mlirAffineConstantExprGet(ctx, 5);
    const affine_add_expr = mlir.mlirAffineAddExprGet(affine_dim_expr, affine_symbol_expr);
    const affine_mul_expr = mlir.mlirAffineMulExprGet(affine_dim_expr, affine_symbol_expr);
    const affine_mod_expr = mlir.mlirAffineModExprGet(affine_dim_expr, affine_symbol_expr);
    const affine_floor_div_expr = mlir.mlirAffineFloorDivExprGet(affine_dim_expr, affine_symbol_expr);
    const affine_ceil_div_expr = mlir.mlirAffineCeilDivExprGet(affine_dim_expr, affine_symbol_expr);

    try session.start();
    mlir.mlirAffineExprDump(affine_dim_expr);
    mlir.mlirAffineExprDump(affine_symbol_expr);
    mlir.mlirAffineExprDump(affine_constant_expr);
    mlir.mlirAffineExprDump(affine_add_expr);
    mlir.mlirAffineExprDump(affine_mul_expr);
    mlir.mlirAffineExprDump(affine_mod_expr);
    mlir.mlirAffineExprDump(affine_floor_div_expr);
    mlir.mlirAffineExprDump(affine_ceil_div_expr);
    try session.stop();
    try expect(session.contentEql(
        \\d5
        \\s5
        \\5
        \\d5 + s5
        \\d5 * s5
        \\d5 mod s5
        \\d5 floordiv s5
        \\d5 ceildiv s5
        \\
    ));

    try session.start();
    mlir.mlirAffineExprDump(mlir.mlirAffineBinaryOpExprGetLHS(affine_add_expr));
    mlir.mlirAffineExprDump(mlir.mlirAffineBinaryOpExprGetRHS(affine_add_expr));
    try session.stop();
    try expect(session.contentEql("d5\ns5\n"));

    try expect(mlir.mlirAffineDimExprGetPosition(affine_dim_expr) == 5);
    try expect(mlir.mlirAffineSymbolExprGetPosition(affine_symbol_expr) == 5);
    try expect(mlir.mlirAffineConstantExprGetValue(affine_constant_expr) == 5);

    const isSymbolOrConst = mlir.mlirAffineExprIsSymbolicOrConstant;
    try expect(!isSymbolOrConst(affine_dim_expr));
    try expect(isSymbolOrConst(affine_symbol_expr));
    try expect(isSymbolOrConst(affine_constant_expr));
    try expect(!isSymbolOrConst(affine_add_expr));
    try expect(!isSymbolOrConst(affine_mul_expr));
    try expect(!isSymbolOrConst(affine_mod_expr));
    try expect(!isSymbolOrConst(affine_floor_div_expr));
    try expect(!isSymbolOrConst(affine_ceil_div_expr));

    const isPureAffine = mlir.mlirAffineExprIsPureAffine;
    try expect(isPureAffine(affine_dim_expr));
    try expect(isPureAffine(affine_symbol_expr));
    try expect(isPureAffine(affine_constant_expr));
    try expect(isPureAffine(affine_add_expr));
    try expect(!isPureAffine(affine_mul_expr));
    try expect(!isPureAffine(affine_mod_expr));
    try expect(!isPureAffine(affine_floor_div_expr));
    try expect(!isPureAffine(affine_ceil_div_expr));

    const getLargestDivisor = mlir.mlirAffineExprGetLargestKnownDivisor;
    try expect(getLargestDivisor(affine_dim_expr) == 1);
    try expect(getLargestDivisor(affine_symbol_expr) == 1);
    try expect(getLargestDivisor(affine_constant_expr) == 5);
    try expect(getLargestDivisor(affine_add_expr) == 1);
    try expect(getLargestDivisor(affine_mul_expr) == 1);
    try expect(getLargestDivisor(affine_mod_expr) == 1);
    try expect(getLargestDivisor(affine_floor_div_expr) == 1);
    try expect(getLargestDivisor(affine_ceil_div_expr) == 1);

    const isMultipleOf = mlir.mlirAffineExprIsMultipleOf;
    try expect(isMultipleOf(affine_dim_expr, 1));
    try expect(isMultipleOf(affine_symbol_expr, 1));
    try expect(isMultipleOf(affine_constant_expr, 5));
    try expect(isMultipleOf(affine_add_expr, 1));
    try expect(isMultipleOf(affine_mul_expr, 1));
    try expect(isMultipleOf(affine_mod_expr, 1));
    try expect(isMultipleOf(affine_floor_div_expr, 1));
    try expect(isMultipleOf(affine_ceil_div_expr, 1));

    const isFunctionOfDim = mlir.mlirAffineExprIsFunctionOfDim;
    try expect(isFunctionOfDim(affine_dim_expr, 5));
    try expect(!isFunctionOfDim(affine_symbol_expr, 5));
    try expect(!isFunctionOfDim(affine_constant_expr, 5));
    try expect(isFunctionOfDim(affine_add_expr, 5));
    try expect(isFunctionOfDim(affine_mul_expr, 5));
    try expect(isFunctionOfDim(affine_mod_expr, 5));
    try expect(isFunctionOfDim(affine_floor_div_expr, 5));
    try expect(isFunctionOfDim(affine_ceil_div_expr, 5));

    try expect(mlir.mlirAffineExprIsAAdd(affine_add_expr));
    try expect(mlir.mlirAffineExprIsAMul(affine_mul_expr));
    try expect(mlir.mlirAffineExprIsAMod(affine_mod_expr));
    try expect(mlir.mlirAffineExprIsAFloorDiv(affine_floor_div_expr));
    try expect(mlir.mlirAffineExprIsACeilDiv(affine_ceil_div_expr));
    try expect(mlir.mlirAffineExprIsABinary(affine_add_expr));
    try expect(mlir.mlirAffineExprIsAConstant(affine_constant_expr));
    try expect(mlir.mlirAffineExprIsASymbol(affine_symbol_expr));

    const other_dim_expr = mlir.mlirAffineDimExprGet(ctx, 5);
    try expect(mlir.mlirAffineExprEqual(affine_dim_expr, other_dim_expr));
    try expect(!mlir.mlirAffineExprIsNull(affine_dim_expr));
}

test "affineMapFromExprs" {
    const ctx = createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const affine_dim_expr = mlir.mlirAffineDimExprGet(ctx, 0);
    const affine_symbol_expr = mlir.mlirAffineSymbolExprGet(ctx, 1);
    var exprs = [_]mlir.MlirAffineExpr{ affine_dim_expr, affine_symbol_expr };
    const map = mlir.mlirAffineMapGet(ctx, 3, 3, 2, &exprs);

    try session.start();
    mlir.mlirAffineMapDump(map);
    try session.stop();
    try expect(session.contentEql("(d0, d1, d2)[s0, s1, s2] -> (d0, s1)\n"));

    const exprEql = mlir.mlirAffineExprEqual;
    const mapGetResult = mlir.mlirAffineMapGetResult;
    try expect(mlir.mlirAffineMapGetNumResults(map) == 2);
    try expect(exprEql(mapGetResult(map, 0), affine_dim_expr));
    try expect(exprEql(mapGetResult(map, 1), affine_symbol_expr));

    const affine_dim2_expr = mlir.mlirAffineDimExprGet(ctx, 1);
    const composed = mlir.mlirAffineExprCompose(affine_dim2_expr, map);
    try session.start();
    mlir.mlirAffineExprDump(composed);
    try session.stop();
    try expect(session.contentEql("s1\n"));
    try expect(exprEql(composed, affine_symbol_expr));
}

test "printIntegerSet" {
    const ctx = createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    var fc_runner = try FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    const empty_set = mlir.mlirIntegerSetEmptyGet(ctx, 2, 1);
    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-08-a");
        mlir.mlirIntegerSetDump(empty_set);

        // We follow the order of test in CAPI/ir.c to assume this is the 8-th case.
        // CASE-08-a: (d0, d1)[s0] : (1 == 0)

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }
    try expect(mlir.mlirIntegerSetIsCanonicalEmpty(empty_set));

    const another_empty_set = mlir.mlirIntegerSetEmptyGet(ctx, 2, 1);
    try expect(mlir.mlirIntegerSetEqual(empty_set, another_empty_set));

    const neg_one = mlir.mlirAffineConstantExprGet(ctx, -1);
    const neg_forty_two = mlir.mlirAffineConstantExprGet(ctx, -42);
    const d0 = mlir.mlirAffineDimExprGet(ctx, 0);
    const d1 = mlir.mlirAffineDimExprGet(ctx, 1);
    const s0 = mlir.mlirAffineSymbolExprGet(ctx, 0);
    const neg_s0 = mlir.mlirAffineMulExprGet(neg_one, s0);
    const d0_minus_s0 = mlir.mlirAffineAddExprGet(d0, neg_s0);
    const d1_minus_42 = mlir.mlirAffineAddExprGet(d1, neg_forty_two);
    var constraints = [_]mlir.MlirAffineExpr{ d0_minus_s0, d1_minus_42 };
    var flags = [_]bool{ true, false };

    const set = mlir.mlirIntegerSetGet(ctx, 2, 1, 2, &constraints, &flags);
    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-08-b");
        mlir.mlirIntegerSetDump(set);

        // CASE-08-b: (d0, d1)[s0] : (
        // CHECK-DAG: d0 - s0 == 0
        // CHECK-DAG: d1 - 42 >= 0

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    const s1 = mlir.mlirAffineSymbolExprGet(ctx, 1);
    var repl = [_]mlir.MlirAffineExpr{ d0, s1 };
    const replaced = mlir.mlirIntegerSetReplaceGet(set, &repl, &s0, 1, 2);
    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-08-c");
        mlir.mlirIntegerSetDump(replaced);

        // CASE-08-c: (d0)[s0, s1] : (
        // CHECK-DAG: d0 - s0 == 0
        // CHECK-DAG: s1 - 42 >= 0

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    try expect(mlir.mlirIntegerSetGetNumDims(set) == 2);
    try expect(mlir.mlirIntegerSetGetNumDims(replaced) == 1);

    try expect(mlir.mlirIntegerSetGetNumSymbols(set) == 1);
    try expect(mlir.mlirIntegerSetGetNumSymbols(replaced) == 2);

    try expect(mlir.mlirIntegerSetGetNumInputs(set) == 3);
    try expect(mlir.mlirIntegerSetGetNumConstraints(set) == 2);
    try expect(mlir.mlirIntegerSetGetNumEqualities(set) == 1);
    try expect(mlir.mlirIntegerSetGetNumInequalities(set) == 1);

    const cstr1 = mlir.mlirIntegerSetGetConstraint(set, 0);
    const cstr2 = mlir.mlirIntegerSetGetConstraint(set, 1);
    const is_eq1 = mlir.mlirIntegerSetIsConstraintEq(set, 0);
    const is_eq2 = mlir.mlirIntegerSetIsConstraintEq(set, 1);
    try expect(mlir.mlirAffineExprEqual(cstr1, if (is_eq1) d0_minus_s0 else d1_minus_42));
    try expect(mlir.mlirAffineExprEqual(cstr2, if (is_eq2) d0_minus_s0 else d1_minus_42));
}

test "registerOnlyStd" {
    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    // The built-in dialect is always loaded
    try expect(mlir.mlirContextGetNumLoadedDialects(ctx) == 1);

    // NOTE: This API is generated by `MLIR_DECLARE_DIALECT_REGISTRATION_CAPI`
    // marco [1], which takes the dialect API name as the namespace.
    // In this case, the namespace is `func`, so that we requires to include
    // the header file "mlir-c/Dialect/func.h".
    //
    // [1]: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/include/mlir-c/IR.h#L185-L208
    const _std_handle = mlir.mlirGetDialectHandle__func__();

    var ns = mlir.mlirDialectHandleGetNamespace(_std_handle);
    var _std = mlir.mlirContextGetOrLoadDialect(ctx, ns);
    try expect(mlir.mlirDialectIsNull(_std));

    mlir.mlirDialectHandleRegisterDialect(_std_handle, ctx);
    ns = mlir.mlirDialectHandleGetNamespace(_std_handle);
    _std = mlir.mlirContextGetOrLoadDialect(ctx, ns);
    try expect(!mlir.mlirDialectIsNull(_std));

    const also_std = mlir.mlirDialectHandleLoadDialect(_std_handle, ctx);
    try expect(mlir.mlirDialectEqual(_std, also_std));

    const strRefEql = struct {
        fn func(str_ref_1: mlir.MlirStringRef, str_ref_2: mlir.MlirStringRef) bool {
            return std.mem.eql(
                u8,
                std.mem.sliceTo(str_ref_1.data, 0),
                std.mem.sliceTo(str_ref_2.data, 0),
            );
        }
    }.func;
    const _std_ns = mlir.mlirDialectGetNamespace(_std);
    const also_std_ns = mlir.mlirDialectHandleGetNamespace(_std_handle);
    try expect(strRefEql(_std_ns, also_std_ns));

    const isRegisteredOp = struct {
        fn func(_ctx: mlir.MlirContext, op_name: []const u8) bool {
            return mlir.mlirContextIsRegisteredOperation(_ctx, strref(@ptrCast(op_name)));
        }
    }.func;
    try expect(isRegisteredOp(ctx, "func.call"));
    try expect(!isRegisteredOp(ctx, "func.not_existing_op"));
    try expect(!isRegisteredOp(ctx, "non_existing_dialect.not_existing_op"));
}

test "testBackreferences" {
    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    mlir.mlirContextSetAllowUnregisteredDialects(ctx, true);
    const loc = mlir.mlirLocationUnknownGet(ctx);

    var op_state = mlir.mlirOperationStateGet(strref("invalid.op"), loc);
    const region = mlir.mlirRegionCreate();
    const block = mlir.mlirBlockCreate(0, null, null);
    mlir.mlirRegionAppendOwnedBlock(region, block);
    mlir.mlirOperationStateAddOwnedRegions(&op_state, 1, &region);

    const op = mlir.mlirOperationCreate(&op_state);
    defer mlir.mlirOperationDestroy(op);
    const ident = mlir.mlirIdentifierGet(ctx, strref("identifier"));

    try expect(mlir.mlirContextEqual(ctx, mlir.mlirOperationGetContext(op)));
    try expect(mlir.mlirOperationEqual(op, mlir.mlirBlockGetParentOperation(block)));
    try expect(mlir.mlirContextEqual(ctx, mlir.mlirIdentifierGetContext(ident)));
}

test "testOperands" {
    var fc_runner = try FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    registerAllUpstreamDialects(ctx);

    _ = mlir.mlirContextGetOrLoadDialect(ctx, strref("arith"));
    _ = mlir.mlirContextGetOrLoadDialect(ctx, strref("test"));
    const loc = mlir.mlirLocationUnknownGet(ctx);
    const index_t = mlir.mlirIndexTypeGet(ctx);
    const id_value = mlir.mlirIdentifierGet(ctx, strref("value"));

    // Create some constants to use as oeprands.
    const index_zero_literal = mlir.mlirAttributeParseGet(ctx, strref("0 : index"));
    const index_zero_value_attr = mlir.mlirNamedAttributeGet(id_value, index_zero_literal);
    var const_zero_state = mlir.mlirOperationStateGet(strref("arith.constant"), loc);
    mlir.mlirOperationStateAddResults(&const_zero_state, 1, &index_t);
    mlir.mlirOperationStateAddAttributes(&const_zero_state, 1, &index_zero_value_attr);
    const const_zero = mlir.mlirOperationCreate(&const_zero_state);
    const const_zero_value = mlir.mlirOperationGetResult(const_zero, 0);

    const index_one_literal = mlir.mlirAttributeParseGet(ctx, strref("1 : index"));
    const index_one_value_attr = mlir.mlirNamedAttributeGet(id_value, index_one_literal);
    var const_one_state = mlir.mlirOperationStateGet(strref("arith.constant"), loc);
    mlir.mlirOperationStateAddResults(&const_one_state, 1, &index_t);
    mlir.mlirOperationStateAddAttributes(&const_one_state, 1, &index_one_value_attr);
    const const_one = mlir.mlirOperationCreate(&const_one_state);
    const const_one_value = mlir.mlirOperationGetResult(const_one, 0);

    // Create the operation under test.
    mlir.mlirContextSetAllowUnregisteredDialects(ctx, true);
    var op_state = mlir.mlirOperationStateGet(strref("dummy.op"), loc);
    var initial_operands = [_]mlir.MlirValue{const_zero_value};
    mlir.mlirOperationStateAddOperands(&op_state, 1, &initial_operands);
    const op = mlir.mlirOperationCreate(&op_state);

    // Test Operand APIs.
    const num_operands = mlir.mlirOperationGetNumOperands(op);
    try expect(num_operands == 1);

    const op_operand1 = mlir.mlirOperationGetOperand(op, 0);
    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-11-a");
        std.debug.print("Original operand: ", .{});
        mlir.mlirValuePrint(op_operand1, printToStderr, null);
        // CASE-11-a: Original operand: {{.+}} arith.constant 0 : index

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    mlir.mlirOperationSetOperand(op, 0, const_one_value);
    const op_operand2 = mlir.mlirOperationGetOperand(op, 0);
    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-11-b");
        std.debug.print("Updated operand: ", .{});
        mlir.mlirValuePrint(op_operand2, printToStderr, null);
        // CASE-11-b: Updated operand: {{.+}} arith.constant 1 : index

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    const use1 = mlir.mlirValueGetFirstUse(op_operand1);
    try expect(mlir.mlirOpOperandIsNull(use1));

    var use2 = mlir.mlirValueGetFirstUse(op_operand2);
    try expect(!mlir.mlirOpOperandIsNull(use2));

    const use2_owner = mlir.mlirOpOperandGetOwner(use2);
    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-11-c");
        std.debug.print("Use owner: ", .{});
        mlir.mlirOperationPrint(use2_owner, printToStderr, null);
        // CASE-11-c: Use owner: "dummy.op"

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    try expect(mlir.mlirOpOperandGetOperandNumber(use2) == 0);

    use2 = mlir.mlirOpOperandGetNextUse(use2);
    try expect(mlir.mlirOpOperandIsNull(use2));

    var op2_state = mlir.mlirOperationStateGet(strref("dummy.op2"), loc);
    const initial_operands2 = [_]mlir.MlirValue{const_one_value};
    mlir.mlirOperationStateAddOperands(&op2_state, 1, &initial_operands2);
    const op2 = mlir.mlirOperationCreate(&op2_state);

    var use3 = mlir.mlirValueGetFirstUse(const_one_value);
    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-11-d");
        std.debug.print("First use owner: ", .{});
        mlir.mlirOperationPrint(mlir.mlirOpOperandGetOwner(use3), printToStderr, null);
        // CASE-11-d: First use owner: "dummy.op2"

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    use3 = mlir.mlirOpOperandGetNextUse(mlir.mlirValueGetFirstUse(const_one_value));
    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-11-e");
        std.debug.print("Second use owner: ", .{});
        mlir.mlirOperationPrint(mlir.mlirOpOperandGetOwner(use3), printToStderr, null);
        // CASE-11-e: Second use owner: "dummy.op"

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    const index_two_literal = mlir.mlirAttributeParseGet(ctx, strref("2 : index"));
    const index_two_value_attr = mlir.mlirNamedAttributeGet(id_value, index_two_literal);
    var const_two_state = mlir.mlirOperationStateGet(strref("arith.constant"), loc);
    mlir.mlirOperationStateAddResults(&const_two_state, 1, &index_t);
    mlir.mlirOperationStateAddAttributes(&const_two_state, 1, &index_two_value_attr);
    const const_two = mlir.mlirOperationCreate(&const_two_state);
    const const_two_value = mlir.mlirOperationGetResult(const_two, 0);

    mlir.mlirValueReplaceAllUsesOfWith(const_one_value, const_two_value);

    use3 = mlir.mlirValueGetFirstUse(const_one_value);
    try expect(mlir.mlirOpOperandIsNull(use3));

    var use4 = mlir.mlirValueGetFirstUse(const_two_value);
    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-11-f");
        std.debug.print("First replacement use owner: ", .{});
        mlir.mlirOperationPrint(mlir.mlirOpOperandGetOwner(use4), printToStderr, null);
        // CASE-11-f: First replacement use owner: "dummy.op"

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    use4 = mlir.mlirOpOperandGetNextUse(mlir.mlirValueGetFirstUse(const_two_value));
    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-11-g");
        std.debug.print("Second replacement use owner: ", .{});
        mlir.mlirOperationPrint(mlir.mlirOpOperandGetOwner(use4), printToStderr, null);
        // CASE-11-g: Second replacement use owner: "dummy.op2"

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    mlir.mlirOperationDestroy(op);
    mlir.mlirOperationDestroy(op2);
    mlir.mlirOperationDestroy(const_zero);
    mlir.mlirOperationDestroy(const_one);
    mlir.mlirOperationDestroy(const_two);
}

test "testClone" {
    var fc_runner = try FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    registerAllUpstreamDialects(ctx);
    _ = mlir.mlirContextGetOrLoadDialect(ctx, strref("func"));
    _ = mlir.mlirContextGetOrLoadDialect(ctx, strref("arith"));

    const loc = mlir.mlirLocationUnknownGet(ctx);
    const index_t = mlir.mlirIndexTypeGet(ctx);
    const id_value = mlir.mlirIdentifierGet(ctx, strref("value"));

    const index_zero_literal = mlir.mlirAttributeParseGet(ctx, strref("0 : index"));
    const index_zero_value_attr = mlir.mlirNamedAttributeGet(id_value, index_zero_literal);
    var const_zero_state = mlir.mlirOperationStateGet(strref("arith.constant"), loc);
    mlir.mlirOperationStateAddResults(&const_zero_state, 1, &index_t);
    mlir.mlirOperationStateAddAttributes(&const_zero_state, 1, &index_zero_value_attr);
    const const_zero = mlir.mlirOperationCreate(&const_zero_state);

    const index_one_literal = mlir.mlirAttributeParseGet(ctx, strref("1 : index"));
    const const_one = mlir.mlirOperationClone(const_zero);
    mlir.mlirOperationSetAttributeByName(const_one, strref("value"), index_one_literal);

    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-12");
        mlir.mlirOperationPrint(const_zero, printToStderr, null);
        mlir.mlirOperationPrint(const_one, printToStderr, null);

        // CASE-12: arith.constant 0 : index
        // CASE-12: arith.constant 1 : index

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    mlir.mlirOperationDestroy(const_zero);
    mlir.mlirOperationDestroy(const_one);
}

test "testTypeID" {
    const ctx = createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const i32_t = mlir.mlirIntegerTypeGet(ctx, 32);
    const i32_id = mlir.mlirTypeGetTypeID(i32_t);
    const u32_t = mlir.mlirIntegerTypeUnsignedGet(ctx, 32);
    const u32_id = mlir.mlirTypeGetTypeID(u32_t);
    const f32_t = mlir.mlirF32TypeGet(ctx);
    const f32_id = mlir.mlirTypeGetTypeID(f32_t);
    const i32_attr = mlir.mlirIntegerAttrGet(i32_t, 1);
    const i32_attr_id = mlir.mlirAttributeGetTypeID(i32_attr);

    try expect(!mlir.mlirTypeIDIsNull(i32_id));
    try expect(!mlir.mlirTypeIDIsNull(u32_id));
    try expect(!mlir.mlirTypeIDIsNull(f32_id));
    try expect(!mlir.mlirTypeIDIsNull(i32_attr_id));

    try expect(mlir.mlirTypeIDEqual(i32_id, u32_id) and
        mlir.mlirTypeIDHashValue(i32_id) == mlir.mlirTypeIDHashValue(u32_id));
    try expect(!mlir.mlirTypeIDEqual(i32_id, f32_id));
    try expect(!mlir.mlirTypeIDEqual(i32_id, i32_attr_id));

    const loc = mlir.mlirLocationUnknownGet(ctx);
    const index_t = mlir.mlirIndexTypeGet(ctx);
    const id_value = mlir.mlirIdentifierGet(ctx, strref("value"));

    // Create an registered operation, which should have a type ID.
    const index_zero_literal = mlir.mlirAttributeParseGet(ctx, strref("0 : index"));
    const index_zero_value_attr = mlir.mlirNamedAttributeGet(id_value, index_zero_literal);
    var const_zero_state = mlir.mlirOperationStateGet(strref("arith.constant"), loc);
    mlir.mlirOperationStateAddResults(&const_zero_state, 1, &index_t);
    mlir.mlirOperationStateAddAttributes(&const_zero_state, 1, &index_zero_value_attr);
    const const_zero = mlir.mlirOperationCreate(&const_zero_state);

    try expect(mlir.mlirOperationVerify(const_zero));
    try expect(!mlir.mlirOperationIsNull(const_zero));

    const registered_op_id = mlir.mlirOperationGetTypeID(const_zero);

    try expect(!mlir.mlirTypeIDIsNull(registered_op_id));

    // Create an unregistered operation, which should not have a type ID.
    mlir.mlirContextSetAllowUnregisteredDialects(ctx, true);
    var op_state = mlir.mlirOperationStateGet(strref("dummy.op"), loc);
    const unregistered_op = mlir.mlirOperationCreate(&op_state);
    try expect(!mlir.mlirOperationIsNull(unregistered_op));

    const unregistered_op_id = mlir.mlirOperationGetTypeID(unregistered_op);
    try expect(mlir.mlirTypeIDIsNull(unregistered_op_id));

    mlir.mlirOperationDestroy(const_zero);
    mlir.mlirOperationDestroy(unregistered_op);
}

test "testSymbolTable" {
    var fc_runner = try FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    const ctx = createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const module_string =
        \\func.func private @foo()
        \\func.func private @bar()
    ;
    const other_module_string =
        \\func.func private @qux()
        \\func.func private @foo()
    ;

    const module = mlir.mlirModuleCreateParse(ctx, strref(module_string));
    const other_module = mlir.mlirModuleCreateParse(ctx, strref(other_module_string));

    const symbol_table = mlir.mlirSymbolTableCreate(mlir.mlirModuleGetOperation(module));

    const func_foo = mlir.mlirSymbolTableLookup(symbol_table, strref("foo"));
    try expect(!mlir.mlirOperationIsNull(func_foo));

    const func_bar = mlir.mlirSymbolTableLookup(symbol_table, strref("bar"));
    try expect(!mlir.mlirOperationEqual(func_foo, func_bar));

    const missing = mlir.mlirSymbolTableLookup(symbol_table, strref("qux"));
    try expect(mlir.mlirOperationIsNull(missing));

    const module_body = mlir.mlirModuleGetBody(module);
    const other_module_body = mlir.mlirModuleGetBody(other_module);
    const operation = mlir.mlirBlockGetFirstOperation(other_module_body);
    mlir.mlirOperationRemoveFromParent(operation);
    mlir.mlirBlockAppendOwnedOperation(module_body, operation);

    // At this moment, the operation is still missing from the symbol table.
    const still_missing = mlir.mlirSymbolTableLookup(symbol_table, strref("qux"));
    try expect(mlir.mlirOperationIsNull(still_missing));

    // After it's added to the symbol table, and not only the operation with
    // which the table is associated, it can be looked up.
    _ = mlir.mlirSymbolTableInsert(symbol_table, operation);
    const func_qux = mlir.mlirSymbolTableLookup(symbol_table, strref("qux"));
    try expect(mlir.mlirOperationEqual(operation, func_qux));

    // Erasing from the symbol table also removes the operation.
    mlir.mlirSymbolTableErase(symbol_table, func_bar);
    const now_missing = mlir.mlirSymbolTableLookup(symbol_table, strref("bar"));
    try expect(mlir.mlirOperationIsNull(now_missing));

    // Adding a symbol with the same name to the table should rename.
    const duplicate_name_op = mlir.mlirBlockGetFirstOperation(other_module_body);
    mlir.mlirOperationRemoveFromParent(duplicate_name_op);
    mlir.mlirBlockAppendOwnedOperation(module_body, duplicate_name_op);
    const new_name = mlir.mlirSymbolTableInsert(symbol_table, duplicate_name_op);
    const new_name_str = mlir.mlirStringAttrGetValue(new_name);
    try expect(!mlir.mlirStringRefEqual(new_name_str, strref("foo")));

    const updated_name = mlir.mlirOperationGetAttributeByName(
        duplicate_name_op,
        mlir.mlirSymbolTableGetSymbolAttributeName(),
    );
    try expect(mlir.mlirAttributeEqual(updated_name, new_name));

    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-14");
        mlir.mlirOperationDump(mlir.mlirModuleGetOperation(module));
        mlir.mlirOperationDump(mlir.mlirModuleGetOperation(other_module));

        // CASE-14: module
        // CASE-14:   func private @foo
        // CASE-14:   func private @qux
        // CASE-14:   func private @foo{{.+}}
        // CASE-14: module
        // CHECK-NOT: @qux
        // CHECK-NOT: @foo

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }

    mlir.mlirSymbolTableDestroy(symbol_table);
    mlir.mlirModuleDestroy(module);
    mlir.mlirModuleDestroy(other_module);
}

test "testDialectRegistry" {
    const registry = mlir.mlirDialectRegistryCreate();
    try expect(!mlir.mlirDialectRegistryIsNull(registry));

    const _std_handle = mlir.mlirGetDialectHandle__func__();
    mlir.mlirDialectHandleInsertDialect(_std_handle, registry);
    defer mlir.mlirDialectRegistryDestroy(registry);

    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    // XXX: We got a different behavior comparing with the original test case.
    // There will be an registered dialect `builtin` existing right after a
    // context is created, while in the original test case there is none.
    //
    // We are not sure about what results in this difference, and there is also
    // no available C-API to get the list of registered dialect names. (we have
    // to manually add it for this purpose)
    //
    // If you want to get the list of registered dialect names, please add the
    // following code to the corresponding files and rebuild MLIR:
    // ```c++
    // // file: mlir/include/mlir-c/IR.h
    // MLIR_CAPI_EXPORTED MlirStringRef
    // mlirContextGetRegisteredDialectName(MlirContext context, unsigned int idx);
    //
    // // file: mlir/lib/CAPI/IR/IR.cpp
    // MlirStringRef mlirContextGetRegisteredDialectName(MlirContext context, unsigned int idx) {
    //     return wrap(unwrap(context)->getAvailableDialects()[idx]);
    // }
    // ```
    //
    // then add these lines to print the list of dialect names:
    // ```zig
    // const num: usize = @intCast(mlir.mlirContextGetNumRegisteredDialects(ctx));
    // for (0..num) |i| {
    //     const name_strref = mlir.mlirContextGetRegisteredDialectName(ctx, i);
    //     const name = std.mem.slice(name_strref.data, 0);
    //     std.debug.print("dialect name: {s}\n", .{name});
    // }
    // ```
    try expect(mlir.mlirContextGetNumRegisteredDialects(ctx) == 1);

    mlir.mlirContextAppendDialectRegistry(ctx, registry);
    try expect(mlir.mlirContextGetNumRegisteredDialects(ctx) == 2);
}

test "testExplicitThreadPools" {
    const thread_pool = mlir.mlirLlvmThreadPoolCreate();
    const registry = mlir.mlirDialectRegistryCreate();
    mlir.mlirRegisterAllDialects(registry);

    // Create context without threading enabled.
    const ctx = mlir.mlirContextCreateWithRegistry(registry, false);

    mlir.mlirContextSetThreadPool(ctx, thread_pool);

    mlir.mlirContextDestroy(ctx);
    mlir.mlirDialectRegistryDestroy(registry);
    mlir.mlirLlvmThreadPoolDestroy(thread_pool);
}

// Wrap a diagnostic into additional text we can match against.
fn errorHandler(diagnostic: mlir.MlirDiagnostic, user_data: ?*anyopaque) callconv(.C) mlir.MlirLogicalResult {
    if (user_data) |ud| {
        const data: *u32 = @alignCast(@ptrCast(ud));
        std.debug.print("processing diagnostic (userData: {d}) <<\n", .{data.*});
    } else {
        std.debug.print("processing diagnostic (userData: null) <<\n", .{});
    }

    mlir.mlirDiagnosticPrint(diagnostic, printToStderr, null);
    std.debug.print("\n", .{});
    const loc = mlir.mlirDiagnosticGetLocation(diagnostic);
    mlir.mlirLocationPrint(loc, printToStderr, null);
    std.debug.assert(mlir.mlirDiagnosticGetNumNotes(diagnostic) == 0);

    if (user_data) |ud| {
        const data: *u32 = @alignCast(@ptrCast(ud));
        std.debug.print("\n>> end of diagnostic (userData: {d})\n", .{data.*});
    } else {
        std.debug.print("\n>> end of diagnostic (userData: null)\n", .{});
    }
    return mlir.mlirLogicalResultSuccess();
}

// Logs when the delete user data callback is called
fn deleteUserData(user_data: ?*anyopaque) callconv(.C) void {
    if (user_data) |ud| {
        const data: *u32 = @alignCast(@ptrCast(ud));
        std.debug.print("deleting user data (userData: {d})\n", .{data.*});
    } else {
        std.debug.print("deleting user data (userData: null)\n", .{});
    }
}

test "testDiagnostics" {
    var fc_runner = try FileCheckRunner.init(test_allocator, @src());
    defer fc_runner.deinit();

    if (fc_runner.canRun()) {
        try fc_runner.runAndWaitForInput("CASE-17");

        const ctx = mlir.mlirContextCreate();
        defer mlir.mlirContextDestroy(ctx);

        var user_data: u32 = 42;
        const id = mlir.mlirContextAttachDiagnosticHandler(
            ctx,
            errorHandler,
            @ptrCast(&user_data),
            deleteUserData,
        );

        const unknown_loc = mlir.mlirLocationUnknownGet(ctx);
        mlir.mlirEmitError(unknown_loc, "test diagnostics");

        const unknown_attr = mlir.mlirLocationGetAttribute(unknown_loc);
        const unknown_clone = mlir.mlirLocationFromAttribute(unknown_attr);
        mlir.mlirEmitError(unknown_clone, "test clone");

        const file_line_col_loc = mlir.mlirLocationFileLineColGet(ctx, strref("file.c"), 1, 2);
        mlir.mlirEmitError(file_line_col_loc, "test diagnostics");

        const call_site_loc = mlir.mlirLocationCallSiteGet(
            mlir.mlirLocationFileLineColGet(ctx, strref("other-file.c"), 2, 3),
            file_line_col_loc,
        );
        mlir.mlirEmitError(call_site_loc, "test diagnostics");

        const _null = mlir.MlirLocation{ .ptr = null };
        const name_loc = mlir.mlirLocationNameGet(ctx, strref("named"), _null);
        mlir.mlirEmitError(name_loc, "test diagnostics");

        const locs = [_]mlir.MlirLocation{ name_loc, call_site_loc };
        const null_attr = mlir.MlirAttribute{ .ptr = null };
        const fused_loc = mlir.mlirLocationFusedGet(ctx, 2, &locs, null_attr);
        mlir.mlirEmitError(fused_loc, "test diagnostics");

        mlir.mlirContextDetachDiagnosticHandler(ctx, id);
        mlir.mlirEmitError(unknown_loc, "more test diagnostics");

        // CASE-17: processing diagnostic (userData: 42) <<
        // CASE-17:   test diagnostics
        // CASE-17:   loc(unknown)
        // CASE-17: processing diagnostic (userData: 42) <<
        // CASE-17:   test clone
        // CASE-17:   loc(unknown)
        // CASE-17: >> end of diagnostic (userData: 42)
        // CASE-17: processing diagnostic (userData: 42) <<
        // CASE-17:   test diagnostics
        // CASE-17:   loc("file.c":1:2)
        // CASE-17: >> end of diagnostic (userData: 42)
        // CASE-17: processing diagnostic (userData: 42) <<
        // CASE-17:   test diagnostics
        // CASE-17:   loc(callsite("other-file.c":2:3 at "file.c":1:2))
        // CASE-17: >> end of diagnostic (userData: 42)
        // CASE-17: processing diagnostic (userData: 42) <<
        // CASE-17:   test diagnostics
        // CASE-17:   loc("named")
        // CASE-17: >> end of diagnostic (userData: 42)
        // CASE-17: processing diagnostic (userData: 42) <<
        // CASE-17:   test diagnostics
        // CASE-17:   loc(fused["named", callsite("other-file.c":2:3 at "file.c":1:2)])
        // CASE-17: deleting user data (userData: 42)
        // CHECK-NOT: processing diagnostic
        // CASE-17:     more test diagnostics

        const term = try fc_runner.cleanup();
        try expect(term != null and term.?.Exited == 0);
    }
}
