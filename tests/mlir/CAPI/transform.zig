const std = @import("std");
const c = @import("c.zig");
const helper = @import("helper.zig");

// NOTE: these are the headers required for tests in this file, but it
// aggregate to "c.zig" to avoid multiple invocations of `@cImport`.
// const mlir = @cImport({
//     @cInclude("mlir-c/IR.h");
//     @cInclude("mlir-c/Support.h");
//     @cInclude("mlir-c/Dialect/Transform.h");
// });
const mlir = c.mlir;

const test_allocator = std.testing.allocator;
const expect = std.testing.expect;
const strref = mlir.mlirStringRefCreateFromCString;

fn createAndInitContext() !mlir.MlirContext {
    const ctx = mlir.mlirContextCreate();
    const handle = mlir.mlirGetDialectHandle__transform__();
    mlir.mlirDialectHandleRegisterDialect(handle, ctx);

    const ns = mlir.mlirDialectHandleGetNamespace(handle);
    const dialect = mlir.mlirContextGetOrLoadDialect(ctx, ns);
    try expect(!mlir.mlirDialectIsNull(dialect));
    return ctx;
}

test "testAnyOpType" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const parsed_t = mlir.mlirTypeParseGet(ctx, strref("!transform.any_op"));
    const constructed_t = mlir.mlirTransformAnyOpTypeGet(ctx);

    try expect(!mlir.mlirTypeIsNull(parsed_t));
    try expect(!mlir.mlirTypeIsNull(constructed_t));

    try expect(mlir.mlirTypeEqual(parsed_t, constructed_t));

    try expect(mlir.mlirTypeIsATransformAnyOpType(parsed_t));
    try expect(!mlir.mlirTypeIsATransformOperationType(parsed_t));

    try session.runOnce(mlir.mlirTypeDump, .{constructed_t});
    try expect(session.contentEql("!transform.any_op\n"));
}

test "testOperationType" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const parsed_t = mlir.mlirTypeParseGet(ctx, strref("!transform.op<\"foo.bar\">"));
    const constructed_t = mlir.mlirTransformOperationTypeGet(ctx, strref("foo.bar"));

    try expect(!mlir.mlirTypeIsNull(parsed_t));
    try expect(!mlir.mlirTypeIsNull(constructed_t));

    try expect(mlir.mlirTypeEqual(parsed_t, constructed_t));

    try expect(!mlir.mlirTypeIsATransformAnyOpType(parsed_t));
    try expect(mlir.mlirTypeIsATransformOperationType(parsed_t));

    const op_name = mlir.mlirTransformOperationTypeGetOperationName(constructed_t);
    try expect(mlir.mlirStringRefEqual(op_name, strref("foo.bar")));

    try session.runOnce(mlir.mlirTypeDump, .{constructed_t});
    try expect(session.contentEql("!transform.op<\"foo.bar\">\n"));
}
