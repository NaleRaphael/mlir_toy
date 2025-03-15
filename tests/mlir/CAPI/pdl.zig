const std = @import("std");
const c = @import("c.zig");
const helper = @import("helper.zig");

// NOTE: these are the headers required for tests in this file, but it
// aggregate to "c.zig" to avoid multiple invocations of `@cImport`.
// const mlir = @cImport({
//     @cInclude("mlir-c/IR.h");
//     @cInclude("mlir-c/BuiltinTypes.h");
//     @cInclude("mlir-c/Dialect/PDL.h");
// });
const mlir = c.mlir;

const test_allocator = std.testing.allocator;
const expect = std.testing.expect;
const strref = mlir.mlirStringRefCreateFromCString;

fn createAndInitContext() !mlir.MlirContext {
    const ctx = mlir.mlirContextCreate();
    const handle = mlir.mlirGetDialectHandle__pdl__();
    mlir.mlirDialectHandleRegisterDialect(handle, ctx);

    // XXX: Here is a different behavior comparing with the original
    // implementation [1].
    //
    // We have to load the PDL dialect beforehand, otherwise we would get
    // the following error while trying to call `mlirPDL<TYPE>TypeGet()` before
    // `mlirTypeParseGet()`:
    // ```
    // // Assume that we called `mlirPDLTypeTypeGet()`
    // LLVM ERROR: can't create type 'mlir::pdl::TypeType' because storage
    // uniquer isn't initialized: the dialect was likely not loaded, or the type
    // wasn't added with addTypes<...>() in the Dialect::initialize() method.
    // ```
    //
    // That is:
    // ```zig
    // // This works
    // const parsed_t = mlir.mlirTypeParseGet(ctx, strref("!pdl.attribute"));
    // const constructed_t = mlir.mlirPDLAttributeTypeGet(ctx);
    //
    // // This won't work, unless the PDL dialect is loaded beforehand
    // const constructed_t = mlir.mlirPDLAttributeTypeGet(ctx);
    // const parsed_t = mlir.mlirTypeParseGet(ctx, strref("!pdl.attribute"));
    // ```
    //
    // [1]: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/test/CAPI/pdl.c#L329-L330
    const ns = mlir.mlirDialectHandleGetNamespace(handle);
    const dialect = mlir.mlirContextGetOrLoadDialect(ctx, ns);
    try expect(!mlir.mlirDialectIsNull(dialect));
    return ctx;
}

fn isPDLType(
    input: mlir.MlirType,
    is_pdl: bool,
    is_attribute: bool,
    is_operation: bool,
    is_range: bool,
    is_type: bool,
    is_value: bool,
) !void {
    try expect(mlir.mlirTypeIsAPDLType(input) == is_pdl);
    try expect(mlir.mlirTypeIsAPDLAttributeType(input) == is_attribute);
    try expect(mlir.mlirTypeIsAPDLOperationType(input) == is_operation);
    try expect(mlir.mlirTypeIsAPDLRangeType(input) == is_range);
    try expect(mlir.mlirTypeIsAPDLTypeType(input) == is_type);
    try expect(mlir.mlirTypeIsAPDLValueType(input) == is_value);
}

test "testAttributeType" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const parsed_t = mlir.mlirTypeParseGet(ctx, strref("!pdl.attribute"));
    const constructed_t = mlir.mlirPDLAttributeTypeGet(ctx);

    try expect(!mlir.mlirTypeIsNull(parsed_t));
    try expect(!mlir.mlirTypeIsNull(constructed_t));

    try isPDLType(parsed_t, true, true, false, false, false, false);
    try isPDLType(constructed_t, true, true, false, false, false, false);

    try expect(mlir.mlirTypeEqual(parsed_t, constructed_t));

    try session.runOnce(mlir.mlirTypeDump, .{parsed_t});
    try expect(session.contentEql("!pdl.attribute\n"));

    try session.runOnce(mlir.mlirTypeDump, .{constructed_t});
    try expect(session.contentEql("!pdl.attribute\n"));
}

test "testOperationType" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const parsed_t = mlir.mlirTypeParseGet(ctx, strref("!pdl.operation"));
    const constructed_t = mlir.mlirPDLOperationTypeGet(ctx);

    try expect(!mlir.mlirTypeIsNull(parsed_t));
    try expect(!mlir.mlirTypeIsNull(constructed_t));

    try isPDLType(parsed_t, true, false, true, false, false, false);
    try isPDLType(constructed_t, true, false, true, false, false, false);

    try expect(mlir.mlirTypeEqual(parsed_t, constructed_t));

    try session.runOnce(mlir.mlirTypeDump, .{parsed_t});
    try expect(session.contentEql("!pdl.operation\n"));

    try session.runOnce(mlir.mlirTypeDump, .{constructed_t});
    try expect(session.contentEql("!pdl.operation\n"));
}

test "testRangeType" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const type_t = mlir.mlirPDLTypeTypeGet(ctx);
    const parsed_t = mlir.mlirTypeParseGet(ctx, strref("!pdl.range<type>"));
    const constructed_t = mlir.mlirPDLRangeTypeGet(type_t);
    const element_t = mlir.mlirPDLRangeTypeGetElementType(constructed_t);

    try expect(!mlir.mlirTypeIsNull(type_t));
    try expect(!mlir.mlirTypeIsNull(parsed_t));
    try expect(!mlir.mlirTypeIsNull(constructed_t));

    try isPDLType(parsed_t, true, false, false, true, false, false);
    try isPDLType(constructed_t, true, false, false, true, false, false);

    try expect(mlir.mlirTypeEqual(parsed_t, constructed_t));
    try expect(mlir.mlirTypeEqual(type_t, element_t));

    try session.runOnce(mlir.mlirTypeDump, .{parsed_t});
    try expect(session.contentEql("!pdl.range<type>\n"));

    try session.runOnce(mlir.mlirTypeDump, .{constructed_t});
    try expect(session.contentEql("!pdl.range<type>\n"));

    try session.runOnce(mlir.mlirTypeDump, .{element_t});
    try expect(session.contentEql("!pdl.type\n"));
}

test "testTypeType" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const parsed_t = mlir.mlirTypeParseGet(ctx, strref("!pdl.type"));
    const constructed_t = mlir.mlirPDLTypeTypeGet(ctx);

    try expect(!mlir.mlirTypeIsNull(parsed_t));
    try expect(!mlir.mlirTypeIsNull(constructed_t));

    try isPDLType(parsed_t, true, false, false, false, true, false);
    try isPDLType(constructed_t, true, false, false, false, true, false);

    try expect(mlir.mlirTypeEqual(parsed_t, constructed_t));

    try session.runOnce(mlir.mlirTypeDump, .{parsed_t});
    try expect(session.contentEql("!pdl.type\n"));

    try session.runOnce(mlir.mlirTypeDump, .{constructed_t});
    try expect(session.contentEql("!pdl.type\n"));
}

test "testValueType" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const parsed_t = mlir.mlirTypeParseGet(ctx, strref("!pdl.value"));
    const constructed_t = mlir.mlirPDLValueTypeGet(ctx);

    try expect(!mlir.mlirTypeIsNull(parsed_t));
    try expect(!mlir.mlirTypeIsNull(constructed_t));

    try isPDLType(parsed_t, true, false, false, false, false, true);
    try isPDLType(constructed_t, true, false, false, false, false, true);

    try expect(mlir.mlirTypeEqual(parsed_t, constructed_t));

    try session.runOnce(mlir.mlirTypeDump, .{parsed_t});
    try expect(session.contentEql("!pdl.value\n"));

    try session.runOnce(mlir.mlirTypeDump, .{constructed_t});
    try expect(session.contentEql("!pdl.value\n"));
}
