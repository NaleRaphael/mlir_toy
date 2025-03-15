const std = @import("std");
const c = @import("c.zig");
const helper = @import("helper.zig");

// NOTE: these are the headers required for tests in this file, but it
// aggregate to "c.zig" to avoid multiple invocations of `@cImport`.
// const mlir = @cImport({
//     @cInclude("mlir-c/IR.h");
//     @cInclude("mlir-c/BuiltinTypes.h");
//     @cInclude("mlir-c/Dialect/Quant.h");
// });
const mlir = c.mlir;

const test_allocator = std.testing.allocator;
const expect = std.testing.expect;
const strref = mlir.mlirStringRefCreateFromCString;
const typeDump = mlir.mlirTypeDump;

fn createAndInitContext() !mlir.MlirContext {
    const ctx = mlir.mlirContextCreate();
    const handle = mlir.mlirGetDialectHandle__quant__();
    mlir.mlirDialectHandleRegisterDialect(handle, ctx);

    const ns = mlir.mlirDialectHandleGetNamespace(handle);
    const dialect = mlir.mlirContextGetOrLoadDialect(ctx, ns);
    try expect(!mlir.mlirDialectIsNull(dialect));
    return ctx;
}

fn isClose(comptime T: type, a: T, b: T, atol: T, rtol: T) bool {
    comptime {
        if (@typeInfo(T) != .Float) {
            @compileError("Expect input type is float");
        }
    }
    std.debug.assert(atol > 0);
    std.debug.assert(rtol > 0);

    const delta = a - b;
    const delta_abs = if (delta > 0) delta else -delta;
    const b_abs = if (b > 0) b else -b;
    const result = delta_abs <= atol + rtol * b_abs;
    return result;
}

fn expectIsClose(comptime T: type, a: T, b: T) !void {
    try expect(isClose(T, a, b, 1e-5, 1e-8));
}

test "testTypeHierarchy" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const i8_t = mlir.mlirIntegerTypeGet(ctx, 8);
    const any_t = mlir.mlirTypeParseGet(ctx, strref("!quant.any<i8<-8:7>:f32>"));
    const uniform_t = mlir.mlirTypeParseGet(
        ctx,
        strref("!quant.uniform<i8<-8:7>:f32, 0.99872:127>"),
    );
    const per_axis_t = mlir.mlirTypeParseGet(
        ctx,
        strref("!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>"),
    );
    const calibrated_t = mlir.mlirTypeParseGet(
        ctx,
        strref("!quant.calibrated<f32<-0.998:1.2321>>"),
    );

    try expect(!mlir.mlirTypeIsNull(any_t));
    try expect(!mlir.mlirTypeIsNull(uniform_t));
    try expect(!mlir.mlirTypeIsNull(per_axis_t));
    try expect(!mlir.mlirTypeIsNull(calibrated_t));

    try expect(!mlir.mlirTypeIsAQuantizedType(i8_t));
    try expect(mlir.mlirTypeIsAQuantizedType(any_t));
    try expect(mlir.mlirTypeIsAQuantizedType(uniform_t));
    try expect(mlir.mlirTypeIsAQuantizedType(per_axis_t));
    try expect(mlir.mlirTypeIsAQuantizedType(calibrated_t));

    try expect(mlir.mlirTypeIsAAnyQuantizedType(any_t));
    try expect(mlir.mlirTypeIsAUniformQuantizedType(uniform_t));
    try expect(mlir.mlirTypeIsAUniformQuantizedPerAxisType(per_axis_t));
    try expect(mlir.mlirTypeIsACalibratedQuantizedType(calibrated_t));

    try expect(!mlir.mlirTypeIsAUniformQuantizedType(per_axis_t));
    try expect(!mlir.mlirTypeIsACalibratedQuantizedType(uniform_t));
}

test "testAnyQuantziedType" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const any_parsed_t = mlir.mlirTypeParseGet(
        ctx,
        strref("!quant.any<i8<-8:7>:f32>"),
    );

    const i8_t = mlir.mlirIntegerTypeGet(ctx, 8);
    const f32_t = mlir.mlirF32TypeGet(ctx);
    const qflag: c_uint = mlir.mlirQuantizedTypeGetSignedFlag();
    const any_t = mlir.mlirAnyQuantizedTypeGet(qflag, i8_t, f32_t, -8, 7);

    try expect(mlir.mlirQuantizedTypeGetFlags(any_t) == 1);
    try expect(mlir.mlirQuantizedTypeIsSigned(any_t));

    try session.runOnce(typeDump, .{mlir.mlirQuantizedTypeGetStorageType(any_t)});
    try expect(session.contentEql("i8\n"));
    try session.runOnce(typeDump, .{mlir.mlirQuantizedTypeGetExpressedType(any_t)});
    try expect(session.contentEql("f32\n"));

    try expect(mlir.mlirQuantizedTypeGetStorageTypeMin(any_t) == -8);
    try expect(mlir.mlirQuantizedTypeGetStorageTypeMax(any_t) == 7);
    try expect(mlir.mlirQuantizedTypeGetStorageTypeIntegralWidth(any_t) == 8);

    try session.runOnce(typeDump, .{mlir.mlirQuantizedTypeGetQuantizedElementType(any_t)});
    try expect(session.contentEql("!quant.any<i8<-8:7>:f32>\n"));

    try expect(mlir.mlirTypeEqual(any_parsed_t, any_t));
    try session.runOnce(typeDump, .{any_t});
    try expect(session.contentEql("!quant.any<i8<-8:7>:f32>\n"));
}

test "testUniformType" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const uniform_parsed_t = mlir.mlirTypeParseGet(
        ctx,
        strref("!quant.uniform<i8<-8:7>:f32, 0.99872:127>"),
    );

    const i8_t = mlir.mlirIntegerTypeGet(ctx, 8);
    const f32_t = mlir.mlirF32TypeGet(ctx);
    const qflag: c_uint = mlir.mlirQuantizedTypeGetSignedFlag();
    const uniform_t = mlir.mlirUniformQuantizedTypeGet(qflag, i8_t, f32_t, 0.99872, 127, -8, 7);

    try expectIsClose(f64, mlir.mlirUniformQuantizedTypeGetScale(uniform_t), 0.99872);
    try expect(mlir.mlirUniformQuantizedTypeGetZeroPoint(uniform_t) == 127);
    try expect(!mlir.mlirUniformQuantizedTypeIsFixedPoint(uniform_t));
    try expect(mlir.mlirTypeEqual(uniform_t, uniform_parsed_t));

    try session.runOnce(typeDump, .{uniform_t});
    try expect(session.contentEql("!quant.uniform<i8<-8:7>:f32, 9.987200e-01:127>\n"));
}

test "testUniformPerAxisType" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const per_axis_parsed_t = mlir.mlirTypeParseGet(
        ctx,
        strref("!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>"),
    );

    const i8_t = mlir.mlirIntegerTypeGet(ctx, 8);
    const f32_t = mlir.mlirF32TypeGet(ctx);
    var scales = [_]f64{ 200.0, 0.99872 };
    var zero_points = [_]i64{ 0, 120 };
    const qflag: c_uint = mlir.mlirQuantizedTypeGetSignedFlag();
    const per_axis_t = mlir.mlirUniformQuantizedPerAxisTypeGet(
        qflag,
        i8_t,
        f32_t,
        2,
        &scales,
        &zero_points,
        1,
        mlir.mlirQuantizedTypeGetDefaultMinimumForInteger(true, 8),
        mlir.mlirQuantizedTypeGetDefaultMaximumForInteger(true, 8),
    );

    try expect(mlir.mlirUniformQuantizedPerAxisTypeGetNumDims(per_axis_t) == 2);
    try expectIsClose(f64, mlir.mlirUniformQuantizedPerAxisTypeGetScale(per_axis_t, 0), 200.0);
    try expectIsClose(f64, mlir.mlirUniformQuantizedPerAxisTypeGetScale(per_axis_t, 1), 0.99872);
    try expect(mlir.mlirUniformQuantizedPerAxisTypeGetZeroPoint(per_axis_t, 0) == 0);
    try expect(mlir.mlirUniformQuantizedPerAxisTypeGetZeroPoint(per_axis_t, 1) == 120);
    try expect(mlir.mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(per_axis_t) == 1);
    try expect(!mlir.mlirUniformQuantizedPerAxisTypeIsFixedPoint(per_axis_t));
    try expect(mlir.mlirTypeEqual(per_axis_t, per_axis_parsed_t));

    try session.runOnce(typeDump, .{per_axis_t});
    try expect(session.contentEql("!quant.uniform<i8:f32:1, {2.000000e+02,9.987200e-01:120}>\n"));
}

test "testCalibratedType" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const calibrated_parsed_t = mlir.mlirTypeParseGet(
        ctx,
        strref("!quant.calibrated<f32<-0.998:1.2321>>"),
    );

    const f32_t = mlir.mlirF32TypeGet(ctx);
    const calibrated_t = mlir.mlirCalibratedQuantizedTypeGet(f32_t, -0.998, 1.2321);

    try expectIsClose(f64, mlir.mlirCalibratedQuantizedTypeGetMin(calibrated_t), -0.998);
    try expectIsClose(f64, mlir.mlirCalibratedQuantizedTypeGetMax(calibrated_t), 1.2321);
    try expect(mlir.mlirTypeEqual(calibrated_t, calibrated_parsed_t));

    try session.runOnce(typeDump, .{calibrated_t});
    try expect(session.contentEql("!quant.calibrated<f32<-0.998:1.232100e+00>>\n"));
}
