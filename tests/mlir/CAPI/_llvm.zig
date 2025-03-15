/// XXX: We cannot name this module as "llvm.zig" which would result in the
/// following compilation error:
/// ```raw
/// error: llvm intrinsics cannot be defined!
/// ptr @llvm.test.testTypeCreation
/// LLVM ERROR: Broken module found, compilation aborted!
/// ```
const std = @import("std");
const c = @import("c.zig");

// NOTE: these are the headers required for tests in this file, but it
// aggregate to "c.zig" to avoid multiple invocations of `@cImport`.
// const mlir = @cImport({
//     @cInclude("mlir-c/IR.h");
//     @cInclude("mlir-c/BuiltinTypes.h");
//     @cInclude("mlir-c/Dialect/LLVM.h");
// });
const mlir = c.mlir;

const expect = std.testing.expect;
const strref = mlir.mlirStringRefCreateFromCString;

test "testTypeCreation" {
    const ctx = mlir.mlirContextCreate();
    defer mlir.mlirContextDestroy(ctx);

    const llvm_handle = mlir.mlirGetDialectHandle__llvm__();
    mlir.mlirDialectHandleRegisterDialect(llvm_handle, ctx);
    _ = mlir.mlirContextGetOrLoadDialect(ctx, strref("llvm"));

    const i8_t = mlir.mlirIntegerTypeGet(ctx, 8);
    const i32_t = mlir.mlirIntegerTypeGet(ctx, 32);
    const i64_t = mlir.mlirIntegerTypeGet(ctx, 64);

    const i32p = mlir.mlirLLVMPointerTypeGet(i32_t, 0);
    const i32p_ref = mlir.mlirTypeParseGet(ctx, strref("!llvm.ptr<i32>"));
    try expect(mlir.mlirTypeEqual(i32p, i32p_ref));

    const i32p4 = mlir.mlirLLVMPointerTypeGet(i32_t, 4);
    const i32p4_ref = mlir.mlirTypeParseGet(ctx, strref("!llvm.ptr<i32, 4>"));
    try expect(mlir.mlirTypeEqual(i32p4, i32p4_ref));

    const void_t = mlir.mlirLLVMVoidTypeGet(ctx);
    const void_t_ref = mlir.mlirTypeParseGet(ctx, strref("!llvm.void"));
    try expect(mlir.mlirTypeEqual(void_t, void_t_ref));

    const i32_4 = mlir.mlirLLVMArrayTypeGet(i32_t, 4);
    const i32_4_ref = mlir.mlirTypeParseGet(ctx, strref("!llvm.array<4 x i32>"));
    try expect(mlir.mlirTypeEqual(i32_4, i32_4_ref));

    const i32_i64_arr = [_]mlir.MlirType{ i32_t, i64_t };
    const i8_i32_i64 = mlir.mlirLLVMFunctionTypeGet(i8_t, 2, &i32_i64_arr, false);
    const i8_i32_i64_ref = mlir.mlirTypeParseGet(ctx, strref("!llvm.func<i8 (i32, i64)>"));
    try expect(mlir.mlirTypeEqual(i8_i32_i64, i8_i32_i64_ref));

    const i32_i64_s = mlir.mlirLLVMStructTypeLiteralGet(ctx, 2, &i32_i64_arr, false);
    const i32_i64_s_ref = mlir.mlirTypeParseGet(ctx, strref("!llvm.struct<(i32, i64)>"));
    try expect(mlir.mlirTypeEqual(i32_i64_s, i32_i64_s_ref));
}
