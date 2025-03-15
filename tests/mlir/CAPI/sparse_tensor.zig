const std = @import("std");
const c = @import("c.zig");
const helper = @import("helper.zig");

// NOTE: these are the headers required for tests in this file, but it
// aggregate to "c.zig" to avoid multiple invocations of `@cImport`.
// const mlir = @cImport({
//     @cInclude("mlir-c/IR.h");
//     @cInclude("mlir-c/RegisterEverything.h");
//     @cInclude("mlir-c/Dialect/SparseTensor.h");
// });
const mlir = c.mlir;

const test_allocator = std.testing.allocator;
const expect = std.testing.expect;
const strref = mlir.mlirStringRefCreateFromCString;

fn createAndInitContext() !mlir.MlirContext {
    const ctx = mlir.mlirContextCreate();
    const handle = mlir.mlirGetDialectHandle__sparse_tensor__();
    mlir.mlirDialectHandleRegisterDialect(handle, ctx);

    const ns = mlir.mlirDialectHandleGetNamespace(handle);
    const dialect = mlir.mlirContextGetOrLoadDialect(ctx, ns);
    try expect(!mlir.mlirDialectIsNull(dialect));
    return ctx;
}

test "testRoundtripEncoding" {
    var session = try helper.StderrToBufferPrintSession.init(test_allocator, 4096);
    defer session.deinit();

    const ctx = try createAndInitContext();
    defer mlir.mlirContextDestroy(ctx);

    const ori_asm =
        \\#sparse_tensor.encoding<{
        \\lvlTypes = [ "dense", "compressed", "compressed"],
        \\dimToLvl = affine_map<(d0, d1)[s0] -> (s0, d0, d1)>,
        \\posWidth = 32, crdWidth = 64 }>
    ;

    const ori_attr = mlir.mlirAttributeParseGet(ctx, strref(ori_asm));
    try expect(mlir.mlirAttributeIsASparseTensorEncodingAttr(ori_attr));

    const dim_to_lvl = mlir.mlirSparseTensorEncodingAttrGetDimToLvl(ori_attr);
    try session.runOnce(mlir.mlirAffineMapDump, .{dim_to_lvl});
    try expect(session.contentEql("(d0, d1)[s0] -> (s0, d0, d1)\n"));

    const lvl_rank = mlir.mlirSparseTensorEncodingGetLvlRank(ori_attr);
    var lvl_types = try test_allocator.alloc(mlir.MlirSparseTensorDimLevelType, @intCast(lvl_rank));
    defer test_allocator.free(lvl_types);

    for (0..@intCast(lvl_rank)) |i| {
        lvl_types[i] = mlir.mlirSparseTensorEncodingAttrGetLvlType(ori_attr, @intCast(i));
    }
    try expect(lvl_rank == 3);
    try expect(lvl_types[0] == 4); // MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE
    try expect(lvl_types[1] == 8); // MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED
    try expect(lvl_types[2] == 8);

    const pos_width = mlir.mlirSparseTensorEncodingAttrGetPosWidth(ori_attr);
    try expect(pos_width == 32);

    const crd_width = mlir.mlirSparseTensorEncodingAttrGetCrdWidth(ori_attr);
    try expect(crd_width == 64);

    const new_attr = mlir.mlirSparseTensorEncodingAttrGet(
        ctx,
        lvl_rank,
        @ptrCast(lvl_types),
        dim_to_lvl,
        pos_width,
        crd_width,
    );
    try expect(mlir.mlirAttributeEqual(ori_attr, new_attr));
}
