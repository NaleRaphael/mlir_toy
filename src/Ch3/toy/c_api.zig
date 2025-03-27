/// This file use `@cImport()` to include necessary MLIR headers, our own
/// C API, and some helper functions.
///
/// To make LSP work with C API defined in "toy/c/toy.h", we can actually write
/// Zig functions for them in this file like how `zml` does [1]. But it seems
/// to me not an important work for now since I want to focus on learning and
/// exploring MLIR.
///
/// [1]: https://github.com/zml/zml/blob/master/mlir/mlir.zig
pub const c = @cImport({
    @cInclude("toy/c/toy.h");
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/BuiltinTypes.h");
    @cInclude("mlir-c/BuiltinAttributes.h");
    @cInclude("mlir-c/Diagnostics.h");
    @cInclude("mlir-c/Pass.h");
    @cInclude("mlir-c/Transforms.h");
});

// XXX: It's generally not recommended to use another `@cImport()` here, but I
// just want to make a namespace to avoid confusion.
pub const stdio = @cImport(@cInclude("stdio.h"));

pub const CAPIError = error{
    FailedToLoadDialect,
};

pub fn loadToyDialect(ctx: c.MlirContext) !void {
    const handle = c.mlirGetDialectHandle__toy__();
    const ns = c.mlirDialectHandleGetNamespace(handle);

    c.mlirDialectHandleRegisterDialect(handle, ctx);
    const dialect = c.mlirContextGetOrLoadDialect(ctx, ns);

    if (c.mlirDialectIsNull(dialect)) {
        return CAPIError.FailedToLoadDialect;
    }
}

pub fn printToStderr(str: c.MlirStringRef, user_data: ?*anyopaque) callconv(.C) void {
    _ = user_data;
    _ = stdio.fwrite(str.data, 1, str.length, stdio.stderr);
}
