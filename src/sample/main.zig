const std = @import("std");
const c = @cImport({
    @cInclude("sample-c/Sample.h");
});

pub fn main() void {
    const allocator = std.heap.page_allocator;
    _ = allocator;

    const ctx = c.mlirContextCreate();
    defer c.mlirContextDestroy(ctx);

    std.debug.print("Currently loaded dialects: {d}\n", .{c.mlirContextGetNumLoadedDialects(ctx)});

    const handle = c.mlirGetDialectHandle__sample__();
    var ns = c.mlirDialectHandleGetNamespace(handle);
    var sample_dialect = c.mlirContextGetOrLoadDialect(ctx, ns);
    std.debug.print("dialect is loaded? {any}\n", .{!c.mlirDialectIsNull(sample_dialect)});

    // Load dialect
    c.mlirDialectHandleRegisterDialect(handle, ctx);
    ns = c.mlirDialectHandleGetNamespace(handle);
    sample_dialect = c.mlirContextGetOrLoadDialect(ctx, ns);

    std.debug.print("dialect is loaded? {any}\n", .{!c.mlirDialectIsNull(sample_dialect)});
}
