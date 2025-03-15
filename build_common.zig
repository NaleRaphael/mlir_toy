//! Common structs / functions used in `build.zig` of this repository.
const std = @import("std");

/// Some common configs for executable and static/shared libs.
pub const BuildConfig = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    pic: ?bool,
};

pub const MiscConfig = struct {
    build_dialect: bool,
    bin_llvm_lit: ?[]const u8,
};

/// Patterns:
/// - XXX: Library name
/// - XXX_inc: Path of "include" directory of XXX library
/// - XXX_lib: Path of "lib" (shared library) directory of XXX library
pub const LibDirs = struct {
    mlir_inc: []const u8,
    mlir_lib: []const u8,
    llvm_inc: []const u8,
    llvm_lib: []const u8,
};

pub fn linkLibs(
    target: *std.Build.Step.Compile,
    comptime prefix: []const u8,
    comptime lib_list: []const []const u8,
) void {
    inline for (lib_list) |lib| {
        target.linkSystemLibrary(prefix ++ lib);
    }
}
