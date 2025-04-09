//! Common structs / functions used in `build.zig` of this repository.
const std = @import("std");

/// Some common configs for executable and static/shared libs.
pub const BuildConfig = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    link_mode: ?std.builtin.LinkMode,
    pic: ?bool,
};

/// Other configs
/// - build_dialect: Run the "build_dialect.sh" script to build dialect library.
///   Set this to false if you want to build it manually without updating it
///   everytime when "build.zig" is invoked.
/// - use_custom_libcxx: Link to the libcxx supplied under `$LLVM_DIR` instead
///   of the one built by Zig.
pub const MiscConfig = struct {
    build_dialect: bool,
    use_custom_libcxx: bool,
};

/// Patterns:
/// - XXX: Library name
/// - XXX_dir: Root path of the library
/// - XXX_inc: Path of "include" directory of XXX library
/// - XXX_lib: Path of "lib" (shared library) directory of XXX library
pub const LibDirs = struct {
    mlir_dir: []const u8,
    mlir_inc: []const u8,
    mlir_lib: []const u8,
    llvm_dir: []const u8,
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
