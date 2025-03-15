const std = @import("std");
const bc = @import("../../build_common.zig");

// https://github.com/llvm/llvm-project/blob/release/17.x/mlir/examples/toy/Ch2/CMakeLists.txt#L20-L26
const MLIR_LIBS = [_][]const u8{
    "Analysis",
    "IR",
    "Parser",
    "SideEffectInterfaces",
    "Transforms",
    // XXX: Without this, we will get linker errors related to `TypeIDResolver`,
    // e.g., "_ZN4mlir6detail14TypeIDResolverINS_3toy10ToyDialectEvE2idE" (
    // `mlir::detail::TypeIDResolver<mlir::toy::ToyDialect, void>::id`).
    "Support",
};

const MLIR_CAPI_LIBS = [_][]const u8{
    "IR",
};

const LLVM_LIBS = [_][]const u8{
    "Support",
};

pub fn build(
    b: *std.Build,
    config: bc.BuildConfig,
    lib_dirs: bc.LibDirs,
    misc: bc.MiscConfig,
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = "toyc-ch2",
        .root_source_file = b.path("src/Ch2/toyc.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .pic = config.pic,
    });

    if (misc.build_dialect) {
        // Build dialect before building this program
        const cmd_build = b.addSystemCommand(&.{"./build_dialect.sh"});
        cmd_build.setCwd(b.path("src/Ch2"));
        cmd_build.has_side_effects = true;
        cmd_build.setEnvironmentVariable(
            "MLIR_DIR",
            std.fs.path.join(b.allocator, &.{ lib_dirs.mlir_lib, "cmake", "mlir" }) catch "",
        );

        exe.step.dependOn(&cmd_build.step);
    }

    exe.addIncludePath(.{ .cwd_relative = "src/Ch2" });
    exe.addIncludePath(.{ .cwd_relative = lib_dirs.mlir_inc });

    exe.addLibraryPath(.{ .cwd_relative = "src/Ch2/inst_toy/lib" });
    exe.addLibraryPath(.{ .cwd_relative = lib_dirs.mlir_lib });

    exe.linkSystemLibrary("MLIRToy");
    exe.linkSystemLibrary("ToyCAPI");
    exe.linkLibC();
    exe.linkLibCpp();

    bc.linkLibs(exe, "MLIR", &MLIR_LIBS);
    bc.linkLibs(exe, "MLIRCAPI", &MLIR_CAPI_LIBS);
    bc.linkLibs(exe, "LLVM", &LLVM_LIBS);
    return exe;
}
