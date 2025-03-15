const std = @import("std");
const bc = @import("../../build_common.zig");

const MLIR_LIBS = [_][]const u8{
    "IR",
    "Parser",
    "Support",
    "Dialect",
    "Pass",
    "Parser",
};

const MLIR_CAPI_LIBS = [_][]const u8{
    "IR",
    "RegisterEverything",
};

// libLLVMSupport is required:
// https://github.com/llvm/llvm-project/blob/release/17.x/mlir/cmake/modules/AddMLIR.cmake#L342-L343
const LLVM_LIBS = [_][]const u8{
    "Support",
};

// NOTE: The standalone example is built in static library, only the test is
// built in shared library. So we try to follow the convention to see whether
// it work.
pub fn build(
    b: *std.Build,
    config: bc.BuildConfig,
    lib_dirs: bc.LibDirs,
    misc: bc.MiscConfig,
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = "sample",
        .root_source_file = b.path("src/sample/main.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .pic = config.pic,
    });

    if (misc.build_dialect) {
        // Build dialect before building this program
        const cmd_build = b.addSystemCommand(&.{"./build_dialect.sh"});
        cmd_build.setCwd(b.path("src/sample"));
        cmd_build.has_side_effects = true;
        cmd_build.setEnvironmentVariable(
            "MLIR_DIR",
            std.fs.path.join(b.allocator, &.{ lib_dirs.mlir_lib, "cmake", "mlir" }) catch "",
        );
        exe.step.dependOn(&cmd_build.step);
    }

    // XXX: Currently dialect cannot be built as shared library, so this isn't needed.
    // exe.root_module.addRPathSpecial("$ORIGIN/../lib");

    exe.addIncludePath(.{ .cwd_relative = "src/sample" });
    exe.addIncludePath(.{ .cwd_relative = lib_dirs.mlir_inc });

    exe.addLibraryPath(.{ .cwd_relative = "src/sample/inst_sample/lib" });
    exe.addLibraryPath(.{ .cwd_relative = lib_dirs.mlir_lib });

    exe.linkSystemLibrary("MLIRSample");
    exe.linkSystemLibrary("SampleCAPI");
    exe.linkLibC();
    exe.linkLibCpp();

    bc.linkLibs(exe, "MLIR", &MLIR_LIBS);
    bc.linkLibs(exe, "MLIRCAPI", &MLIR_CAPI_LIBS);
    bc.linkLibs(exe, "LLVM", &LLVM_LIBS);
    return exe;
}
