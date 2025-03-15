const std = @import("std");

// MLIR CAPI lib names (libMLIRCAPI<name>.so)
const MLIR_CAPI_LIBS = [_][]const u8{
    "Arith",
    "Async",
    "ControlFlow",
    "Conversion",
    "Debug",
    "ExecutionEngine",
    "Func",
    "GPU",
    "Interfaces",
    "IR",
    "Linalg",
    "LLVM",
    "Math",
    "MemRef",
    "MLProgram",
    "PDL",
    "Quant",
    "RegisterEverything",
    "SCF",
    "Shape",
    "SparseTensor",
    "Tensor",
    "TransformDialect",
    "Transforms",
    "Vector",
};

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const build_opts = .{
        .llvm_dir = b.option(
            []const u8,
            "llvm_dir",
            "Root directory of LLVM",
        ) orelse "/usr",
        .mlir_dir = b.option(
            []const u8,
            "mlir_dir",
            "Root directory of MLIR",
        ) orelse "/usr",
    };
    const build_options = b.addOptions();
    inline for (std.meta.fields(@TypeOf(build_opts))) |field| {
        build_options.addOption(field.type, field.name, @field(build_opts, field.name));
    }

    const test_opts = .{
        .src_dir = b.option([]const u8, "src_dir",
            \\Source directory of this project, this will be specified automatically.
            \\This value is used as a workaround to get path of source file to
            \\run with FileCheck.
        ) orelse b.path(".").getPath(b),
        .link_openmp = b.option(bool, "link_openmp",
            \\Link to OpenMP. This option would enable further test in some test
            \\cases, e.g., "testOmpCreation" in "tests/execution_engine.zig".
        ) orelse false,
    };
    const test_options = b.addOptions();
    inline for (std.meta.fields(@TypeOf(test_opts))) |field| {
        test_options.addOption(field.type, field.name, @field(test_opts, field.name));
    }

    const mlir_include_dir = try std.fs.path.join(b.allocator, &.{ build_opts.mlir_dir, "include" });
    const mlir_library_dir = try std.fs.path.join(b.allocator, &.{ build_opts.mlir_dir, "lib" });

    const test_step = b.step("test", "Run unit test");
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("CAPI/tests.zig"),
        .target = target,
        .optimize = optimize,
        // NOTE: Equivalent to `zig test --test-filter=...`, but need to be
        // passed as `zig build test -Dtest-filter=...`.
        .filter = b.option([]const u8, "test-filter", "Filter strings for test"),
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    // XXX: currently we don't want tests being cached
    run_unit_tests.has_side_effects = true;
    test_step.dependOn(&run_unit_tests.step);

    unit_tests.addIncludePath(.{ .cwd_relative = mlir_include_dir });
    unit_tests.addLibraryPath(.{ .cwd_relative = mlir_library_dir });
    inline for (MLIR_CAPI_LIBS) |lib| {
        unit_tests.linkSystemLibrary("MLIRCAPI" ++ lib);
    }
    unit_tests.linkLibC();

    if (test_opts.link_openmp) {
        unit_tests.root_module.linkSystemLibrary("omp", .{ .needed = true });
    }

    // Expose `test_options` to test cases
    unit_tests.root_module.addOptions("test_options", test_options);
}
