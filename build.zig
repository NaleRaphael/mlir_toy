const std = @import("std");
const bc = @import("build_common.zig");

/// The prototype of `build()` to define in `build.zig` of each submodule.
const BuildProto = *const fn (
    b: *std.Build,
    config: bc.BuildConfig,
    lib_dirs: bc.LibDirs,
    misc: bc.MiscConfig,
) *std.Build.Step.Compile;

const Chapter = enum { ch1, ch2, ch3, ch4, ch5 };

pub fn getChapterBuildFn(ch: Chapter) BuildProto {
    return switch (ch) {
        .ch1 => @import("src/Ch1/build.zig").build,
        .ch2 => @import("src/Ch2/build.zig").build,
        .ch3 => @import("src/Ch3/build.zig").build,
        .ch4 => @import("src/Ch4/build.zig").build,
        .ch5 => @import("src/Ch5/build.zig").build,
        // NOTE: this one is excluded by default. If you want to build it,
        // please uncomment this and add "sample" back to the `Chapter` enum.
        // .sample => @import("src/sample/build.zig").build,
    };
}

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const options = .{
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
        .chapters = b.option(
            []const u8,
            "chapters",
            "Chapters to build (comma separated), default to \"all\" if not specified.",
        ) orelse "all",
        .build_dialect = b.option(
            bool,
            "build_dialect",
            "Run `build_dialect.sh` in each chapter folder before building Zig program." ++
                "If it's set to false, user has to run it manually. (default: true)",
        ) orelse true,
        .link_mode = b.option(
            std.builtin.LinkMode,
            "link_mode",
            "Build and link the dialect library as a static/dynamic library. " ++
                "(default: null)",
        ),
        .use_custom_libcxx = b.option(
            bool,
            "use_custom_libcxx",
            "Use the libc++ supplied under `$LLVM_DIR` instead of the one built by Zig.",
        ) orelse false,
        // TODO: this option seems not been used anymore, maybe we can remove it?
        .bin_llvm_lit = b.option(
            []const u8,
            "bin_llvm_lit",
            "Path of `llvm-lit` tool",
        ) orelse "",
    };
    const options_step = b.addOptions();
    inline for (std.meta.fields(@TypeOf(options))) |field| {
        options_step.addOption(field.type, field.name, @field(options, field.name));
    }

    const build_config = bc.BuildConfig{
        .target = target,
        .optimize = optimize,
        .link_mode = options.link_mode,
        .pic = true,
    };
    const misc_config = bc.MiscConfig{
        .build_dialect = options.build_dialect,
        .use_custom_libcxx = options.use_custom_libcxx,
        .bin_llvm_lit = options.bin_llvm_lit,
    };

    const path_join = std.fs.path.join;
    const str_t = []const u8;
    const alloc = b.allocator;
    const lib_dirs = bc.LibDirs{
        .mlir_dir = options.mlir_dir,
        .mlir_inc = try path_join(alloc, &[_]str_t{ options.mlir_dir, "include" }),
        .mlir_lib = try path_join(alloc, &[_]str_t{ options.mlir_dir, "lib" }),
        .llvm_dir = options.llvm_dir,
        .llvm_inc = try path_join(alloc, &[_]str_t{ options.llvm_dir, "include" }),
        .llvm_lib = try path_join(alloc, &[_]str_t{ options.llvm_dir, "lib" }),
    };

    if (std.mem.eql(u8, options.chapters, "all")) {
        for (std.meta.tags(Chapter)) |ch| {
            const buildFn = getChapterBuildFn(ch);
            const exe = buildFn(b, build_config, lib_dirs, misc_config);
            b.installArtifact(exe);
        }
    } else {
        var it = std.mem.splitSequence(u8, options.chapters, ",");
        while (it.next()) |part| {
            const ch = std.meta.stringToEnum(Chapter, part) orelse {
                std.debug.print("Found unknown chapter to build: \"{s}\", " ++
                    "please check the input list \"-Dchapters=...\"\n", .{part});
                return error.UnknownChapterToBuild;
            };

            const buildFn = getChapterBuildFn(ch);
            const exe = buildFn(b, build_config, lib_dirs, misc_config);
            b.installArtifact(exe);
        }
    }
}
