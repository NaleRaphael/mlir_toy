const std = @import("std");
const bc = @import("../../build_common.zig");

const CHAPTER_N = "4";
const THIS_DIR = "src/Ch" ++ CHAPTER_N;

const MLIR_LIBS = [_][]const u8{
    "Analysis",
    "IR",
    "Parser",
    "SideEffectInterfaces",
    "Transforms",
    "Support",
    "Pass",
    "CastInterfaces",
};

const MLIR_CAPI_LIBS = [_][]const u8{
    "IR",
    "Transforms",
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
        .name = "toyc-ch" ++ CHAPTER_N,
        .root_source_file = b.path(THIS_DIR ++ "/toyc.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .pic = config.pic,
        // XXX: libc needs to be static linked when it's set by `linkLibC()`,
        // so we don't set the linkage type here.
        // .linkage = config.link_mode,
    });

    if (misc.build_dialect) {
        // Build dialect before building this program
        const cmd_build = b.addSystemCommand(&.{"./build_dialect.sh"});
        cmd_build.setCwd(b.path(THIS_DIR));
        cmd_build.has_side_effects = true;

        const build_shared = if (config.link_mode == .dynamic) "1" else "0";
        cmd_build.setEnvironmentVariable("LLVM_DIR", lib_dirs.llvm_dir);
        cmd_build.setEnvironmentVariable("MLIR_DIR", lib_dirs.mlir_dir);
        cmd_build.setEnvironmentVariable("BUILD_SHARED", build_shared);

        exe.step.dependOn(&cmd_build.step);
    }

    exe.addIncludePath(.{ .cwd_relative = THIS_DIR });
    exe.addIncludePath(.{ .cwd_relative = lib_dirs.mlir_inc });

    exe.addLibraryPath(.{ .cwd_relative = THIS_DIR ++ "/inst_toy/lib" });
    exe.addLibraryPath(.{ .cwd_relative = lib_dirs.mlir_lib });

    // In case the toy dialect and its C-API are dynamic linked, we should mark
    // the toy dialect library `MLIRToy` as needed to prevent its linkage being
    // ignored.
    const link_cfg = std.Build.Module.LinkSystemLibraryOptions{
        .needed = true,
        .preferred_link_mode = config.link_mode orelse .dynamic,
    };
    exe.linkSystemLibrary2("MLIRToy", link_cfg);
    exe.linkSystemLibrary2("ToyCAPI", link_cfg);
    exe.linkLibC();

    // XXX: once using `linkSystemLibrary()` or `linkLibCpp()`, zig will
    // normalize any `libc++`-like query to link against its own `libc++`. So
    // we cannot use this approach when we want to link against the custom one.
    // - https://github.com/ziglang/zig/blob/0.13.0/lib/std/Target.zig#L2742-L2748
    // - https://github.com/ziglang/zig/blob/0.13.0/lib/std/Build/Module.zig#L436-L439
    // - https://github.com/ziglang/zig/blob/0.13.0/lib/std/Build/Step/Compile.zig#L1349-L1351
    if (misc.use_custom_libcxx) {
        const link_mode = config.link_mode orelse .dynamic;
        linkCustomLibCxx(exe, lib_dirs.llvm_lib, link_mode, b.allocator) catch {
            @panic("Failed to search and link libc++, libc++abi, and libunwind");
        };
    } else {
        exe.linkLibCpp();
    }

    bc.linkLibs(exe, "MLIR", &MLIR_LIBS);
    bc.linkLibs(exe, "MLIRCAPI", &MLIR_CAPI_LIBS);
    bc.linkLibs(exe, "LLVM", &LLVM_LIBS);
    return exe;
}

fn linkCustomLibCxx(
    exe: *std.Build.Step.Compile,
    search_root: []const u8,
    link_mode: std.builtin.LinkMode,
    allocator: std.mem.Allocator,
) !void {
    var targets = std.BufSet.init(allocator);
    var candidates = std.ArrayList([]const u8).init(allocator);
    defer {
        for (candidates.items) |v| {
            allocator.free(v);
        }
        candidates.deinit();
        defer targets.deinit();
    }

    try targets.insert("libc++");
    try targets.insert("libc++abi");
    try targets.insert("libunwind");

    const dir = try std.fs.openDirAbsolute(search_root, .{
        .iterate = true,
        .access_sub_paths = false,
        .no_follow = false,
    });

    const build_target = exe.rootModuleTarget();
    const suffix = blk: {
        if (link_mode == .dynamic) {
            break :blk build_target.os.tag.dynamicLibSuffix();
        } else {
            break :blk build_target.os.tag.staticLibSuffix(build_target.abi);
        }
    };

    var dir_iter = dir.iterate();

    while (try dir_iter.next()) |entry| {
        const stem = std.fs.path.stem(entry.name);
        const extension = std.fs.path.extension(entry.name);
        if (targets.contains(stem) and std.mem.eql(u8, extension, suffix)) {
            const name = try allocator.dupe(u8, entry.name);
            try candidates.append(name);
        }
    }

    // Provide details for easier debugging
    if (candidates.items.len != targets.count()) {
        std.debug.print("Expect {d} libraries to find under {s}, found {d}\n", .{
            targets.count(),
            search_root,
            candidates.items.len,
        });
        std.debug.print("Expected: ", .{});

        var iter = targets.iterator();
        while (iter.next()) |v| {
            const name = try std.mem.concat(allocator, u8, &.{ v.*, suffix });
            std.debug.print("{s} ", .{name});
        }
        std.debug.print("\n", .{});

        std.debug.print("Found: ", .{});
        for (candidates.items) |v| {
            std.debug.print("{s} ", .{v});
        }
        std.debug.print("\n", .{});
        return error.FailedToLinkCustomLibCxx;
    }

    for (candidates.items) |v| {
        const lib_path = try std.fs.path.join(allocator, &.{ search_root, v });
        exe.addObjectFile(.{ .cwd_relative = lib_path });
    }

    exe.addLibraryPath(.{ .cwd_relative = search_root });
}
