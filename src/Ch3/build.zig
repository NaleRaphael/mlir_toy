const std = @import("std");
const bc = @import("../../build_common.zig");

const MLIR_LIBS = [_][]const u8{
    "Analysis",
    "IR",
    "Parser",
    "SideEffectInterfaces",
    "Transforms",
    "Support",
    "Dialect",
};

const MLIR_CAPI_LIBS = [_][]const u8{
    "IR",
};

const LLVM_LIBS = [_][]const u8{
    "Support",
};

// NOTE: for the dialect library, it's currently fine to link against to Zig's
// libc++ if it's already built with LLVM's libc++. But we keep this option
// available in case we want to link to a custom built libc++ for any reason.
// const USE_CUSTOM_LIBCXX = true;

pub fn build(
    b: *std.Build,
    config: bc.BuildConfig,
    lib_dirs: bc.LibDirs,
    misc: bc.MiscConfig,
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = "toyc-ch3",
        .root_source_file = b.path("src/Ch3/toyc.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .linkage = config.link_mode,
        .pic = config.pic,
    });

    if (misc.build_dialect) {
        // Build dialect before building this program
        const cmd_build = b.addSystemCommand(&.{"./build_dialect.sh"});
        cmd_build.setCwd(b.path("src/Ch3"));
        cmd_build.has_side_effects = true;

        const build_shared = if (config.link_mode == .dynamic) "1" else "0";
        cmd_build.setEnvironmentVariable("LLVM_DIR", lib_dirs.llvm_dir);
        cmd_build.setEnvironmentVariable("MLIR_DIR", lib_dirs.mlir_dir);
        cmd_build.setEnvironmentVariable("BUILD_SHARED", build_shared);

        exe.step.dependOn(&cmd_build.step);
    }

    exe.addIncludePath(.{ .cwd_relative = "src/Ch3" });
    exe.addIncludePath(.{ .cwd_relative = lib_dirs.mlir_inc });

    exe.addLibraryPath(.{ .cwd_relative = "src/Ch3/inst_toy/lib" });
    exe.addLibraryPath(.{ .cwd_relative = lib_dirs.mlir_lib });

    if (misc.use_custom_libcxx) {
        linkLibCxxAndCxxabi(exe, lib_dirs.llvm_lib, true, b.allocator) catch {
            @panic("Failed to search and link libc++ and libc++abi");
        };
    } else {
        // XXX: once using `linkSystemLibrary()`, zig will normalize any
        // `libc++`-like query to its own `libc++`. So we cannot use this
        // approach when we want to link against to custom `libc++`.
        // - https://github.com/ziglang/zig/blob/0.13.0/lib/std/Target.zig#L2742-L2748
        // - https://github.com/ziglang/zig/blob/0.13.0/lib/std/Build/Module.zig#L436-L439
        // - https://github.com/ziglang/zig/blob/0.13.0/lib/std/Build/Step/Compile.zig#L1349-L1351
        exe.linkSystemLibrary("c++");
        exe.linkSystemLibrary("c++abi");
    }

    // In case the toy dialect and its C-API are dynamic linked, we should mark
    // the toy dialect library `MLIRToy` as needed to prevent its linkage being
    // ignored.
    exe.linkSystemLibrary2("MLIRToy", .{ .needed = true });
    exe.linkSystemLibrary("ToyCAPI");

    exe.linkLibC();
    if (!misc.use_custom_libcxx) {
        // XXX: don't use the builtin function to link `libc++`, see also the
        // comment above for the same reason.
        exe.linkLibCpp();
    }

    bc.linkLibs(exe, "MLIR", &MLIR_LIBS);
    bc.linkLibs(exe, "MLIRCAPI", &MLIR_CAPI_LIBS);
    bc.linkLibs(exe, "LLVM", &LLVM_LIBS);
    return exe;
}

fn linkLibCxxAndCxxabi(
    exe: *std.Build.Step.Compile,
    search_root: []const u8,
    prefer_dynlib: bool,
    allocator: std.mem.Allocator,
) !void {
    const dir = try std.fs.openDirAbsolute(search_root, .{
        .iterate = true,
        .access_sub_paths = true,
        .no_follow = false,
    });
    var walker = try dir.walk(allocator);
    defer walker.deinit();

    var libcxx_candiates = std.ArrayList([]const u8).init(allocator);
    var libcxxabi_candiates = std.ArrayList([]const u8).init(allocator);
    defer {
        for (libcxx_candiates.items, libcxxabi_candiates.items) |v1, v2| {
            allocator.free(v1);
            allocator.free(v2);
        }
        libcxx_candiates.deinit();
        libcxxabi_candiates.deinit();
    }

    while (try walker.next()) |entry| {
        if (std.mem.startsWith(u8, entry.basename, "libc++")) {
            const stem = std.fs.path.stem(entry.basename);
            if (std.mem.eql(u8, stem, "libc++")) {
                const path = try allocator.dupe(u8, entry.path);
                try libcxx_candiates.append(path);
                continue;
            }
            if (std.mem.eql(u8, stem, "libc++abi")) {
                const path = try allocator.dupe(u8, entry.path);
                try libcxxabi_candiates.append(path);
                continue;
            }
        }
    }

    const suffix = if (prefer_dynlib) ".so" else ".a";

    for (libcxx_candiates.items) |v| {
        if (std.mem.endsWith(u8, v, suffix)) {
            const lib_path = try std.fs.path.join(allocator, &.{ search_root, v });
            exe.addObjectFile(.{ .cwd_relative = lib_path });
            break;
        }
    } else @panic("cannot link to libc++ because it's not found");

    for (libcxxabi_candiates.items) |v| {
        if (std.mem.endsWith(u8, v, suffix)) {
            const lib_path = try std.fs.path.join(allocator, &.{ search_root, v });
            exe.addObjectFile(.{ .cwd_relative = lib_path });
            break;
        }
    } else @panic("cannot link to libc++abi because it's not found");

    std.debug.assert(libcxx_candiates.items.len > 0);
    const libcxx_dir = try std.fs.path.join(
        allocator,
        &.{ search_root, std.fs.path.dirname(libcxx_candiates.items[0]) orelse "" },
    );
    exe.addLibraryPath(.{ .cwd_relative = libcxx_dir });
}
