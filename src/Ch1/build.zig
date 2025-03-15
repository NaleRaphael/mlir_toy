const std = @import("std");
const bc = @import("../../build_common.zig");

pub fn build(
    b: *std.Build,
    config: bc.BuildConfig,
    _: bc.LibDirs,
    _: bc.MiscConfig,
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = "toyc-ch1",
        .root_source_file = b.path("src/Ch1/toyc.zig"),
        .target = config.target,
        .optimize = config.optimize,
        .pic = config.pic,
    });
    return exe;
}
