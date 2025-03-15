const std = @import("std");
const test_options = @import("test_options");

const c = @import("c.zig");

pub fn getCallerFilePath(
    allocator: std.mem.Allocator,
    src: std.builtin.SourceLocation,
) ![]const u8 {
    return try std.fs.path.join(allocator, &.{ test_options.src_dir, src.file });
}

pub const FileCheckRunner = struct {
    src_file: []const u8,
    filecheck_bin: ?[]const u8,
    allocator: std.mem.Allocator,
    child: ?std.process.Child = null,
    child_stdin: ?std.fs.File = null,
    fd_stderr: ?i32 = null,
    parent_stderr: ?std.fs.File = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, src: std.builtin.SourceLocation) !Self {
        return .{
            .src_file = try getCallerFilePath(allocator, src),
            .filecheck_bin = std.process.getEnvVarOwned(allocator, "FILECHECK_BIN") catch null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.src_file);
        if (self.filecheck_bin) |v| self.allocator.free(v);
    }

    pub fn canRun(self: Self) bool {
        if (self.filecheck_bin) |v| {
            std.fs.cwd().access(v, .{ .mode = .read_only }) catch {
                std.log.warn("Path of FileCheck is specified but it's not available", .{});
                return false;
            };
            // TODO: check the output of `FileCheck --version`
            return true;
        } else {
            return false;
        }
    }

    /// Run FileCheck and wait for input from stderr.
    /// Note that we use different prefix in each test case to make FileCheck
    /// able to verify the output of specific test case. So that we can run
    /// test cases one by one instead of running all of them in order.
    ///
    /// Regarding other input arguments for FileCheck, we currently support
    /// passing them with environment variable. See [1] for available options.
    ///
    /// [1]: https://llvm.org/docs/CommandGuide/FileCheck.html#options
    pub fn runAndWaitForInput(self: *Self, prefix: ?[]const u8) !void {
        const strfmt = std.fmt.allocPrint;

        const check_prefix = try strfmt(
            self.allocator,
            "--check-prefix={s}",
            .{prefix orelse "CHECK"},
        );
        defer self.allocator.free(check_prefix);

        var child = std.process.Child.init(&.{
            self.filecheck_bin.?,
            self.src_file,
            check_prefix,
        }, self.allocator);
        child.stdin_behavior = .Pipe;

        try child.spawn();

        self.child = child;
        self.parent_stderr = std.io.getStdErr();

        const child_stdin = if (child.stdin) |val| val else @panic("child stdin is not available");

        // Save current stderr fd for later restoration
        self.fd_stderr = @intCast(std.os.linux.dup(self.parent_stderr.?.handle));

        // Pipe stderr of current process to stdin of child process
        // Note that we won't see any output to the terminal after this call
        // (because stderr will be closed first by `dup2()`), so we need to
        // restore it later.
        _ = std.os.linux.dup2(child_stdin.handle, self.parent_stderr.?.handle);
    }

    pub fn cleanup(self: *Self) !?std.process.Child.Term {
        var term: ?std.process.Child.Term = null;

        // Close and clean up streams to make child process able to terminate
        // normally. Otherwise, it will hang when `child.wait()` is called.
        if (self.child) |*child| {
            if (child.stdin) |c_stdin| {
                c_stdin.close();
                // NOTE: we also need to reset it to null to avoid further `close()` on a
                // closed stream. See also [1] for similar situation and [2] for details.
                // [1]: https://www.reddit.com/r/Zig/comments/1dawjr6/comment/l7rvh51/
                // [2]: https://github.com/ziglang/zig/blob/cf90dfd/lib/std/process/Child.zig#L473-L486
                child.stdin = null;
            }

            if (self.parent_stderr) |p_stderr| {
                p_stderr.close();
            }

            term = try child.wait();

            // Restore the original stderr
            _ = std.os.linux.dup2(self.fd_stderr.?, self.parent_stderr.?.handle);
        }

        if (self.parent_stderr) |_| {
            self.parent_stderr = null;
        }

        return term;
    }

    // NOTE: Caller should always check whether `canRun()` is true before
    // running this one.
    pub fn runOnce(
        self: *Self,
        prefix: ?[]const u8,
        func: anytype,
        args: anytype,
    ) !?std.process.Child.Term {
        const ti = @typeInfo(@TypeOf(func));
        if (ti != .Fn) {
            @compileError(std.fmt.comptimePrint(
                "Expect `func` is a function pointer, got {}",
                .{@TypeOf(func)},
            ));
        }

        try self.runAndWaitForInput(prefix);
        _ = @call(.auto, func, args);
        const term = self.cleanup();
        return term;
    }
};

// A helper for redirecting content written in stderr to buffer. So we can
// compare the content directly without calling FileCheck.
pub const StderrToBufferPrintSession = struct {
    buf: []u8,
    pipe_fds: ?[2]c_int = null,
    c_stderr_fd: ?c_int = null,
    n_read: usize = 0,
    allocator: std.mem.Allocator,

    const Self = @This();
    const PrintSessionError = error{
        FailedToCreatePipe,
        FailedToCreateAMemoryStream,
        MemoryStreamIsClosedAlready,
        FailedToReadFromPipe,
        SessionIsNotStartedYet,
    };

    pub fn init(allocator: std.mem.Allocator, buf_size: usize) !Self {
        return .{
            .buf = try allocator.alloc(u8, buf_size),
            .allocator = allocator,
        };
    }

    pub fn start(self: *Self) !void {
        // Reset buffer and counter
        @memset(self.buf, 0);
        self.n_read = 0;

        var fds: [2]c_int = undefined;
        if (c.unistd.pipe(&fds) == -1) {
            return PrintSessionError.FailedToCreatePipe;
        }
        self.pipe_fds = fds;

        // Save the original stderr fd, and redirect stderr to the write end of pipe
        self.c_stderr_fd = c.unistd.dup(c.stdio.fileno(c.stdio.stderr));
        _ = c.unistd.dup2(self.pipe_fds.?[1], c.stdio.fileno(c.stdio.stderr));
    }

    pub fn stop(self: *Self) !void {
        if (self.pipe_fds == null or self.c_stderr_fd == null) {
            return PrintSessionError.SessionIsNotStartedYet;
        }

        // Close the write end of pipe
        _ = c.unistd.close(self.pipe_fds.?[1]);

        // Read to buffer and close the read end of pipe
        const n_read = c.unistd.read(self.pipe_fds.?[0], @ptrCast(self.buf), self.buf.len - 1);
        _ = c.unistd.close(self.pipe_fds.?[0]);

        // Reset pipe_fds to null to avoid malfunction
        self.pipe_fds = null;

        // Restore the original stderr fd, and close the saved one
        _ = c.unistd.dup2(self.c_stderr_fd.?, c.stdio.fileno(c.stdio.stderr));
        _ = c.unistd.close(self.c_stderr_fd.?);
        self.c_stderr_fd = null;

        if (n_read < 0) {
            return PrintSessionError.FailedToReadFromPipe;
        }
        self.n_read = @intCast(n_read);
    }

    pub fn getContent(self: Self) []const u8 {
        return self.buf[0..self.n_read];
    }

    pub fn contentEql(self: Self, expected: []const u8) bool {
        return std.mem.eql(u8, self.getContent(), expected);
    }

    pub fn deinit(self: *Self) void {
        if (self.c_stderr_fd) |fd| {
            _ = c.unistd.dup2(fd, c.stdio.fileno(c.stdio.stderr));
            _ = c.unistd.close(fd);
        }

        if (self.pipe_fds) |fds| {
            _ = c.unistd.close(fds[0]);
            _ = c.unistd.close(fds[1]);
            self.pipe_fds = null;
        }

        self.allocator.free(self.buf);
    }

    pub fn runOnce(
        self: *Self,
        func: anytype,
        args: anytype,
    ) !void {
        const ti = @typeInfo(@TypeOf(func));
        if (ti != .Fn) {
            @compileError(std.fmt.comptimePrint(
                "Expect `func` is a function pointer, got {}",
                .{@TypeOf(func)},
            ));
        }

        try self.start();
        _ = @call(.auto, func, args);
        try self.stop();
    }
};
