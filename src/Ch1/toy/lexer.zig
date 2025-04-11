const std = @import("std");
const builtin = @import("builtin");
const utils = @import("utils.zig");

const assert = std.debug.assert;
const StaticStringMap = std.static_string_map.StaticStringMap;

pub const StringRef = struct {
    str: []const u8,
    idx: u64,

    const Self = @This();

    pub fn init(str: []const u8) Self {
        return .{
            .str = str,
            .idx = 0,
        };
    }

    pub fn empty(self: Self) bool {
        return self.idx == self.str.len;
    }

    pub fn front(self: Self) u8 {
        std.debug.assert(self.idx < self.str.len);
        return self.str[self.idx];
    }

    pub fn step(self: *Self, step_size: u64) void {
        std.debug.assert(self.idx + step_size <= self.str.len);
        self.idx += step_size;
    }
};

// NOTE: This is required by MLIR, see also:
// https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#interfacing-with-mlir
pub const Location = struct {
    file: []const u8,
    line: u32,
    col: u32,
};

pub const Token = union(enum) {
    tok_ident: []const u8,
    tok_num: f64,
    tok_other: u8,

    tok_semicolon,
    tok_parenthese_open,
    tok_parenthese_close,
    tok_cbracket_open,
    tok_cbracket_close,
    tok_sbracket_open,
    tok_sbracket_close,
    tok_abracket_open,
    tok_abracket_close,
    tok_eof,

    // commands
    tok_return,
    tok_var,
    tok_def,

    // A helper function to return a token as keyword or identifier.
    // ref: https://github.com/ThePrimeagen/ts-rust-zig-deez/blob/3441b21/zig/src/lexer/lexer.zig#L36-L47
    fn keyword(ident: []const u8) ?Token {
        const map = StaticStringMap(Token).initComptime(.{
            .{ "return", .tok_return },
            .{ "var", .tok_var },
            .{ "def", .tok_def },
        });
        return map.get(ident);
    }

    pub fn str(self: @This(), buf: *[1]u8) []const u8 {
        return switch (self) {
            .tok_ident => |ident| ident,
            .tok_other => |v| {
                const ret = std.fmt.bufPrint(buf, "{c}", .{v}) catch "";
                return ret;
            },
            .tok_semicolon => ";",
            .tok_parenthese_open => "(",
            .tok_parenthese_close => ")",
            .tok_cbracket_open => "{",
            .tok_cbracket_close => "}",
            .tok_sbracket_open => "[",
            .tok_sbracket_close => "]",
            .tok_abracket_open => "<",
            .tok_abracket_close => ">",
            .tok_eof, .tok_return, .tok_var, .tok_def, .tok_num => @tagName(self),
        };
    }
};

pub const LexerError = error{
    FailedToParseNumber,
};

pub const Lexer = struct {
    cur_tok: Token = .tok_eof,
    last_location: Location,
    identifier_str: []const u8,
    num_val: f64 = 0,
    last_char: ?u8 = ' ', // NOTE: if we ran into EOF, this will be set to null.
    cur_line: u32 = 0,
    cur_col: u32 = 0,
    cur_line_buf: StringRef,
    buffer: LexerBuffer,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(filename: []const u8, allocator: std.mem.Allocator) !*Self {
        const file = try std.fs.cwd().openFile(filename, .{ .mode = .read_only });
        defer file.close();

        const ptr = try allocator.create(Self);
        errdefer allocator.destroy(ptr);

        ptr.* = .{
            .last_location = .{
                .file = filename,
                .line = 0,
                .col = 0,
            },
            .identifier_str = "",
            .cur_line_buf = StringRef.init("\n"),
            .buffer = try LexerBuffer.init(file),
            .allocator = allocator,
        };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        self.buffer.deinit();
        self.allocator.destroy(self);
    }

    pub fn getCurToken(self: Self) Token {
        return self.cur_tok;
    }

    pub fn getNextToken(self: *Self) LexerError!Token {
        self.cur_tok = try self.getTok();
        return self.cur_tok;
    }

    pub fn consume(self: *Self, tok: Token) LexerError!void {
        switch (self.cur_tok) {
            .tok_num => |v| assert(utils.isClose(f64, v, self.cur_tok.tok_num)),
            else => assert(std.meta.eql(tok, self.cur_tok)),
        }
        _ = try self.getNextToken();
    }

    pub fn getId(self: Self) []const u8 {
        assert(self.cur_tok == Token.tok_ident);
        return self.identifier_str;
    }

    pub fn getValue(self: Self) f64 {
        assert(self.cur_tok == Token.tok_num);
        return self.num_val;
    }

    pub fn getLastLocation(self: Self) Location {
        return self.last_location;
    }

    pub fn getLine(self: Self) u32 {
        return self.cur_line;
    }

    pub fn getCol(self: Self) u32 {
        return self.cur_col;
    }

    pub fn getTok(self: *Self) LexerError!Token {
        self.skipWhitespace();

        self.last_location.line = self.cur_line;
        self.last_location.col = self.cur_col;

        const tok: Token = if (self.last_char) |c| switch (c) {
            ';' => .tok_semicolon,
            '(' => .tok_parenthese_open,
            ')' => .tok_parenthese_close,
            '{' => .tok_cbracket_open,
            '}' => .tok_cbracket_close,
            '<' => .tok_abracket_open,
            '>' => .tok_abracket_close,
            '[' => .tok_sbracket_open,
            ']' => .tok_sbracket_close,
            'a'...'z', 'A'...'Z', '_' => {
                self.identifier_str = self.readIdentifier();
                if (Token.keyword(self.identifier_str)) |val| {
                    return val;
                }
                return .{ .tok_ident = self.identifier_str };
            },
            '0'...'9' => {
                self.num_val = try self.readNumber();
                return .{ .tok_num = self.num_val };
            },
            '#' => {
                self.skipComment();
                return try self.getTok();
            },
            else => .{ .tok_other = c },
        } else {
            return .tok_eof; // check for end of file (so we don't consume EOF here)
        };

        self.last_char = self.getNextChar();
        return tok;
    }

    fn getNextChar(self: *Self) ?u8 {
        if (self.cur_line_buf.empty()) return null; // EOF

        self.cur_col += 1;
        const next_char = self.cur_line_buf.front();
        self.cur_line_buf.step(1);
        if (self.cur_line_buf.empty()) {
            self.cur_line_buf = self.buffer.readNextLine();
        }
        if (next_char == '\n') {
            self.cur_line += 1;
            self.cur_col = 0;
        }
        return next_char;
    }

    fn readIdentifier(self: *Self) []const u8 {
        const start = self.cur_col - 1; // NOTE: cur_col is 1-based indexing
        const cur_line = self.cur_line_buf.str;

        var n_char: usize = 0;
        while (self.last_char) |c| {
            if (std.ascii.isAlphanumeric(c) or c == '_') {
                n_char += 1;
                self.last_char = self.getNextChar();
            } else break;
        }

        const ident = cur_line[start..][0..n_char];
        return ident;
    }

    fn readNumber(self: *Self) LexerError!f64 {
        const start = self.cur_col - 1;
        const cur_line = self.cur_line_buf.str;

        var n_char: usize = 0;
        while (self.last_char) |c| {
            if (std.ascii.isDigit(c) or c == '.') {
                n_char += 1;
                self.last_char = self.getNextChar();
            } else break;
        }

        const val = cur_line[start..][0..n_char];
        const num = std.fmt.parseFloat(f64, val) catch {
            // Wrap `ParseFloatError` to unify the type of error to return
            return LexerError.FailedToParseNumber;
        };
        return num;
    }

    fn skipWhitespace(self: *Self) void {
        while (self.last_char) |c| {
            if (std.ascii.isWhitespace(c)) {
                self.last_char = self.getNextChar();
            } else break;
        }
    }

    fn skipComment(self: *Self) void {
        self.last_char = self.getNextChar();
        while (self.last_char) |c| {
            if (c != '\n' and c != '\r') {
                self.last_char = self.getNextChar();
            } else break;
        }
    }
};

/// Just a buffer storing file content in memory, so we don't need to hold the
/// file handler during the process.
pub const LexerBuffer = struct {
    current: u64,
    end: u64,
    data: ?[]align(std.mem.page_size) u8,

    const Self = @This();

    pub fn init(file: std.fs.File) !Self {
        const file_size = (try file.stat()).size;
        const data = try std.posix.mmap(
            null,
            file_size,
            std.posix.system.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(data);

        return .{
            .data = data,
            .current = 0,
            .end = file_size,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.data) |data_ptr| {
            std.posix.munmap(data_ptr);
            self.data = null;
        }
    }

    pub fn readNextLine(self: *Self) StringRef {
        const begin = self.current;
        while (self.current < self.end and self.data.?[self.current] != '\n') {
            self.current += 1;
        }
        // If we haven't reach to the last line, read the newline char as well.
        if (self.current < self.end) {
            self.current += 1;
        }
        const len = self.current - begin;
        return StringRef.init(self.data.?[begin..][0..len]);
    }
};

test "Lexer__getTok" {
    const content =
        \\var foo = 42.1;
        \\def multiply_transpose(a, b) {
        \\  return transpose(a) * transpose(b);
        \\};
    ;
    const filename = "test__Lexer__getTok.zig";

    const expected_tokens = [_]Token{
        .tok_var,
        .{ .tok_ident = "foo" },
        .{ .tok_other = 61 },
        .{ .tok_num = 4.21e1 },
        .tok_semicolon,
        .tok_def,
        .{ .tok_ident = "multiply_transpose" },
        .tok_parenthese_open,
        .{ .tok_ident = "a" },
        .{ .tok_other = 44 },
        .{ .tok_ident = "b" },
        .tok_parenthese_close,
        .tok_cbracket_open,
        .tok_return,
        .{ .tok_ident = "transpose" },
        .tok_parenthese_open,
        .{ .tok_ident = "a" },
        .tok_parenthese_close,
        .{ .tok_other = 42 },
        .{ .tok_ident = "transpose" },
        .tok_parenthese_open,
        .{ .tok_ident = "b" },
        .tok_parenthese_close,
        .tok_semicolon,
        .tok_cbracket_close,
        .tok_semicolon,
    };

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = filename, .data = content });

    var path_buf: [256]u8 = undefined;
    const tmpfile_path = try tmp.dir.realpath(filename, &path_buf);

    var lexer = try Lexer.init(tmpfile_path, std.testing.allocator);
    defer lexer.deinit();

    for (expected_tokens) |expected| {
        const tok = try lexer.getTok();
        try std.testing.expectEqualDeep(tok, expected);
    }
    const tok = try lexer.getTok();
    try std.testing.expectEqualDeep(tok, Token.tok_eof);
}

test "LexerBuffer__readline_memfd" {
    if (builtin.target.os.tag != .linux) return error.SkipZigTest;

    // Skip this for linux kernel < 3.17
    // https://man7.org/linux/man-pages/man2/memfd_create.2.html#HISTORY
    {
        var utsname: std.os.linux.utsname = undefined;
        _ = std.os.linux.uname(&utsname);
        std.debug.assert(std.ascii.isDigit(utsname.release[0]));

        var ver = [_]u32{0} ** 3;
        var n_parsed: usize = 0;
        var n_period: usize = 0;

        for (utsname.release) |c| {
            if (n_parsed >= 3) break;

            if (std.ascii.isDigit(c)) {
                const num = try std.fmt.parseInt(u32, &[1]u8{c}, 10);
                const res = @mulWithOverflow(ver[n_parsed], 10);
                // If it's overflow, there might be something wrong while
                // parsing version. e.g., unexpected length of version number
                std.debug.assert(res[1] == 0);
                ver[n_parsed] = res[0] + num;
            } else {
                n_parsed += 1;
                n_period += if (c == '.') 1 else 0;
            }
        }
        std.debug.assert(n_period == 2);

        if (ver[0] <= 3 and ver[1] < 17) return error.SkipZigTest;
    }

    const content =
        \\const std = @import("std");
        \\
        \\pub fn main() void {
        \\    std.debug.print("foobar", .{});
        \\}
    ;
    const filename = "test__LexerBuffer__readline_memfd.zig";

    const memfd = try std.posix.memfd_create(filename, std.posix.MFD.CLOEXEC);

    // Set size of the in-memory file
    try std.posix.ftruncate(memfd, content.len);

    // Write data into in-memory file
    const write_len = try std.posix.write(memfd, content);
    try std.testing.expect(write_len == content.len);

    // Wrap the fd in `File`
    const memfile = std.fs.File{ .handle = memfd };
    defer memfile.close();

    var lexbuf = try LexerBuffer.init(memfile);
    var line_cnt: u32 = 0;
    var idx_start: usize = 0;
    var idx_end: usize = 0;

    var strref = lexbuf.readNextLine();
    while (!strref.empty()) {
        line_cnt += 1;
        idx_end = std.mem.indexOfScalarPos(u8, content, idx_start, '\n') orelse content.len - 1;
        idx_end += 1; // include the newline char

        const expected_line = content[idx_start..idx_end];
        try std.testing.expect(std.mem.eql(u8, expected_line, strref.str));

        idx_start = idx_end;
        strref = lexbuf.readNextLine();
    }

    try std.testing.expect(idx_end == content.len);
}

test "LexerBuffer__readline" {
    const content =
        \\const std = @import("std");
        \\
        \\pub fn main() void {
        \\    std.debug.print("foobar", .{});
        \\}
    ;
    const filename = "test__LexerBuffer__readline_memfd.zig";

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // - approach 1:
    try tmp.dir.writeFile(.{ .sub_path = filename, .data = content });
    const tmpfile = try tmp.dir.openFile(filename, .{});

    // - approach 2:
    // {
    //     var tmpfile = try tmp.dir.createFile(filename, .{});
    //     defer tmpfile.close();
    //
    //     const write_len = try tmpfile.write(content);
    //     try std.testing.expect(write_len == content.len);
    // }
    // const tmpfile = try tmp.dir.openFile(filename, .{});

    // var tmpfile = try tmp.dir.createFile(filename, .{});
    // const write_len = try tmpfile.write(content);
    // try std.testing.expect(write_len == content.len);

    var lexbuf = try LexerBuffer.init(tmpfile);
    var line_cnt: u32 = 0;
    var idx_start: usize = 0;
    var idx_end: usize = 0;

    var strref = lexbuf.readNextLine();
    while (!strref.empty()) {
        line_cnt += 1;
        idx_end = std.mem.indexOfScalarPos(u8, content, idx_start, '\n') orelse content.len - 1;
        idx_end += 1; // include the newline char

        const expected_line = content[idx_start..idx_end];
        try std.testing.expect(std.mem.eql(u8, expected_line, strref.str));

        idx_start = idx_end;
        strref = lexbuf.readNextLine();
    }

    try std.testing.expect(idx_end == content.len);
}
