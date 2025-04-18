const std = @import("std");
const lexer = @import("lexer.zig");
const utils = @import("utils.zig");

const test_alloc = std.testing.allocator;
const test_expect = std.testing.expect;
const isClose = utils.isClose;

// Usage of `ListType` (assume `T = ListType(ElementType)`):
// 1. Create a temporary ArrayList `al` by `T.ArrayList.init()`.
// 2. Fill data into that ArrayList `al`.
// 3. Create a desired instance `obj` with type `T` by `T.fromArrayList(al)`
// 4. Use the instance `obj` directly or pass it to any other AST type's
//    `init()`.
// 5. Call `obj.deinit()` whenever we want to free the memory.
//    (If `obj` is passed to another AST instance, `obj.deinit()` should be
//    called from that instance's `deinit()`. That is, the owndership is
//    handed over to that instance. This principle is also applied to the
//    temporary ArraryList created in step 1.)
fn ListType(comptime T: type) type {
    return struct {
        slice: SliceType,
        allocator: std.mem.Allocator,

        const Self = @This();
        pub const ArrayList = std.ArrayList(T);
        pub const SliceType = []T;
        pub const ElementType = T;

        pub fn fromArrayList(data: *Self.ArrayList) !Self {
            return .{
                .slice = try data.toOwnedSlice(),
                .allocator = data.allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.slice);
        }
    };
}

// See also `ListType` for the usage
pub const VarType = struct {
    shape: []i64,
    allocator: std.mem.Allocator,

    const Self = @This();
    pub const ArrayList = std.ArrayList(i64);
    pub const ElementType = i64;

    pub fn fromArrayList(data: *Self.ArrayList) !Self {
        return .{
            .shape = try data.toOwnedSlice(),
            .allocator = data.allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.shape);
    }
};

pub const ExprASTKind = enum {
    VarDecl,
    Return,
    Num,
    Literal,
    Var,
    BinOp,
    Call,
    Print,
};

pub const ExprAST = union(ExprASTKind) {
    VarDecl: *VarDeclExprAST,
    Return: *ReturnExprAST,
    Num: *NumberExprAST,
    Literal: *LiteralExprAST,
    Var: *VariableExprAST,
    BinOp: *BinaryExprAST,
    Call: *CallExprAST,
    Print: *PrintExprAST,

    const Self = @This();

    pub fn getKind(self: Self) ExprASTKind {
        return std.meta.activeTag(self);
    }

    pub fn loc(self: Self) lexer.Location {
        return switch (self) {
            inline else => |v| v.loc(),
        };
    }

    pub fn deinit(self: Self) void {
        switch (self) {
            inline else => |v| v.deinit(),
        }
    }
};

pub fn BaseExprAST(comptime Context: type) type {
    return struct {
        context: *Context,
        _kind: ExprASTKind,
        _loc: lexer.Location,
        alloc: std.mem.Allocator,

        const Self = @This();

        pub fn init(
            context: *Context,
            _kind: ExprASTKind,
            _loc: lexer.Location,
            alloc: std.mem.Allocator,
        ) Self {
            return .{ .context = context, ._kind = _kind, ._loc = _loc, .alloc = alloc };
        }

        pub fn deinit(self: *Self) void {
            self.alloc.destroy(self.context);
        }
    };
}

pub const ExprASTListType = ListType(ExprAST);
pub const ExprASTList = ExprASTListType.SliceType;

pub const NumberExprAST = struct {
    base: BaseType,
    val: f64,

    const Self = @This();
    const BaseType = BaseExprAST(Self);

    pub fn init(alloc: std.mem.Allocator, _loc: lexer.Location, val: f64) !*Self {
        const ptr = try alloc.create(Self);
        ptr.* = .{
            .base = BaseType.init(ptr, .Num, _loc, alloc),
            .val = val,
        };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        self.base.deinit();
    }

    pub fn getKind(self: Self) ExprASTKind {
        return self.base._kind;
    }

    pub fn loc(self: Self) lexer.Location {
        return self.base._loc;
    }

    pub fn tagged(self: *Self) ExprAST {
        return .{ .Num = self };
    }

    pub fn getValue(self: Self) f64 {
        return self.val;
    }
};

pub const LiteralExprAST = struct {
    base: BaseType,
    values: ExprASTListType,
    dims: VarType,

    const Self = @This();
    const BaseType = BaseExprAST(Self);

    pub fn init(
        alloc: std.mem.Allocator,
        _loc: lexer.Location,
        values: ExprASTListType,
        dims: VarType,
    ) !*Self {
        const ptr = try alloc.create(Self);
        ptr.* = .{
            .base = BaseType.init(ptr, .Literal, _loc, alloc),
            .values = values,
            .dims = dims,
        };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        for (self.values.slice) |*v| {
            v.deinit();
        }
        self.values.deinit();
        self.dims.deinit();
        self.base.deinit();
    }

    pub fn getKind(self: Self) ExprASTKind {
        return self.base._kind;
    }

    pub fn loc(self: Self) lexer.Location {
        return self.base._loc;
    }

    pub fn tagged(self: *Self) ExprAST {
        return .{ .Literal = self };
    }

    pub fn getValues(self: Self) ExprASTList {
        return self.values.slice;
    }

    pub fn getDims(self: Self) VarType {
        return self.dims;
    }
};

pub const VariableExprAST = struct {
    base: BaseType,
    name: []const u8,

    const Self = @This();
    const BaseType = BaseExprAST(Self);

    pub fn init(alloc: std.mem.Allocator, _loc: lexer.Location, name: []const u8) !*Self {
        const ptr = try alloc.create(Self);
        ptr.* = .{
            .base = BaseType.init(ptr, .Var, _loc, alloc),
            .name = name,
        };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        self.base.deinit();
    }

    pub fn getKind(self: Self) ExprASTKind {
        return self.base._kind;
    }

    pub fn loc(self: Self) lexer.Location {
        return self.base._loc;
    }

    pub fn tagged(self: *Self) ExprAST {
        return .{ .Var = self };
    }

    pub fn getName(self: Self) []const u8 {
        return self.name;
    }
};

pub const VarDeclExprAST = struct {
    base: BaseType,
    name: []const u8,
    type: VarType,
    init_val: ?ExprAST,

    const Self = @This();
    const BaseType = BaseExprAST(Self);

    pub fn init(
        alloc: std.mem.Allocator,
        _loc: lexer.Location,
        name: []const u8,
        _type: VarType,
        init_val: ?ExprAST,
    ) !*Self {
        const ptr = try alloc.create(Self);
        ptr.* = .{
            .base = BaseType.init(ptr, .VarDecl, _loc, alloc),
            .name = name,
            .type = _type,
            .init_val = init_val,
        };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        self.type.deinit();
        if (self.init_val) |*v| {
            v.deinit();
        }
        self.base.deinit();
    }

    pub fn getKind(self: Self) ExprASTKind {
        return self.base._kind;
    }

    pub fn loc(self: Self) lexer.Location {
        return self.base._loc;
    }

    pub fn tagged(self: *Self) ExprAST {
        return .{ .VarDecl = self };
    }

    pub fn getName(self: Self) []const u8 {
        return self.name;
    }

    pub fn getType(self: Self) VarType {
        return self.type;
    }

    pub fn getInitVal(self: Self) ?ExprAST {
        return self.init_val;
    }
};

pub const ReturnExprAST = struct {
    base: BaseType,
    expr: ?ExprAST,

    const Self = @This();
    const BaseType = BaseExprAST(Self);

    pub fn init(alloc: std.mem.Allocator, _loc: lexer.Location, expr: ?ExprAST) !*Self {
        const ptr = try alloc.create(Self);
        ptr.* = .{
            .base = BaseType.init(ptr, .Return, _loc, alloc),
            .expr = expr,
        };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        if (self.expr) |v| {
            v.deinit();
        }
        self.base.deinit();
    }

    pub fn getKind(self: Self) ExprASTKind {
        return self.base._kind;
    }

    pub fn loc(self: Self) lexer.Location {
        return self.base._loc;
    }

    pub fn tagged(self: *Self) ExprAST {
        return .{ .Return = self };
    }

    pub fn getExpr(self: Self) ?ExprAST {
        return self.expr;
    }
};

pub const BinaryExprAST = struct {
    base: BaseType,
    op: u8,
    lhs: ExprAST,
    rhs: ExprAST,

    const Self = @This();
    const BaseType = BaseExprAST(Self);

    pub fn init(
        alloc: std.mem.Allocator,
        _loc: lexer.Location,
        op: u8,
        lhs: ExprAST,
        rhs: ExprAST,
    ) !*Self {
        const ptr = try alloc.create(Self);
        ptr.* = .{ .base = BaseType.init(ptr, .BinOp, _loc, alloc), .op = op, .lhs = lhs, .rhs = rhs };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        self.lhs.deinit();
        self.rhs.deinit();
        self.base.deinit();
    }

    pub fn getKind(self: Self) ExprASTKind {
        return self.base._kind;
    }

    pub fn loc(self: Self) lexer.Location {
        return self.base._loc;
    }

    pub fn tagged(self: *Self) ExprAST {
        return .{ .BinOp = self };
    }

    pub fn getOp(self: Self) u8 {
        return self.op;
    }

    pub fn getLHS(self: Self) ExprAST {
        return self.lhs;
    }

    pub fn getRHS(self: Self) ExprAST {
        return self.rhs;
    }
};

pub const CallExprAST = struct {
    base: BaseType,
    callee: []const u8,
    args: ExprASTListType,

    const Self = @This();
    const BaseType = BaseExprAST(Self);

    pub fn init(
        alloc: std.mem.Allocator,
        _loc: lexer.Location,
        callee: []const u8,
        args: ExprASTListType,
    ) !*Self {
        const ptr = try alloc.create(Self);
        ptr.* = .{
            .base = BaseType.init(ptr, .Call, _loc, alloc),
            .callee = callee,
            .args = args,
        };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        for (self.args.slice) |*arg| {
            arg.deinit();
        }
        self.args.deinit();
        self.base.deinit();
    }

    pub fn getKind(self: Self) ExprASTKind {
        return self.base._kind;
    }

    pub fn loc(self: Self) lexer.Location {
        return self.base._loc;
    }

    pub fn tagged(self: *Self) ExprAST {
        return .{ .Call = self };
    }

    pub fn getCallee(self: Self) []const u8 {
        return self.callee;
    }

    pub fn getArgs(self: Self) ExprASTList {
        return self.args.slice;
    }
};

pub const PrintExprAST = struct {
    base: BaseType,
    arg: ExprAST,

    const Self = @This();
    const BaseType = BaseExprAST(Self);

    pub fn init(alloc: std.mem.Allocator, _loc: lexer.Location, arg: ExprAST) !*Self {
        const ptr = try alloc.create(Self);
        ptr.* = .{ .base = BaseType.init(ptr, .Print, _loc, alloc), .arg = arg };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        self.arg.deinit();
        self.base.deinit();
    }

    pub fn getKind(self: Self) ExprASTKind {
        return self.base._kind;
    }

    pub fn loc(self: Self) lexer.Location {
        return self.base._loc;
    }

    pub fn tagged(self: *Self) ExprAST {
        return .{ .Print = self };
    }

    pub fn getArg(self: Self) ExprAST {
        return self.arg;
    }
};

pub const VariableExprASTListType = ListType(*VariableExprAST);
pub const VariableExprASTList = VariableExprASTListType.SliceType;

pub const PrototypeAST = struct {
    _loc: lexer.Location,
    name: []const u8,
    args: ArgsType,
    alloc: std.mem.Allocator,

    pub const ArgsType = VariableExprASTListType;
    const Self = @This();

    pub fn init(
        alloc: std.mem.Allocator,
        _loc: lexer.Location,
        name: []const u8,
        args: ArgsType,
    ) !*Self {
        const ptr = try alloc.create(Self);
        ptr.* = .{ ._loc = _loc, .name = name, .args = args, .alloc = alloc };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        for (self.args.slice) |arg| {
            arg.deinit();
        }
        self.args.deinit();
        self.alloc.destroy(self);
    }

    pub fn loc(self: Self) lexer.Location {
        return self._loc;
    }

    pub fn getName(self: Self) []const u8 {
        return self.name;
    }

    pub fn getArgs(self: Self) ArgsType.SliceType {
        return self.args.slice;
    }
};

pub const FunctionAST = struct {
    proto: *PrototypeAST,
    body: BodyType,
    alloc: std.mem.Allocator,

    pub const BodyType = ExprASTListType;
    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, proto: *PrototypeAST, body: BodyType) !*Self {
        const ptr = try alloc.create(Self);
        ptr.* = .{ .proto = proto, .body = body, .alloc = alloc };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        self.proto.deinit();
        for (self.body.slice) |*v| {
            v.deinit();
        }
        self.body.deinit();
        self.alloc.destroy(self);
    }

    pub fn getProto(self: Self) *PrototypeAST {
        return self.proto;
    }

    pub fn getBody(self: Self) BodyType.SliceType {
        return self.body.slice;
    }
};

pub const FunctionASTListType = ListType(*FunctionAST);
pub const FunctionASTList = FunctionASTListType.SliceType;

pub const ModuleAST = struct {
    functions: FunctionASTListType,
    alloc: std.mem.Allocator,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, functions: FunctionASTListType) !*Self {
        const ptr = try alloc.create(Self);
        ptr.* = .{ .functions = functions, .alloc = alloc };
        return ptr;
    }

    pub fn deinit(self: *Self) void {
        for (self.functions.slice) |func| {
            func.deinit();
        }
        self.functions.deinit();
        self.alloc.destroy(self);
    }

    pub fn getFunctions(self: Self) FunctionASTListType.SliceType {
        return self.functions.slice;
    }
};

pub const ASTDumper = struct {
    cur_indent: u32,
    buf: []u8,
    allocator: std.mem.Allocator,

    const Self = @This();
    const print = std.debug.print;

    pub fn init(allocator: std.mem.Allocator, buf_size: usize) !Self {
        return .{
            .cur_indent = 0,
            .buf = try allocator.alloc(u8, buf_size),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.buf);
    }

    pub fn dump(self: *Self, node: *const ModuleAST) !void {
        self.incIndentLv();
        defer self.decIndentLv();

        self.indent();
        print("Module:\n", .{});
        for (node.getFunctions()) |func| {
            try self.dumpFunction(func);
        }
    }

    fn dumpFunction(self: *Self, node: *const FunctionAST) !void {
        self.incIndentLv();
        defer self.decIndentLv();

        self.indent();
        print("Function \n", .{});

        try self.dumpPrototype(node.getProto());
        self.dumpExprList(node.getBody());
    }

    fn dumpPrototype(self: *Self, node: *const PrototypeAST) !void {
        self.incIndentLv();
        defer self.decIndentLv();

        self.indent();
        print("Proto '{s}' {s}\n", .{ node.getName(), self.formatLoc(&node.loc()) });

        self.indent();
        print("Params: [", .{});
        const arg_t = PrototypeAST.ArgsType.ElementType;
        const expr = struct {
            fn func(arg: arg_t, _: ?[]u8) []const u8 {
                return arg.getName();
            }
        }.func;
        printSlice(arg_t, node.getArgs(), expr, null);
        print("]\n", .{});
    }

    fn dumpExprList(self: *Self, nodes: ExprASTList) void {
        self.incIndentLv();
        defer self.decIndentLv();
        self.indent();

        print("Block {{\n", .{});
        for (nodes) |node| {
            self.dumpExpr(node);
        }
        self.indent();
        print("}} // Block\n", .{});
    }

    fn dumpExpr(self: *Self, node: ExprAST) void {
        switch (node) {
            .VarDecl => |v| self.dumpVarDecl(v.*),
            .Return => |v| self.dumpReturn(v.*),
            .Num => |v| self.dumpNumber(v.*),
            .Literal => |v| self.dumpLiteral(v.*),
            .Var => |v| self.dumpVariable(v.*),
            .BinOp => |v| self.dumpBinOp(v.*),
            .Call => |v| self.dumpCall(v.*),
            .Print => |v| self.dumpPrint(v.*),
        }
    }

    fn dumpVarDecl(self: *Self, node: VarDeclExprAST) void {
        self.incIndentLv();
        defer self.decIndentLv();
        self.indent();

        print("VarDecl {s}", .{node.getName()});
        self.dumpVarType(node.getType());
        print(" {s}\n", .{self.formatLoc(&node.loc())});
        if (node.getInitVal()) |v| {
            self.dumpExpr(v);
        }
    }

    fn dumpReturn(self: *Self, node: ReturnExprAST) void {
        self.incIndentLv();
        defer self.decIndentLv();
        self.indent();

        print("Return\n", .{});
        if (node.getExpr()) |v| {
            self.dumpExpr(v);
        } else {
            self.incIndentLv();
            defer self.decIndentLv();
            self.indent();
            print("(void)\n", .{});
        }
    }

    fn dumpNumber(self: *Self, node: NumberExprAST) void {
        self.incIndentLv();
        defer self.decIndentLv();
        self.indent();

        var ff = FloatFormatter.init(6, 2, true);
        print("{s} {s}\n", .{ ff.format(node.getValue()), self.formatLoc(&node.loc()) });
    }

    fn dumpLiteral(self: *Self, node: LiteralExprAST) void {
        self.incIndentLv();
        defer self.decIndentLv();
        self.indent();

        print("Literal: ", .{});
        // NOTE: It's safe to duplicate `node` as a mutable here. The actual
        // instance is referenced by a pointer in `node.base`. Here we just
        // need to duplicate it as a mutable to call `tagged()` for conversion.
        var _node = node;
        printLitHelper(_node.tagged(), self.buf);
        print(" {s}\n", .{self.formatLoc(&node.loc())});
    }

    fn printLitHelper(litOrNum: ExprAST, ctx_buf: []u8) void {
        if (litOrNum == .Num) {
            var ff = FloatFormatter.init(6, 2, true);
            print("{s}", .{ff.format(litOrNum.Num.getValue())});
            return;
        }
        const literal = litOrNum.Literal;

        // Print shape
        print("<", .{});
        const type_t = VarType.ElementType;
        const dimsExpr = struct {
            pub fn func(arg: type_t, buf: ?[]u8) []const u8 {
                return std.fmt.bufPrint(buf.?, "{d}", .{arg}) catch {
                    std.debug.print("Failed to format value in VarType\n", .{});
                    return "";
                };
            }
        }.func;

        printSlice(type_t, literal.getDims().shape, dimsExpr, ctx_buf);
        print(">", .{});

        // Print content (number/literal)
        print("[ ", .{});
        const vals_t = ExprASTListType.ElementType;
        const valsExpr = struct {
            fn func(arg: vals_t, buf: ?[]u8) []const u8 {
                return switch (arg) {
                    .Num => |v| {
                        var ff = FloatFormatter.init(6, 2, true);
                        return ff.format(v.getValue());
                    },
                    .Literal => {
                        printLitHelper(arg, buf.?);
                        return "";
                    },
                    else => {
                        print(
                            "[ERROR] invalid type in LiteralExprAST.values: {}\n",
                            .{arg.getKind()},
                        );
                        return "";
                    },
                };
            }
        }.func;
        printSlice(vals_t, literal.getValues(), valsExpr, ctx_buf);
        print("]", .{});
    }

    fn dumpVariable(self: *Self, node: VariableExprAST) void {
        self.incIndentLv();
        defer self.decIndentLv();
        self.indent();

        print("var: {s} {s}\n", .{ node.getName(), self.formatLoc(&node.loc()) });
    }

    fn dumpBinOp(self: *Self, node: BinaryExprAST) void {
        self.incIndentLv();
        defer self.decIndentLv();
        self.indent();

        print("BinOp: {c} {s}\n", .{ node.getOp(), self.formatLoc(&node.loc()) });

        self.dumpExpr(node.getLHS());
        self.dumpExpr(node.getRHS());
    }

    fn dumpCall(self: *Self, node: CallExprAST) void {
        self.incIndentLv();
        defer self.decIndentLv();
        self.indent();

        print("Call '{s}' [ {s}\n", .{ node.getCallee(), self.formatLoc(&node.loc()) });
        for (node.getArgs()) |arg| {
            self.dumpExpr(arg);
        }
        self.indent();
        print("]\n", .{});
    }

    fn dumpPrint(self: *Self, node: PrintExprAST) void {
        self.incIndentLv();
        defer self.decIndentLv();
        self.indent();

        print("Print [ {s}\n", .{self.formatLoc(&node.loc())});
        self.dumpExpr(node.getArg());
        self.indent();
        print("]\n", .{});
    }

    fn dumpVarType(self: *Self, var_type: VarType) void {
        print("<", .{});

        const arg_t = VarType.ElementType;
        const expr = struct {
            fn func(arg: arg_t, buf: ?[]u8) []const u8 {
                return std.fmt.bufPrint(buf.?, "{d}", .{arg}) catch {
                    std.debug.print("Failed to format value in VarType\n", .{});
                    return "";
                };
            }
        }.func;

        printSlice(arg_t, var_type.shape, expr, self.buf);
        print(">", .{});
    }

    fn indent(self: Self) void {
        for (0..self.cur_indent) |_| {
            std.debug.print("  ", .{});
        }
    }

    fn incIndentLv(self: *Self) void {
        self.cur_indent += 1;
    }

    fn decIndentLv(self: *Self) void {
        std.debug.assert(self.cur_indent >= 1);
        self.cur_indent -= 1;
    }

    fn formatLoc(self: *Self, loc: *const lexer.Location) []const u8 {
        return std.fmt.bufPrint(self.buf, "@{s}:{d}:{d}", .{
            loc.file,
            loc.line,
            loc.col,
        }) catch {
            std.debug.print("Insufficient buffer size to print location information", .{});
            return "";
        };
    }

    /// exprFn: A function to format the field that user want to access in `data`
    /// ctx_buf: A buffer to pass into `exprFn` provided by context
    fn printSlice(
        comptime T: type,
        data: []T,
        exprFn: *const fn (arg: T, buf: ?[]u8) []const u8,
        ctx_buf: ?[]u8,
    ) void {
        if (data.len == 0) return;

        const i_last = data.len - 1;
        for (0..i_last) |i| {
            print("{s}, ", .{exprFn(data[i], ctx_buf)});
        }
        print("{s}", .{exprFn(data[i_last], ctx_buf)});
    }
};

const FloatFormatter = struct {
    buf: [BUF_SIZE]u8,
    precision: usize,
    exp_digits: ?usize,
    always_signed: bool,

    const Self = @This();
    const BUF_SIZE = 128;

    pub fn init(
        comptime precision: usize,
        comptime exp_digits: ?usize,
        comptime always_signed: bool,
    ) Self {
        comptime std.debug.assert(BUF_SIZE > std.fmt.format_float.min_buffer_size);
        return .{
            .buf = [_]u8{0} ** BUF_SIZE,
            .precision = precision,
            .exp_digits = exp_digits,
            .always_signed = always_signed,
        };
    }

    pub fn format(self: *Self, v: anytype) []const u8 {
        const ret = std.fmt.format_float.formatFloat(
            &self.buf,
            v,
            .{ .mode = .scientific, .precision = self.precision },
        ) catch |err| {
            std.debug.print("Failed to format value, error: {}\n", .{err});
            return "";
        };
        const ori_len = ret.len; // including null sentinel

        const idx_e = std.mem.indexOf(u8, ret, "e") orelse unreachable;
        const exponent = ret[idx_e + 1 .. ori_len];
        const is_neg = ret[idx_e + 1] == '-';
        const vals = exponent[@as(usize, @intFromBool(is_neg))..exponent.len];

        // Reserved digits for exponent without sign
        const nv = blk: {
            if (self.exp_digits) |d| {
                break :blk if (d > vals.len) d else vals.len;
            } else {
                break :blk vals.len;
            }
        };
        // Reserved digit for sign
        const ns = @as(usize, @intFromBool(self.always_signed or is_neg));
        const nr = nv + ns;

        // Back-shift to reserve space for paddings
        if (nr > exponent.len) {
            const d = nr - exponent.len;
            const end = ori_len + d;

            for (0..vals.len) |i| {
                const isrc = ori_len - i - 1;
                const idst = end - i - 1;
                self.buf[idst] = self.buf[isrc];
            }

            // Padding 0s from the digit reserved for sign to the first valid
            // digit copied from original exponent.
            @memset(self.buf[idx_e + ns + 1 .. end - vals.len], '0');

            if (self.always_signed) {
                self.buf[idx_e + ns] = if (is_neg) '-' else '+';
            }
            return self.buf[0..end];
        }
        return ret;
    }
};

fn newLoc(file: []const u8, line: u32, col: u32) lexer.Location {
    return .{ .file = file, .line = line, .col = col };
}

test "number expr" {
    const loc = newLoc("foobar.toy", 10, 20);
    const val: f64 = 42;
    var num_expr = try NumberExprAST.init(test_alloc, loc, val);

    var _expr = num_expr.tagged();

    // Make sure it's fine to call `deinit()` from up-casted instance.
    defer _expr.deinit();

    const _loc = _expr.loc();
    try test_expect(_expr.getKind() == .Num);
    try test_expect(std.mem.eql(u8, _loc.file, "foobar.toy"));
    try test_expect(_loc.line == 10);
    try test_expect(_loc.col == 20);

    const _num_expr = _expr.Num;
    try test_expect(isClose(f64, _num_expr.getValue(), 42));

    // TEST: Make sure we can update value after casting from AnyExprAST back
    // to NumberExprAST without being applied constness.
    _num_expr.val = 44;
    try test_expect(isClose(f64, _num_expr.getValue(), 44));
}

test "literal expr" {
    const fname = "foobar.toy";

    var values_al = ExprASTListType.ArrayList.init(test_alloc);
    var num01 = try NumberExprAST.init(
        test_alloc,
        newLoc(fname, 10, 20),
        42,
    );
    var num02 = try NumberExprAST.init(
        test_alloc,
        newLoc(fname, 10, 23),
        43,
    );
    var num03 = try NumberExprAST.init(
        test_alloc,
        newLoc(fname, 10, 26),
        44,
    );
    try values_al.append(num01.tagged());
    try values_al.append(num02.tagged());
    try values_al.append(num03.tagged());

    var shape_al = VarType.ArrayList.init(test_alloc);
    try shape_al.append(3);

    const values = try ExprASTListType.fromArrayList(&values_al);
    const dims = try VarType.fromArrayList(&shape_al);

    var lit_expr = try LiteralExprAST.init(
        test_alloc,
        newLoc(fname, 10, 20),
        values,
        dims,
    );

    var expr = lit_expr.tagged();

    // Make sure it's fine to call `deinit()` from up-casted instance.
    defer expr.deinit();

    const _loc = expr.loc();
    try test_expect(expr.getKind() == .Literal);
    try test_expect(std.mem.eql(u8, _loc.file, "foobar.toy"));
    try test_expect(_loc.line == 10);
    try test_expect(_loc.col == 20);

    const _lit = expr.Literal;
    const _values = _lit.getValues();
    try test_expect(_values[2].getKind() == .Num);

    // Check whether there is any problem resulted by constness
    var _var_num: *NumberExprAST = _values[2].Num;
    try test_expect(isClose(f64, _var_num.getValue(), 44));

    _var_num.val = 45;
    try test_expect(utils.isClose(f64, _var_num.getValue(), 45));

    const _const_num: *const NumberExprAST = _values[1].Num;
    try test_expect(isClose(f64, _const_num.getValue(), 43));

    const _dims = _lit.getDims();
    try test_expect(_dims.shape.len == 1);
    try test_expect(_dims.shape[0] == 3);
}

test "variable expr" {
    const loc = newLoc("foobar.toy", 10, 20);
    var var_expr = try VariableExprAST.init(test_alloc, loc, "std");

    var expr = var_expr.tagged();
    defer expr.deinit();

    const _loc = expr.loc();
    try test_expect(expr.getKind() == .Var);
    try test_expect(std.mem.eql(u8, _loc.file, "foobar.toy"));
    try test_expect(_loc.line == 10);
    try test_expect(_loc.col == 20);

    try test_expect(std.mem.eql(u8, expr.Var.getName(), "std"));
}

// decl ::= var idenitifier [ type ] = expr
test "declaration expr" {
    const fname = "foobar.toy";
    const ident = "answer";

    var type_al = VarType.ArrayList.init(test_alloc);
    try type_al.append(1);

    const value = 42;
    var num_expr = try NumberExprAST.init(
        test_alloc,
        newLoc(fname, 1, 22),
        value,
    );

    // Test with a VarDeclExprAST with non-null init_val
    var decl_expr_1 = try VarDeclExprAST.init(
        test_alloc,
        newLoc(fname, 1, 1),
        ident,
        try VarType.fromArrayList(&type_al),
        num_expr.tagged(),
    );

    var expr_1 = decl_expr_1.tagged();
    defer expr_1.deinit();

    try test_expect(expr_1.getKind() == .VarDecl);
    try test_expect(expr_1.loc().col == 1);

    const _decl_expr_1 = expr_1.VarDecl;
    try test_expect(std.mem.eql(u8, _decl_expr_1.getName(), ident));

    const _type_1 = _decl_expr_1.getType();
    try test_expect(_type_1.shape.len == 1);
    try test_expect(_type_1.shape[0] == 1);

    const _init_val_1 = _decl_expr_1.getInitVal();
    try test_expect(_init_val_1 != null);
    try test_expect(_init_val_1.?.getKind() == .Num);

    const _num_expr_1 = _init_val_1.?.Num;
    try test_expect(isClose(f64, _num_expr_1.getValue(), value));
    try test_expect(_num_expr_1.loc().col == 22);

    // Test with a VarDeclExprAST with null init_val
    var decl_expr_2 = try VarDeclExprAST.init(
        test_alloc,
        newLoc(fname, 2, 1),
        ident,
        try VarType.fromArrayList(&type_al),
        null,
    );

    var expr_2 = decl_expr_2.tagged();
    defer expr_2.deinit();

    const _decl_expr_2 = expr_2.VarDecl;
    const _init_val_2 = _decl_expr_2.getInitVal();
    try test_expect(_init_val_2 == null);
}

test "return expr" {
    const fname = "foobar.toy";

    // Expected:
    //     return 42;
    var num_expr = try NumberExprAST.init(
        test_alloc,
        newLoc(fname, 2, 12),
        42,
    );
    var ret_expr_1 = try ReturnExprAST.init(
        test_alloc,
        newLoc(fname, 2, 5),
        num_expr.tagged(),
    );

    // Expected:
    //     return;
    var ret_expr_2 = try ReturnExprAST.init(
        test_alloc,
        newLoc(fname, 6, 5),
        null,
    );

    var expr_1 = ret_expr_1.tagged();
    var expr_2 = ret_expr_2.tagged();

    // NOTE: No dynamic allocated memory is used in this kind of expr, but it's
    // still fine to call `deinit()`.
    defer expr_1.deinit();
    defer expr_2.deinit();

    const _ret_expr_1 = expr_1.Return;
    const _ret_expr_2 = expr_2.Return;

    try test_expect(_ret_expr_1.getExpr() != null);
    try test_expect(_ret_expr_2.getExpr() == null);

    try test_expect(_ret_expr_1.getExpr().?.getKind() == .Num);

    const _num_expr = _ret_expr_1.getExpr().?.Num;
    try test_expect(isClose(f64, _num_expr.getValue(), 42));
}

test "return expr with call" {
    // Expected:
    //     return mul(a, add(b, c));
    const fname = "foobar.toy";

    const makeVar = struct {
        fn func(f: []const u8, l: u32, c: u32, n: []const u8) !ExprAST {
            const v = try VariableExprAST.init(test_alloc, newLoc(f, l, c), n);
            return v.tagged();
        }
    }.func;

    const makeCall = struct {
        fn func(f: []const u8, l: u32, c: u32, n: []const u8, v1: ExprAST, v2: ExprAST) !ExprAST {
            var args_al = ExprASTListType.ArrayList.init(test_alloc);
            try args_al.append(v1);
            try args_al.append(v2);
            const args = try ExprASTListType.fromArrayList(&args_al);
            const call = try CallExprAST.init(test_alloc, newLoc(f, l, c), n, args);
            return call.tagged();
        }
    }.func;

    const vars = [_]ExprAST{
        try makeVar(fname, 2, 16, "a"),
        try makeVar(fname, 2, 23, "b"),
        try makeVar(fname, 2, 26, "c"),
    };

    const call_1 = try makeCall(fname, 2, 19, "add", vars[1], vars[2]);
    const call_2 = try makeCall(fname, 2, 12, "mul", vars[0], call_1);
    const ret_expr = try ReturnExprAST.init(
        test_alloc,
        newLoc(fname, 1, 5),
        call_2,
    );
    var expr = ret_expr.tagged();
    defer expr.deinit();

    try test_expect(expr.getKind() == .Return);

    const _ret_expr = expr.Return;
    const _expr = _ret_expr.getExpr();
    try test_expect(_expr.?.getKind() == .Call);

    const _args_2 = _expr.?.Call.getArgs();
    try test_expect(_args_2.len == 2);
    try test_expect(_args_2[0].getKind() == .Var);
    try test_expect(_args_2[1].getKind() == .Call);
}

test "binary expr" {
    // 9 + 14
    const fname = "foobar.toy";
    const op = '+';

    var lhs = try NumberExprAST.init(test_alloc, newLoc(fname, 2, 9), 11);
    var rhs = try NumberExprAST.init(test_alloc, newLoc(fname, 2, 14), 13);
    var bin_expr = try BinaryExprAST.init(
        test_alloc,
        newLoc(fname, 2, 5),
        op,
        lhs.tagged(),
        rhs.tagged(),
    );

    var expr = bin_expr.tagged();
    defer expr.deinit();

    try test_expect(expr.getKind() == .BinOp);

    const _bin_expr = expr.BinOp;
    const _lhs = _bin_expr.getLHS();
    const _rhs = _bin_expr.getRHS();
    try test_expect(_bin_expr.getOp() == op);
    try test_expect(_lhs.getKind() == .Num);
    try test_expect(_rhs.getKind() == .Num);

    const _num_expr_l = _lhs.Num;
    const _num_expr_r = _rhs.Num;
    try test_expect(isClose(f64, _num_expr_l.getValue(), 11));
    try test_expect(isClose(f64, _num_expr_r.getValue(), 13));
}

test "binary expr nested" {
    // Expected:
    // 3 * (1 + 2) - 5
    const fname = "foobar.toy";

    const makeNum = struct {
        fn func(f: []const u8, l: u32, c: u32, v: f64) !ExprAST {
            const num = try NumberExprAST.init(test_alloc, newLoc(f, l, c), v);
            return num.tagged();
        }
    }.func;

    const makeBin = struct {
        fn func(f: []const u8, l: u32, c: u32, o: u8, left: ExprAST, right: ExprAST) !ExprAST {
            const bin = try BinaryExprAST.init(
                test_alloc,
                newLoc(f, l, c),
                o,
                left,
                right,
            );
            return bin.tagged();
        }
    }.func;

    const nums = [4]ExprAST{
        try makeNum(fname, 1, 1, 3),
        try makeNum(fname, 1, 6, 1),
        try makeNum(fname, 1, 10, 2),
        try makeNum(fname, 1, 15, 5),
    };

    const expr_1 = try makeBin(fname, 1, 5, '+', nums[1], nums[2]);
    const expr_2 = try makeBin(fname, 1, 1, '*', nums[0], expr_1);
    const expr_3 = try makeBin(fname, 1, 1, '-', expr_2, nums[3]);

    defer expr_3.deinit();

    try test_expect(expr_3.getKind() == .BinOp);

    const _bin_expr_3 = expr_3.BinOp;
    const _lhs_3 = _bin_expr_3.getLHS();
    const _rhs_3 = _bin_expr_3.getRHS();
    try test_expect(_bin_expr_3.getOp() == '-');
    try test_expect(_lhs_3.getKind() == .BinOp);
    try test_expect(_rhs_3.getKind() == .Num);
    try test_expect(isClose(f64, _rhs_3.Num.getValue(), 5));

    const _bin_expr_2 = _lhs_3.BinOp;
    const _lhs_2 = _bin_expr_2.getLHS();
    const _rhs_2 = _bin_expr_2.getRHS();
    try test_expect(_bin_expr_2.getOp() == '*');
    try test_expect(_lhs_2.getKind() == .Num);
    try test_expect(isClose(f64, _lhs_2.Num.getValue(), 3));
    try test_expect(_rhs_2.getKind() == .BinOp);

    const _bin_expr_1 = _rhs_2.BinOp;
    const _lhs_1 = _bin_expr_1.getLHS();
    const _rhs_1 = _bin_expr_1.getRHS();
    try test_expect(_bin_expr_1.getOp() == '+');
    try test_expect(_lhs_1.getKind() == .Num);
    try test_expect(isClose(f64, _lhs_1.Num.getValue(), 1));
    try test_expect(_rhs_1.getKind() == .Num);
    try test_expect(isClose(f64, _rhs_1.Num.getValue(), 2));
}

test "call expr" {
    // Expected:
    //     mul(11, 13);

    const fname = "foobar.toy";
    const arg_1 = try NumberExprAST.init(test_alloc, newLoc(fname, 2, 9), 11);
    const arg_2 = try NumberExprAST.init(test_alloc, newLoc(fname, 2, 13), 13);
    var args_al = ExprASTListType.ArrayList.init(test_alloc);
    try args_al.append(arg_1.tagged());
    try args_al.append(arg_2.tagged());

    const args = try ExprASTListType.fromArrayList(&args_al);
    const callee = "mul";
    const call_expr = try CallExprAST.init(
        test_alloc,
        newLoc(fname, 2, 5),
        callee,
        args,
    );

    var expr = call_expr.tagged();
    defer expr.deinit();

    try test_expect(expr.getKind() == .Call);

    const _call_expr = expr.Call;
    try test_expect(std.mem.eql(u8, _call_expr.getCallee(), callee));

    const _args = _call_expr.getArgs();
    try test_expect(_args.len == 2);
    try test_expect(_args[0].getKind() == .Num);
    try test_expect(_args[1].getKind() == .Num);

    const _num_expr_1 = _args[0].Num;
    const _num_expr_2 = _args[1].Num;
    try test_expect(isClose(f64, _num_expr_1.getValue(), 11));
    try test_expect(isClose(f64, _num_expr_2.getValue(), 13));
}

test "call expr nested" {
    // Expected:
    // add(1, mul(2, 3))
    const fname = "foobar.toy";

    // We are intended to create exprs inside a function to check whether the
    // data passing is correct.
    const makeNum = struct {
        fn func(f: []const u8, l: u32, c: u32, v: f64) !ExprAST {
            const num = try NumberExprAST.init(test_alloc, newLoc(f, l, c), v);
            return num.tagged();
        }
    }.func;

    const makeCall_1 = struct {
        fn func(f: []const u8, l: u32, c: u32, name: []const u8, v1: f64, v2: f64) !ExprAST {
            const arg_1 = try makeNum(f, l, c, v1);
            const arg_2 = try makeNum(f, l, c, v2);
            var args_al = ExprASTListType.ArrayList.init(test_alloc);
            try args_al.append(arg_1);
            try args_al.append(arg_2);
            const args = try ExprASTListType.fromArrayList(&args_al);
            const call = try CallExprAST.init(
                test_alloc,
                newLoc(f, l, c),
                name,
                args,
            );
            return call.tagged();
        }
    }.func;

    const makeCall_2 = struct {
        fn func(f: []const u8, l: u32, c: u32, name: []const u8, v: f64, e: ExprAST) !ExprAST {
            const arg_1 = try makeNum(f, l, c, v);
            var args_al = ExprASTListType.ArrayList.init(test_alloc);
            try args_al.append(arg_1);
            try args_al.append(e);
            const args = try ExprASTListType.fromArrayList(&args_al);
            const call = try CallExprAST.init(
                test_alloc,
                newLoc(f, l, c),
                name,
                args,
            );
            return call.tagged();
        }
    }.func;

    const call_1 = try makeCall_1(fname, 1, 1, "mul", 2, 3);
    var call_2 = try makeCall_2(fname, 1, 1, "add", 1, call_1);

    defer call_2.deinit();

    try test_expect(call_2.getKind() == .Call);
}

test "print expr" {
    // Expected:
    // print(11);

    const fname = "foobar.toy";
    const arg = try NumberExprAST.init(test_alloc, newLoc(fname, 2, 11), 11);
    const print_expr = try PrintExprAST.init(
        test_alloc,
        newLoc(fname, 2, 5),
        arg.tagged(),
    );

    var expr = print_expr.tagged();
    defer expr.deinit();

    try test_expect(expr.getKind() == .Print);

    const _print_expr = expr.Print;
    const _arg = _print_expr.getArg();
    try test_expect(_arg.getKind() == .Num);

    const _num_expr = _arg.Num;
    try test_expect(isClose(f64, _num_expr.getValue(), 11));
}

test "prototype ast" {
    // Expected:
    // def mul(a, b)

    const fname = "foobar.toy";
    const proto_name = "mul";

    var var_exprs_al = PrototypeAST.ArgsType.ArrayList.init(test_alloc);
    const var_expr_1 = try VariableExprAST.init(test_alloc, newLoc(fname, 1, 9), "a");
    const var_expr_2 = try VariableExprAST.init(test_alloc, newLoc(fname, 1, 12), "b");
    try var_exprs_al.append(var_expr_1);
    try var_exprs_al.append(var_expr_2);

    const var_exprs = try PrototypeAST.ArgsType.fromArrayList(&var_exprs_al);
    var proto = try PrototypeAST.init(
        test_alloc,
        newLoc(fname, 1, 5),
        proto_name,
        var_exprs,
    );
    defer proto.deinit();

    try test_expect(std.mem.eql(u8, proto.getName(), proto_name));

    const args = proto.getArgs();
    try test_expect(args.len == 2);
    try test_expect(std.mem.eql(u8, args[0].getName(), "a"));
    try test_expect(std.mem.eql(u8, args[1].getName(), "b"));
}

test "function ast" {
    // Expected:
    // def mul(a, b) {
    //     return a * b;
    // }

    const fname = "foobar.toy";
    const proto_name = "mul";

    var args_al = PrototypeAST.ArgsType.ArrayList.init(test_alloc);
    const arg_1 = try VariableExprAST.init(test_alloc, newLoc(fname, 1, 10), "a");
    const arg_2 = try VariableExprAST.init(test_alloc, newLoc(fname, 1, 13), "b");
    try args_al.append(arg_1);
    try args_al.append(arg_2);

    const proto_args = try PrototypeAST.ArgsType.fromArrayList(&args_al);
    const proto = try PrototypeAST.init(
        test_alloc,
        newLoc(fname, 1, 5),
        proto_name,
        proto_args,
    );

    var lhs = try VariableExprAST.init(test_alloc, newLoc(fname, 2, 12), "a");
    var rhs = try VariableExprAST.init(test_alloc, newLoc(fname, 2, 16), "b");
    var binary_expr = try BinaryExprAST.init(
        test_alloc,
        newLoc(fname, 2, 14),
        '*',
        lhs.tagged(),
        rhs.tagged(),
    );
    var return_expr = try ReturnExprAST.init(test_alloc, newLoc(fname, 2, 5), binary_expr.tagged());

    var body_al = FunctionAST.BodyType.ArrayList.init(test_alloc);
    try body_al.append(return_expr.tagged());

    const body = try FunctionAST.BodyType.fromArrayList(&body_al);

    var func = try FunctionAST.init(test_alloc, proto, body);
    defer func.deinit();

    try test_expect(std.mem.eql(u8, func.getProto().getName(), proto_name));
    try test_expect(func.getProto().getArgs().len == 2);

    const _body = func.getBody();
    try test_expect(_body.len == 1);
    try test_expect(_body[0].getKind() == .Return);

    const _return_expr = _body[0].Return;
    try test_expect(_return_expr.getExpr() != null);
    try test_expect(_return_expr.getExpr().?.getKind() == .BinOp);

    const _binary_expr = _return_expr.getExpr().?.BinOp;
    try test_expect(_binary_expr.getOp() == '*');
    try test_expect(_binary_expr.getLHS().getKind() == .Var);
    try test_expect(_binary_expr.getRHS().getKind() == .Var);

    const _lhs = _binary_expr.getLHS().Var;
    const _rhs = _binary_expr.getRHS().Var;
    try test_expect(std.mem.eql(u8, _lhs.getName(), "a"));
    try test_expect(std.mem.eql(u8, _rhs.getName(), "b"));
}

test "module ast" {
    // def mul(a, b) {
    //     return a * b;
    // }
    //
    // def add(a, b) {
    //     return a + b;
    // }
    const fname = "foobar.toy";

    // Bulding AST for function `mul(a, b)`:
    var args_al_1 = PrototypeAST.ArgsType.ArrayList.init(test_alloc);
    const arg_1_1 = try VariableExprAST.init(test_alloc, newLoc(fname, 1, 9), "a");
    const arg_1_2 = try VariableExprAST.init(test_alloc, newLoc(fname, 1, 12), "b");
    try args_al_1.append(arg_1_1);
    try args_al_1.append(arg_1_2);

    const args_1 = try PrototypeAST.ArgsType.fromArrayList(&args_al_1);
    const proto_1 = try PrototypeAST.init(
        test_alloc,
        newLoc(fname, 1, 5),
        "mul",
        args_1,
    );

    var lhs_1 = try VariableExprAST.init(test_alloc, newLoc(fname, 2, 12), "a");
    var rhs_1 = try VariableExprAST.init(test_alloc, newLoc(fname, 2, 16), "b");
    var binary_expr_1 = try BinaryExprAST.init(
        test_alloc,
        newLoc(fname, 2, 5),
        '*',
        lhs_1.tagged(),
        rhs_1.tagged(),
    );
    var return_expr_1 = try ReturnExprAST.init(
        test_alloc,
        newLoc(fname, 2, 5),
        binary_expr_1.tagged(),
    );

    var body_al_1 = FunctionAST.BodyType.ArrayList.init(test_alloc);
    try body_al_1.append(return_expr_1.tagged());

    const body_1 = try FunctionAST.BodyType.fromArrayList(&body_al_1);

    const func_1 = try FunctionAST.init(test_alloc, proto_1, body_1);

    // Bulding AST for function `add(a, b)`:
    var args_al_2 = PrototypeAST.ArgsType.ArrayList.init(test_alloc);
    const arg_2_1 = try VariableExprAST.init(test_alloc, newLoc(fname, 5, 9), "a");
    const arg_2_2 = try VariableExprAST.init(test_alloc, newLoc(fname, 5, 12), "b");
    try args_al_2.append(arg_2_1);
    try args_al_2.append(arg_2_2);

    const args_2 = try PrototypeAST.ArgsType.fromArrayList(&args_al_2);
    const proto_2 = try PrototypeAST.init(test_alloc, newLoc(fname, 5, 5), "add", args_2);

    var lhs_2 = try VariableExprAST.init(test_alloc, newLoc(fname, 6, 12), "a");
    var rhs_2 = try VariableExprAST.init(test_alloc, newLoc(fname, 6, 16), "b");
    var binary_expr_2 = try BinaryExprAST.init(
        test_alloc,
        newLoc(fname, 6, 5),
        '+',
        lhs_2.tagged(),
        rhs_2.tagged(),
    );
    var return_expr_2 = try ReturnExprAST.init(
        test_alloc,
        newLoc(fname, 6, 5),
        binary_expr_2.tagged(),
    );

    var body_al_2 = FunctionAST.BodyType.ArrayList.init(test_alloc);
    try body_al_2.append(return_expr_2.tagged());

    const body_2 = try FunctionAST.BodyType.fromArrayList(&body_al_2);

    const func_2 = try FunctionAST.init(test_alloc, proto_2, body_2);

    // Build AST for module:
    var functions_al = FunctionASTListType.ArrayList.init(test_alloc);
    try functions_al.append(func_1);
    try functions_al.append(func_2);

    const functions = try FunctionASTListType.fromArrayList(&functions_al);

    var module = try ModuleAST.init(test_alloc, functions);
    defer module.deinit();

    // Test, test, test...
    const _functions = module.getFunctions();
    try test_expect(_functions.len == 2);

    const _func_1 = _functions[0];
    try test_expect(std.mem.eql(u8, _func_1.getProto().getName(), "mul"));
    try test_expect(_func_1.getProto().getArgs().len == 2);

    const _func_2 = _functions[1];
    try test_expect(std.mem.eql(u8, _func_2.getProto().getName(), "add"));
    try test_expect(_func_2.getProto().getArgs().len == 2);
}
