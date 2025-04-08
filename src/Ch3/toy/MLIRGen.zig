ctx: c.MlirContext,
module: c.MlirModule,
op_builder: c.MlirOpBuilder,
symbol_table: symbol_table_t,
allocator: Allocator,

const symbol_table_t = ScopedHashMap(
    []const u8,
    c.MlirValue,
    std.hash_map.StringContext,
    std.hash_map.default_max_load_percentage,
);

const ArrayListF64 = std.ArrayList(f64);
const MLIRGenError = Allocator.Error || error{
    Module,
    Function,
    Var,
    VarDecl,
    BinOp,
    Literal,
    Call,
    Expr,
    Redeclaration,
};

pub fn init(ctx: c.MlirContext, allocator: Allocator) Self {
    const op_builder = c.mlirOpBuilderCreate(ctx);

    return Self{
        .ctx = ctx,
        .module = c.MlirModule{},
        .op_builder = op_builder,
        .symbol_table = symbol_table_t.init(allocator),
        .allocator = allocator,
    };
}

pub fn deinit(self: *Self) void {
    self.symbol_table.deinit();
    c.mlirModuleDestroy(self.module);
}

pub fn fromModule(self: *Self, module_ast: *ast.ModuleAST) MLIRGenError!c.MlirModule {
    // Delete existing module and create a new & empty one
    if (!c.mlirModuleIsNull(self.module)) {
        c.mlirModuleDestroy(self.module);
    }
    const unknown_loc = c.mlirOpBuilderGetUnknownLoc(self.op_builder);
    self.module = c.mlirModuleCreateEmpty(unknown_loc);

    for (module_ast.getFunctions()) |f| {
        _ = try self.fromFunc(f);
    }

    // Verify the module after it's constructed
    const module_op = c.mlirModuleGetOperation(self.module);
    if (c.mlirLogicalResultIsFailure(c.mlirExtVerify(module_op))) {
        c.mlirExtOperationEmitError(module_op, "module verification error");
        return MLIRGenError.Module;
    }
    return self.module;
}

fn declare(self: *Self, name: []const u8, value: c.MlirValue) MLIRGenError!void {
    if (self.symbol_table.contains(name)) {
        return MLIRGenError.Redeclaration;
    }
    try self.symbol_table.put(name, value);
}

fn fromProto(self: *Self, proto_ast: *ast.PrototypeAST) MLIRGenError!c.MlirToyFuncOp {
    const loc = locToMlirLoc(self.ctx, proto_ast.loc());
    const name = z2strref(proto_ast.getName());
    const args = proto_ast.getArgs();

    var inputs = try std.ArrayList(c.MlirType).initCapacity(self.allocator, args.len);
    defer inputs.deinit();

    for (0..args.len) |_| {
        // XXX: here we directly unwrap what `getType()` does in the original
        // example, see also:
        // https://github.com/llvm/llvm-project/blob/release/17.x/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp#L422
        const input_t = c.mlirUnrankedTensorTypeGet(c.mlirF64TypeGet(self.ctx));
        inputs.appendAssumeCapacity(input_t);
    }

    // XXX: the return type will be inferred later.
    const results = [_]c.MlirType{undefined} ** 0;
    const func_t = c.mlirFunctionTypeGet(
        self.ctx,
        @intCast(inputs.items.len),
        inputs.items.ptr,
        0,
        &results,
    );

    return c.mlirToyFuncOpCreateFromFunctionType(self.op_builder, loc, name, func_t);
}

fn fromFunc(self: *Self, func_ast: *ast.FunctionAST) MLIRGenError!c.MlirToyFuncOp {
    try self.symbol_table.createScope();
    defer self.symbol_table.destroyScope();

    const block = c.mlirModuleGetBody(self.module);
    c.mlirOpBuilderSetInsertionPointToEnd(self.op_builder, block);
    const func_op = try self.fromProto(func_ast.getProto());

    const op = c.mlirToyFuncOpToMlirOperation(func_op);
    if (c.mlirOperationIsNull(op)) {
        const _proto = func_ast.getProto();
        const _loc = locToMlirLoc(self.ctx, _proto.loc());
        emitError(_loc, "failed to generate `toy::FuncOp` for function '{s}'", .{_proto.getName()});
        return MLIRGenError.Function;
    }

    const first_region = c.mlirOperationGetFirstRegion(op);
    const entry_block = c.mlirRegionGetFirstBlock(first_region);
    const proto_args = func_ast.getProto().getArgs();

    const n_args = c.mlirBlockGetNumArguments(entry_block);
    for (0..@intCast(n_args)) |i| {
        const arg = c.mlirBlockGetArgument(entry_block, @intCast(i));
        self.declare(proto_args[i].getName(), arg) catch |err| {
            const _loc = locToMlirLoc(self.ctx, proto_args[i].loc());
            emitError(_loc, "variable '{s}' is already declared in prototype '{s}'", .{
                proto_args[i].getName(), func_ast.getProto().getName(),
            });
            return err;
        };
    }

    c.mlirOpBuilderSetInsertionPointToStart(self.op_builder, entry_block);

    const func_body = func_ast.getBody();
    self.fromBlock(func_body) catch |err| {
        c.mlirToyFuncOpErase(func_op);
        return err;
    };

    // Implicitly return void if no return statement was emitted.
    // NOTE: in the original example "toy/Ch2/mlir/MLIRGen.cpp", here the
    // author consider to always emit a ReturnOp for the last expression.
    var return_op = c.MlirToyReturnOp{};
    if (!c.mlirExtBlockIsEmpty(entry_block)) {
        const last_op = c.mlirExtBlockGetLastOperation(entry_block);
        return_op = c.mlirToyReturnOpFromMlirOperation(last_op);
    }

    if (c.mlirToyReturnOpIsNull(return_op)) {
        const loc = locToMlirLoc(self.ctx, func_ast.getProto().loc());
        c.mlirToyReturnOpCreate(self.op_builder, loc, 0, &[0]c.MlirValue{});
    } else if (c.mlirToyReturnOpHasOperand(return_op)) {
        // If the `ReturnOp` has an operand, add the result to this function.
        const ori_func_t = c.mlirToyFuncOpGetFunctionType(func_op);
        const n_inputs: usize = @intCast(c.mlirFunctionTypeGetNumInputs(ori_func_t));
        std.debug.assert(n_inputs == n_args);

        var inputs = try std.ArrayList(c.MlirType).initCapacity(self.allocator, n_inputs);
        defer inputs.deinit();

        for (0..n_inputs) |i| {
            const input_t = c.mlirFunctionTypeGetInput(ori_func_t, @intCast(i));
            inputs.appendAssumeCapacity(input_t);
        }

        const result_t = c.mlirUnrankedTensorTypeGet(c.mlirF64TypeGet(self.ctx));
        const results = [1]c.MlirType{result_t};
        const func_t = c.mlirFunctionTypeGet(
            self.ctx,
            @intCast(inputs.items.len),
            inputs.items.ptr,
            @intCast(results.len),
            &results,
        );

        c.mlirToyFuncOpSetType(func_op, func_t);
    }
    return func_op;
}

fn fromBinary(self: *Self, bin: *ast.BinaryExprAST) MLIRGenError!c.MlirValue {
    const lhs = try self.fromExpr(bin.getLHS());
    const rhs = try self.fromExpr(bin.getRHS());
    const loc = locToMlirLoc(self.ctx, bin.loc());

    // NOTE: Currently we only support AddOp and MulOp in this chapter.
    switch (bin.getOp()) {
        '+' => return c.mlirToyAddOpCreate(self.op_builder, loc, lhs, rhs),
        '*' => return c.mlirToyMulOpCreate(self.op_builder, loc, lhs, rhs),
        else => |v| {
            emitError(loc, "invalid binary operator '{c}'", .{v});
            return MLIRGenError.BinOp;
        },
    }
}

fn fromVar(self: *Self, expr: *ast.VariableExprAST) MLIRGenError!c.MlirValue {
    const name = expr.getName();
    if (self.symbol_table.get(name)) |v| {
        return v;
    }

    const loc = locToMlirLoc(self.ctx, expr.loc());
    emitError(loc, "error: unknown variable '{s}'", .{name});
    return MLIRGenError.Var;
}

fn fromReturn(self: *Self, ret: *ast.ReturnExprAST) MLIRGenError!void {
    const loc = locToMlirLoc(self.ctx, ret.loc());

    var operands = [1]c.MlirValue{undefined};
    var n_operands: i64 = 0;
    if (ret.getExpr()) |expr| {
        operands[0] = try self.fromExpr(expr);
        n_operands += 1;
    }

    std.debug.assert(n_operands <= 1);
    c.mlirToyReturnOpCreate(self.op_builder, loc, n_operands, &operands);
}

fn fromLiteral(self: *Self, lit: *ast.LiteralExprAST) MLIRGenError!c.MlirValue {
    const dims = lit.getDims();
    const capacity: usize = blk: {
        var res: i64 = 1;
        for (dims.shape) |v| {
            res *= v;
        }
        break :blk @intCast(res);
    };

    var data = try ArrayListF64.initCapacity(self.allocator, capacity);
    defer data.deinit();
    try collectData(lit.tagged(), &data, true);

    const element_t = c.mlirF64TypeGet(self.ctx);
    const encoding = c.mlirAttributeGetNull();
    const shape_t = c.mlirRankedTensorTypeGet(
        @intCast(dims.shape.len),
        dims.shape.ptr,
        element_t,
        encoding,
    );

    // NOTE: data will be copied and stored in its internal storage by
    // `mlir::DenseElementsAttr::get()`. See `writeBits()` in the link below.
    // https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/IR/BuiltinAttributes.cpp#L969
    const data_attr = c.mlirDenseElementsAttrDoubleGet(
        shape_t,
        @intCast(data.items.len),
        data.items.ptr,
    );

    const loc = locToMlirLoc(self.ctx, lit.loc());
    return c.mlirToyConstantOpCreateFromTensor(self.op_builder, loc, shape_t, data_attr);
}

fn fromCall(self: *Self, call: *ast.CallExprAST) MLIRGenError!c.MlirValue {
    const callee = call.getCallee();
    const loc = locToMlirLoc(self.ctx, call.loc());

    const args = call.getArgs();
    var operands = try std.ArrayList(c.MlirValue).initCapacity(self.allocator, args.len);
    defer operands.deinit();

    for (args) |arg| {
        const result = try self.fromExpr(arg);
        try operands.append(result);
    }

    if (std.mem.eql(u8, callee, "transpose")) {
        if (args.len != 1) {
            emitError(loc, "toy.transpose does not accept multiple arguments", .{});
            return MLIRGenError.Call;
        }
        return c.mlirToyTransposeOpCreate(self.op_builder, loc, operands.items[0]);
    }

    return c.mlirToyGenericCallOpCreate(
        self.op_builder,
        loc,
        z2strref(callee),
        @intCast(operands.items.len),
        operands.items.ptr,
    );
}

fn fromPrint(self: *Self, call: *ast.PrintExprAST) MLIRGenError!void {
    const arg = try self.fromExpr(call.getArg());
    const loc = locToMlirLoc(self.ctx, call.loc());
    c.mlirToyPrintOpCreate(self.op_builder, loc, arg);
}

fn fromNumber(self: *Self, num: *ast.NumberExprAST) c.MlirValue {
    const loc = locToMlirLoc(self.ctx, num.loc());
    return c.mlirToyConstantOpCreateFromDouble(self.op_builder, loc, num.getValue());
}

fn fromExpr(self: *Self, expr: ast.ExprAST) MLIRGenError!c.MlirValue {
    switch (expr) {
        .BinOp => |v| return try self.fromBinary(v),
        .Var => |v| return try self.fromVar(v),
        .Literal => |v| return try self.fromLiteral(v),
        .Call => |v| return try self.fromCall(v),
        .Num => |v| return self.fromNumber(v),
        inline else => |_, tag| {
            const loc = locToMlirLoc(self.ctx, expr.loc());
            const msg = "MLIR codegen encountered an unhandled expr kind '{s}'";
            emitError(loc, msg, .{@tagName(tag)});
            return MLIRGenError.Expr;
        },
    }
}

fn fromVarDecl(self: *Self, var_decl: *ast.VarDeclExprAST) MLIRGenError!c.MlirValue {
    const init_val = var_decl.getInitVal() orelse {
        const loc = locToMlirLoc(self.ctx, var_decl.loc());
        emitError(loc, "missing initializer in variable declaration.", .{});
        return MLIRGenError.VarDecl;
    };

    var value = try self.fromExpr(init_val);

    // We have the initializer value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    if (var_decl.getType().shape.len != 0) {
        const loc = locToMlirLoc(self.ctx, var_decl.loc());
        const dims = var_decl.getType();

        const element_t = c.mlirF64TypeGet(self.ctx);
        const encoding = c.mlirAttributeGetNull();
        const shape_t = c.mlirRankedTensorTypeGet(
            @intCast(dims.shape.len),
            dims.shape.ptr,
            element_t,
            encoding,
        );

        value = c.mlirToyReshapeOpCreate(self.op_builder, loc, shape_t, value);
    }

    self.declare(var_decl.getName(), value) catch |err| {
        const _loc = locToMlirLoc(self.ctx, var_decl.loc());
        emitError(_loc, "variable '{s}' has been declared already", .{var_decl.getName()});
        return err;
    };
    return value;
}

fn fromBlock(self: *Self, block: ast.ExprASTList) MLIRGenError!void {
    try self.symbol_table.createScope();
    defer self.symbol_table.destroyScope();

    for (block) |expr| switch (expr) {
        .VarDecl => |v| {
            _ = try self.fromVarDecl(v);
        },
        .Return => |v| {
            _ = try self.fromReturn(v);
        },
        .Print => |v| {
            try self.fromPrint(v);
        },
        inline else => {
            _ = try self.fromExpr(expr);
        },
    };
}

fn locToMlirLoc(ctx: c.MlirContext, loc: lexer.Location) c.MlirLocation {
    return c.mlirLocationFileLineColGet(ctx, z2strref(loc.file), loc.line, loc.col);
}

// Convert Zig string to LLVM's StringRef.
fn z2strref(src: []const u8) c.MlirStringRef {
    // - mlirStringRefCreateFromCString: requires a null-terminated string
    // - mlirStringRefCreate: requires a string with known length
    return c.mlirStringRefCreate(src.ptr, src.len);
}

fn emitError(loc: c.MlirLocation, comptime fmt: []const u8, args: anytype) void {
    var buf = [_]u8{0} ** 256;
    const msg = std.fmt.bufPrint(&buf, fmt, args) catch @panic("Message length exceeds buffer length.");
    c.mlirEmitError(loc, msg.ptr);
}

fn collectData(
    expr: ast.ExprAST,
    data: *ArrayListF64,
    assume_capacity: bool,
) MLIRGenError!void {
    switch (expr) {
        .Literal => {
            const lit = expr.Literal;
            for (lit.getValues()) |v| {
                try collectData(v, data, assume_capacity);
            }
        },
        .Num => {
            if (assume_capacity) {
                data.appendAssumeCapacity(expr.Num.getValue());
            } else {
                try data.append(expr.Num.getValue());
            }
        },
        else => {
            return MLIRGenError.Literal;
        },
    }
}

const std = @import("std");
const c = @import("c_api.zig").c;
const lexer = @import("lexer.zig");
const ast = @import("ast.zig");
const ScopedHashMap = @import("./scoped_hash_map.zig").ScopedHashMap;

const Self = @This();
const Allocator = std.mem.Allocator;
