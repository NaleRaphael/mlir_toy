ctx: c.MlirContext,
module: c.MlirModule,
op_builder: c.MlirOpBuilder,
symbol_table: symbol_table_t,
function_map: func_map_t,
struct_map: struct_map_t,
allocator: Allocator,

const SymbolTableValue = makePair(c.MlirValue, *ast.VarDeclExprAST);
const StructMapValue = makePair(c.MlirType, *ast.StructAST);
const MLIRStructLiteralData = makePair(c.MlirAttribute, c.MlirType);

const symbol_table_t = ScopedHashMap(
    []const u8,
    SymbolTableValue,
    std.hash_map.StringContext,
    std.hash_map.default_max_load_percentage,
);
const func_map_t = std.StringHashMap(c.MlirToyFuncOp);
const struct_map_t = std.StringHashMap(StructMapValue);

const ArrayListF64 = std.ArrayList(f64);
const MLIRGenError = Allocator.Error || error{
    Type,
    Var,
    VarDecl,
    BinOp,
    Struct,
    StructField,
    StructLiteral,
};

pub fn init(ctx: c.MlirContext, allocator: Allocator) Self {
    const op_builder = c.mlirOpBuilderCreate(ctx);

    return Self{
        .ctx = ctx,
        .module = c.MlirModule{},
        .op_builder = op_builder,
        .symbol_table = symbol_table_t.init(allocator),
        .function_map = func_map_t.init(allocator),
        .struct_map = struct_map_t.init(allocator),
        .allocator = allocator,
    };
}

pub fn deinit(self: *Self) void {
    self.symbol_table.deinit();
    self.function_map.deinit();
    c.mlirModuleDestroy(self.module);
}

pub fn fromModule(self: *Self, module_ast: *ast.ModuleAST) MLIRGenError!?c.MlirModule {
    // Delete existing module and create a new & empty one
    if (!c.mlirModuleIsNull(self.module)) {
        c.mlirModuleDestroy(self.module);
    }
    const unknown_loc = c.mlirOpBuilderGetUnknownLoc(self.op_builder);
    self.module = c.mlirModuleCreateEmpty(unknown_loc);

    for (module_ast.getRecords()) |record| {
        switch (record) {
            .Function => |func_ast| {
                // TODO: rewrite this once we don't return null anymore
                if (try self.fromFunc(func_ast)) |func_op| {
                    try self.function_map.put(func_ast.getProto().getName(), func_op);
                }
            },
            .Struct => |struct_ast| {
                try self.fromStruct(struct_ast);
            },
        }
    }

    // Verify the module after it's constructed
    const module_op = c.mlirModuleGetOperation(self.module);
    if (c.mlirLogicalResultIsFailure(c.mlirExtVerify(module_op))) {
        c.mlirExtOperationEmitError(module_op, "module verification error");
        return null;
    }
    return self.module;
}

fn declare(self: *Self, decl: *ast.VarDeclExprAST, value: c.MlirValue) MLIRGenError!bool {
    const name = decl.getName();
    if (self.symbol_table.contains(name)) {
        return false;
    }
    try self.symbol_table.put(name, .{ .v1 = value, .v2 = decl });
    return true;
}

fn varTypeToMlirType(self: *Self, var_type: ast.VarType, loc: lexer.Location) MLIRGenError!c.MlirType {
    switch (var_type) {
        .named => |v| {
            const type_name = v.name;
            if (self.struct_map.get(type_name)) |pair| {
                return pair.v1;
            } else {
                const _loc = locToMlirLoc(self.ctx, loc);
                emitError(_loc, "unknown struct type '{s}'", .{type_name});
                return MLIRGenError.Type;
            }
        },
        .shaped => |v| {
            if (v.shape.len == 0) {
                return c.mlirUnrankedTensorTypeGet(c.mlirF64TypeGet(self.ctx));
            } else {
                const element_t = c.mlirF64TypeGet(self.ctx);
                const encoding = c.mlirAttributeGetNull();
                return c.mlirRankedTensorTypeGet(
                    @intCast(v.shape.len),
                    v.shape.ptr,
                    element_t,
                    encoding,
                );
            }
        },
    }
}

// Get `StructAST` from a field access pattern like `Foo.bar`.
fn getStructFor(self: *Self, expr: ast.ExprAST) MLIRGenError!*ast.StructAST {
    const expr_loc = expr.loc();

    var struct_name: []const u8 = undefined;
    switch (expr) {
        .Var => |_var| if (self.symbol_table.get(_var.getName())) |pair| {
            if (c.mlirValueIsNull(pair.v1)) {
                const loc = locToMlirLoc(self.ctx, expr_loc);
                emitError(loc, "Failed to get struct for '{s}' declaration", .{_var.getName()});
                return MLIRGenError.Struct;
            }

            const var_type = pair.v2.getType();
            std.debug.assert(var_type == .named);
            struct_name = var_type.named.name;
        },
        // Handle struct field accessing, e.g., `Foo.bar`:
        // - `Foo`: parent_struct
        // - `.`: accessor
        // - `bar`: field
        .BinOp => |access| {
            if (access.getOp() != '.') {
                const loc = locToMlirLoc(self.ctx, expr_loc);
                emitError(loc, "Invalid accessor '{c}' for struct field", .{access.getOp()});
                return MLIRGenError.StructField;
            }

            const rhs = access.getRHS();
            if (rhs != .Var) {
                const loc = locToMlirLoc(self.ctx, expr_loc);
                emitError(loc, "Expect a struct field to access", .{});
                return MLIRGenError.StructField;
            }

            const parent_struct = try self.getStructFor(access.getLHS());
            const parent_name = parent_struct.getName();
            const field_name = rhs.Var.getName();

            // Get the element within the struct corresponding to the name
            const decl = blk: for (parent_struct.getVariables()) |v| {
                if (std.mem.eql(u8, v.getName(), field_name)) {
                    break :blk v;
                }
            } else {
                const loc = locToMlirLoc(self.ctx, expr_loc);
                emitError(loc, "Struct '{s}' does not have a field '{s}'", .{ parent_name, field_name });
                return MLIRGenError.StructField;
            };

            struct_name = decl.getType().named.name;
        },
        else => {
            const loc = locToMlirLoc(self.ctx, expr_loc);
            emitError(loc, "Unexpect expr to access struct field", .{});
            return MLIRGenError.StructField;
        },
    }

    if (self.struct_map.get(struct_name)) |the_struct| {
        return the_struct.v2;
    } else {
        const loc = locToMlirLoc(self.ctx, expr_loc);
        emitError(loc, "Undefined struct '{s}'", .{struct_name});
        return MLIRGenError.StructField;
    }
}

fn getMemberIndex(self: *Self, access_op: *ast.BinaryExprAST) MLIRGenError!usize {
    std.debug.assert(access_op.getOp() == '.');

    const struct_ast = try self.getStructFor(access_op.getLHS());
    const rhs = access_op.getRHS();

    if (rhs != .Var) {
        const loc = locToMlirLoc(self.ctx, rhs.loc());
        emitError(loc, "Expect a struct field to access", .{});
        return MLIRGenError.StructField;
    }
    const rhs_name = rhs.Var.getName();

    const struct_vars = struct_ast.getVariables();
    for (0..struct_vars.len, struct_vars) |i, v| {
        if (std.mem.eql(u8, v.getName(), rhs_name)) {
            return i;
        }
    }

    const loc = locToMlirLoc(self.ctx, rhs.loc());
    emitError(loc, "Struct '{s}' does not have a field '{s}'", .{ struct_ast.getName(), rhs_name });
    return MLIRGenError.StructField;
}

fn fromStruct(self: *Self, struct_ast: *ast.StructAST) MLIRGenError!void {
    const struct_name = struct_ast.getName();
    if (self.struct_map.contains(struct_name)) {
        const loc = locToMlirLoc(self.ctx, struct_ast.loc());
        emitError(loc, "struct type with name `{s}` already exists", .{struct_ast.getName()});
    }

    const vars = struct_ast.getVariables();
    var el_types_al = std.ArrayList(c.MlirType).init(self.allocator);
    // errdefer el_types_al.deinit();
    defer el_types_al.deinit();

    for (vars) |v| {
        const var_type = v.getType();
        const loc = v.loc();

        if (v.getInitVal()) |_| {
            const _loc = locToMlirLoc(self.ctx, loc);
            emitError(_loc, "variables within a struct definition must not have initializers", .{});
            return MLIRGenError.Struct;
        }
        // TODO: check whether these conditions is useful for checking initializer
        if (var_type == .shaped and var_type.shaped.shape.len != 0) {
            const _loc = locToMlirLoc(self.ctx, loc);
            emitError(_loc, "variables within a struct definition must not have initializers", .{});
            return MLIRGenError.Struct;
        }

        const el_type = try self.varTypeToMlirType(var_type, loc);
        try el_types_al.append(el_type);
    }

    const el_types = el_types_al.items;
    const struct_type = c.mlirToyStructTypeGet(@intCast(el_types.len), el_types.ptr);
    try self.struct_map.put(struct_name, .{ .v1 = struct_type, .v2 = struct_ast });
}

fn fromProto(self: *Self, proto_ast: *ast.PrototypeAST) MLIRGenError!c.MlirToyFuncOp {
    const loc = locToMlirLoc(self.ctx, proto_ast.loc());
    const name = z2strref(proto_ast.getName());
    const args = proto_ast.getArgs();

    var inputs = try std.ArrayList(c.MlirType).initCapacity(self.allocator, args.len);
    defer inputs.deinit();

    for (proto_ast.getArgs()) |arg| {
        const arg_type = arg.getType();
        const arg_loc = arg.loc();
        const input_t = try self.varTypeToMlirType(arg_type, arg_loc);
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

fn fromFunc(self: *Self, func_ast: *ast.FunctionAST) MLIRGenError!?c.MlirToyFuncOp {
    try self.symbol_table.createScope();
    defer self.symbol_table.destroyScope();

    const block = c.mlirModuleGetBody(self.module);
    c.mlirOpBuilderSetInsertionPointToEnd(self.op_builder, block);
    const func_op = try self.fromProto(func_ast.getProto());

    const op = c.mlirToyFuncOpToMlirOperation(func_op);
    if (c.mlirOperationIsNull(op)) {
        return null;
    }

    const first_region = c.mlirOperationGetFirstRegion(op);
    const entry_block = c.mlirRegionGetFirstBlock(first_region);
    const proto_args = func_ast.getProto().getArgs();

    const n_args = c.mlirBlockGetNumArguments(entry_block);
    for (0..@intCast(n_args)) |i| {
        const arg = c.mlirBlockGetArgument(entry_block, @intCast(i));
        const success = try self.declare(proto_args[i], arg);
        if (!success) {
            return null;
        }
    }

    c.mlirOpBuilderSetInsertionPointToStart(self.op_builder, entry_block);

    const func_body = func_ast.getBody();
    if (!try self.fromBlock(func_body)) {
        c.mlirToyFuncOpErase(func_op);
        return null;
    }

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

    // If this function isn't main, then set the visibility to private.
    if (!std.mem.eql(u8, func_ast.getProto().getName(), "main")) {
        c.mlirToyFuncOpSetPrivate(func_op);
    }
    return func_op;
}

fn fromBinary(self: *Self, bin: *ast.BinaryExprAST) MLIRGenError!?c.MlirValue {
    const lhs = try self.fromExpr(bin.getLHS()) orelse return null;
    const loc = locToMlirLoc(self.ctx, bin.loc());

    // Handle field accessor first
    if (bin.getOp() == '.') {
        const access_idx = try self.getMemberIndex(bin);
        std.debug.assert(!c.mlirValueIsNull(lhs));
        return c.mlirToyStructAccessOpCreate(self.op_builder, loc, lhs, @intCast(access_idx));
    }

    // Otherwise, this is a normal binary op
    const rhs = try self.fromExpr(bin.getRHS()) orelse return null;

    // NOTE: Currently we only support AddOp and MulOp in this chapter.
    switch (bin.getOp()) {
        '+' => return c.mlirToyAddOpCreate(self.op_builder, loc, lhs, rhs),
        '*' => return c.mlirToyMulOpCreate(self.op_builder, loc, lhs, rhs),
        else => |v| {
            emitError(loc, "invalid binary operator '{c}'", .{v});
            return null;
        },
    }
}

fn fromVar(self: *Self, expr: *ast.VariableExprAST) MLIRGenError!c.MlirValue {
    const name = expr.getName();
    if (self.symbol_table.get(name)) |pair| {
        return pair.v1;
    }

    const loc = locToMlirLoc(self.ctx, expr.loc());
    emitError(loc, "unknown variable '{s}'", .{name});
    return MLIRGenError.Var;
}

fn fromReturn(self: *Self, ret: *ast.ReturnExprAST) MLIRGenError!bool {
    const loc = locToMlirLoc(self.ctx, ret.loc());

    var operands = [1]c.MlirValue{undefined};
    var n_operands: i64 = 0;
    if (ret.getExpr()) |expr| {
        operands[0] = try self.fromExpr(expr) orelse return false;
        n_operands += 1;
    }

    std.debug.assert(n_operands <= 1);
    c.mlirToyReturnOpCreate(self.op_builder, loc, n_operands, &operands);
    return true;
}

fn getConstantAttrFromNumber(self: *Self, num: *ast.NumberExprAST) MLIRGenError!c.MlirAttribute {
    var data = try std.ArrayList(f64).initCapacity(self.allocator, 1);
    defer data.deinit();
    data.appendAssumeCapacity(num.getValue());

    const element_t = c.mlirF64TypeGet(self.ctx);
    const data_t = c.mlirUnrankedTensorTypeGet(element_t);
    const data_attr = c.mlirDenseElementsAttrDoubleGet(
        data_t,
        @intCast(data.items.len),
        data.items.ptr,
    );
    return data_attr;
}

fn getConstantAttrFromLiteral(self: *Self, lit: *ast.LiteralExprAST) MLIRGenError!c.MlirAttribute {
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
    const data_t = c.mlirRankedTensorTypeGet(
        @intCast(dims.shape.len),
        dims.shape.ptr,
        element_t,
        encoding,
    );

    // NOTE: data will be copied and stored in its internal storage by
    // `mlir::DenseElementsAttr::get()`. See `writeBits()` in the link below.
    // https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/IR/BuiltinAttributes.cpp#L969
    const data_attr = c.mlirDenseElementsAttrDoubleGet(
        data_t,
        @intCast(data.items.len),
        data.items.ptr,
    );
    return data_attr;
}

fn getConstantAttrFromStructLiteral(self: *Self, slit: *ast.StructLiteralExprAST) MLIRGenError!MLIRStructLiteralData {
    var attrs_al = std.ArrayList(c.MlirAttribute).init(self.allocator);
    defer attrs_al.deinit();
    var types_al = std.ArrayList(c.MlirType).init(self.allocator);
    defer types_al.deinit();

    for (slit.getValues()) |val| {
        switch (val) {
            .Num => |num| {
                const val_attr = try self.getConstantAttrFromNumber(num);
                const element_t = c.mlirF64TypeGet(self.ctx);
                const val_type = c.mlirUnrankedTensorTypeGet(element_t);
                try attrs_al.append(val_attr);
                try types_al.append(val_type);
            },
            .Literal => |lit| {
                const val_attr = try self.getConstantAttrFromLiteral(lit);
                const element_t = c.mlirF64TypeGet(self.ctx);
                const val_type = c.mlirUnrankedTensorTypeGet(element_t);
                try attrs_al.append(val_attr);
                try types_al.append(val_type);
            },
            .StructLiteral => |inner_slit| {
                const slit_data = try self.getConstantAttrFromStructLiteral(inner_slit);
                const val_attr = slit_data.v1;
                const val_type = slit_data.v2;
                try attrs_al.append(val_attr);
                try types_al.append(val_type);
            },
            else => {
                const _loc = locToMlirLoc(self.ctx, slit.loc());
                emitError(_loc, "unsupported AST type '{s}' for StructLiteral", .{@tagName(val)});
                return MLIRGenError.StructLiteral;
            },
        }
    }

    const attrs = attrs_al.items;
    const types = types_al.items;
    const data_attr = c.mlirArrayAttrGet(self.ctx, @intCast(attrs.len), attrs.ptr);
    const data_type = c.mlirToyStructTypeGet(@intCast(types.len), types.ptr);
    std.debug.assert(!c.mlirAttributeIsNull(data_attr));
    std.debug.assert(!c.mlirTypeIsNull(data_type));
    return .{ .v1 = data_attr, .v2 = data_type };
}

fn fromLiteral(self: *Self, lit: *ast.LiteralExprAST) MLIRGenError!c.MlirValue {
    const dims = lit.getDims();

    const element_t = c.mlirF64TypeGet(self.ctx);
    const encoding = c.mlirAttributeGetNull();
    const data_type = c.mlirRankedTensorTypeGet(
        @intCast(dims.shape.len),
        dims.shape.ptr,
        element_t,
        encoding,
    );

    const data_attr = try self.getConstantAttrFromLiteral(lit);
    std.debug.assert(!c.mlirAttributeIsNull(data_attr));

    const loc = locToMlirLoc(self.ctx, lit.loc());
    return c.mlirToyConstantOpCreateFromTensor(self.op_builder, loc, data_type, data_attr);
}

fn fromStructLiteral(self: *Self, slit: *ast.StructLiteralExprAST) MLIRGenError!c.MlirValue {
    const slit_data = try self.getConstantAttrFromStructLiteral(slit);
    const data_attr = slit_data.v1;
    const data_type = slit_data.v2;

    return c.mlirToyStructConstantOpCreate(
        self.op_builder,
        locToMlirLoc(self.ctx, slit.loc()),
        data_type,
        data_attr,
    );
}

fn fromCall(self: *Self, call: *ast.CallExprAST) MLIRGenError!?c.MlirValue {
    const callee = call.getCallee();
    const loc = locToMlirLoc(self.ctx, call.loc());

    const args = call.getArgs();
    var operands = try std.ArrayList(c.MlirValue).initCapacity(self.allocator, args.len);
    defer operands.deinit();

    for (args) |arg| {
        const result = try self.fromExpr(arg) orelse return null;
        try operands.append(result);
    }

    if (std.mem.eql(u8, callee, "transpose")) {
        if (args.len != 1) {
            const msg = "MLIR codegen encountered an error: toy.transpose does not accept multiple arguments";
            emitError(loc, msg, .{});
            return null;
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

fn fromPrint(self: *Self, call: *ast.PrintExprAST) MLIRGenError!bool {
    const arg = try self.fromExpr(call.getArg()) orelse return false;
    const loc = locToMlirLoc(self.ctx, call.loc());
    c.mlirToyPrintOpCreate(self.op_builder, loc, arg);
    return true;
}

fn fromNumber(self: *Self, num: *ast.NumberExprAST) c.MlirValue {
    const loc = locToMlirLoc(self.ctx, num.loc());
    return c.mlirToyConstantOpCreateFromDouble(self.op_builder, loc, num.getValue());
}

fn fromExpr(self: *Self, expr: ast.ExprAST) MLIRGenError!?c.MlirValue {
    switch (expr) {
        .BinOp => |v| return try self.fromBinary(v),
        .Var => |v| return try self.fromVar(v),
        .Literal => |v| return try self.fromLiteral(v),
        .Call => |v| return try self.fromCall(v),
        .Num => |v| return self.fromNumber(v),
        .StructLiteral => |v| return try self.fromStructLiteral(v),
        inline else => |_, tag| {
            const loc = locToMlirLoc(self.ctx, expr.loc());
            const msg = "MLIR codegen encountered an unhandled expr kind '{s}'";
            emitError(loc, msg, .{@tagName(tag)});
            return null;
        },
    }
}

fn fromVarDecl(self: *Self, var_decl: *ast.VarDeclExprAST) MLIRGenError!?c.MlirValue {
    const init_val = var_decl.getInitVal() orelse {
        const loc = locToMlirLoc(self.ctx, var_decl.loc());
        emitError(loc, "missing initializer in variable declaration.", .{});
        return null;
    };

    var value = try self.fromExpr(init_val) orelse return null;

    switch (var_decl.getType()) {
        .named => {
            // Check that the initializer type is the same as the variable declaration.
            const decl_mlir_type = try self.varTypeToMlirType(var_decl.getType(), var_decl.loc());
            const value_mlir_type = c.mlirValueGetType(value);

            if (!c.mlirTypeEqual(decl_mlir_type, value_mlir_type)) {
                var pbuf1 = try c_api.PrintBuffer.init(self.allocator, 1024);
                defer pbuf1.deinit();
                var pbuf2 = try c_api.PrintBuffer.init(self.allocator, 1024);
                defer pbuf2.deinit();

                const msg_buf = try self.allocator.alloc(u8, 2048);
                defer self.allocator.free(msg_buf);

                c.mlirTypePrint(value_mlir_type, c_api.printToBuf, &pbuf1);
                c.mlirTypePrint(decl_mlir_type, c_api.printToBuf, &pbuf2);

                const loc = locToMlirLoc(self.ctx, var_decl.loc());
                emitErrorWithBuffer(loc, "struct type of initializer is different than the variable declaration. Got '{s}', but expected '{s}'", .{
                    pbuf1.buf[0..pbuf1.print_len], pbuf2.buf[0..pbuf2.print_len],
                }, msg_buf);
                return MLIRGenError.VarDecl;
            }
        },
        .shaped => |shaped_type| {
            // We have the initializer value, but in case the variable was declared
            // with specific shape, we emit a "reshape" operation. It will get
            // optimized out later as needed.
            if (shaped_type.shape.len != 0) {
                const loc = locToMlirLoc(self.ctx, var_decl.loc());

                const element_t = c.mlirF64TypeGet(self.ctx);
                const encoding = c.mlirAttributeGetNull();
                const data_t = c.mlirRankedTensorTypeGet(
                    @intCast(shaped_type.shape.len),
                    shaped_type.shape.ptr,
                    element_t,
                    encoding,
                );

                value = c.mlirToyReshapeOpCreate(self.op_builder, loc, data_t, value);
            }
        },
    }

    if (!try self.declare(var_decl, value)) {
        return null;
    }
    return value;
}

fn fromBlock(self: *Self, block: ast.ExprASTList) MLIRGenError!bool {
    try self.symbol_table.createScope();
    defer self.symbol_table.destroyScope();

    for (block) |expr| switch (expr) {
        .VarDecl => |v| {
            _ = try self.fromVarDecl(v) orelse return false;
        },
        .Return => |v| {
            _ = try self.fromReturn(v);
        },
        .Print => |v| {
            // XXX: here we return false when it failed to generate `PrintOp`,
            // but it's weird that it returns `mlir::success` in the original
            // example.
            if (!try self.fromPrint(v)) {
                return false;
            }
        },
        inline else => {
            _ = try self.fromExpr(expr) orelse return false;
        },
    };
    return true;
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

fn emitErrorWithBuffer(loc: c.MlirLocation, comptime fmt: []const u8, args: anytype, buf: []u8) void {
    const msg = std.fmt.bufPrintZ(buf, fmt, args) catch @panic("Message length exceeds buffer length.");
    c.mlirEmitError(loc, msg.ptr);
}

fn collectData(
    expr: ast.ExprAST,
    data: *ArrayListF64,
    assume_capacity: bool,
) Allocator.Error!void {
    if (expr == .Literal) {
        const lit = expr.Literal;
        for (lit.getValues()) |v| {
            try collectData(v, data, assume_capacity);
        }
        return;
    }

    std.debug.assert(expr.getKind() == .Num);
    if (assume_capacity) {
        data.appendAssumeCapacity(expr.Num.getValue());
    } else {
        try data.append(expr.Num.getValue());
    }
}

fn makePair(comptime t1: type, comptime t2: type) type {
    return struct { v1: t1, v2: t2 };
}

const std = @import("std");
const c_api = @import("c_api.zig");
const c = c_api.c;
const lexer = @import("lexer.zig");
const ast = @import("ast.zig");
const ScopedHashMap = @import("./scoped_hash_map.zig").ScopedHashMap;

const Self = @This();
const Allocator = std.mem.Allocator;
