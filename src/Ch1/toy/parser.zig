const std = @import("std");
const ast = @import("ast.zig");
const lexer = @import("lexer.zig");

pub const ParseFailure = error{
    Module,
    Definition,
    Prototype,
    ExprAST,
    Block,
    VarDeclExprAST,
    VarType,
    Literal,
    Call,
    Return,
};

pub const ParseError = ParseFailure ||
    lexer.LexerError ||
    error{OutOfMemory};

pub const Parser = struct {
    _lexer: *lexer.Lexer,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(_lexer: *lexer.Lexer, allocator: std.mem.Allocator) Self {
        return .{ ._lexer = _lexer, .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        self._lexer.deinit();
    }

    pub fn parseModule(self: Self) ParseError!*ast.ModuleAST {
        _ = try self.getNextToken();

        var functions_al = ast.FunctionASTListType.ArrayList.init(self.allocator);
        errdefer {
            for (functions_al.items) |v| {
                v.deinit();
            }
            functions_al.deinit();
        }

        while (true) {
            const func_ast = try self.parseDefinition();
            functions_al.append(func_ast) catch |err| {
                // Cleanup the just-created expr here, the remaining ones
                // in list will be handled by the `errdefer` above.
                func_ast.deinit();
                return err;
            };

            if (self._lexer.getCurToken() == .tok_eof) break;
        }

        if (self._lexer.getCurToken() != .tok_eof) {
            self.printErrMsg("nothing", "at end of module");
            return ParseFailure.Module;
        }

        var functions = try ast.FunctionASTListType.fromArrayList(&functions_al);
        errdefer functions.deinit();

        const module = try ast.ModuleAST.init(self.allocator, functions);
        return module;
    }

    inline fn getNextToken(self: Self) !lexer.Token {
        return self._lexer.getNextToken() catch |err| {
            const loc = self._lexer.getLastLocation();
            std.debug.print(
                "Parse error ({d}, {d}): failed to get next token because of: {}\n",
                .{ loc.line, loc.col, err },
            );
            return err;
        };
    }

    inline fn consumeToken(self: Self, tok: lexer.Token) !void {
        return self._lexer.consume(tok) catch |err| {
            const loc = self._lexer.getLastLocation();
            std.debug.print(
                "Parse error ({d}, {d}): failed to consume token because of: {}\n",
                .{ loc.line, loc.col, err },
            );
            return err;
        };
    }

    /// return ::= return ; | return expr ;
    fn parseReturn(self: Self) ParseError!ast.ExprAST {
        const loc = self._lexer.getLastLocation();
        try self.consumeToken(.tok_return);

        var ret: *ast.ReturnExprAST = undefined;
        if (self._lexer.getCurToken() != .tok_semicolon) {
            var expr = try self.parseExpression();
            errdefer expr.deinit();

            ret = try ast.ReturnExprAST.init(self.allocator, loc, expr);
        } else {
            ret = try ast.ReturnExprAST.init(self.allocator, loc, null);
        }
        return ret.tagged();
    }

    /// numberexpr ::= number
    fn parseNumberExpr(self: Self) ParseError!ast.ExprAST {
        const loc = self._lexer.getLastLocation();
        const val = self._lexer.getValue();
        var num = try ast.NumberExprAST.init(self.allocator, loc, val);
        errdefer num.deinit();

        try self.consumeToken(lexer.Token{ .tok_num = val });
        return num.tagged();
    }

    /// tensorLiteral ::= [ literalList ] | number
    /// literalList ::= tensorLiteral | tensorLiteral, literalList
    fn parseTensorLiteralExpr(self: Self) ParseError!ast.ExprAST {
        const loc = self._lexer.getLastLocation();
        try self.consumeToken(.tok_sbracket_open);

        var values_al = ast.ExprASTListType.ArrayList.init(self.allocator);
        var dims_al = ast.VarType.ArrayList.init(self.allocator);
        errdefer {
            for (values_al.items) |v| {
                v.deinit();
            }
            values_al.deinit();
            dims_al.deinit();
        }

        const tok_comma = lexer.Token{ .tok_other = ',' };
        while (true) {
            if (self._lexer.getCurToken() == .tok_sbracket_open) {
                var literal = try self.parseTensorLiteralExpr();
                values_al.append(literal) catch |err| {
                    literal.deinit();
                    return err;
                };
            } else {
                if (self._lexer.getCurToken() != .tok_num) {
                    self.printErrMsg("<num> or [", "in literal expression");
                    return ParseFailure.Literal;
                }

                var num = try self.parseNumberExpr();
                values_al.append(num) catch |err| {
                    num.deinit();
                    return err;
                };
            }

            // End of this list on ']'
            if (self._lexer.getCurToken() == .tok_sbracket_close) {
                break;
            }

            // Elements are separated by a comma
            if (!std.meta.eql(self._lexer.getCurToken(), tok_comma)) {
                self.printErrMsg("] or ,", "in literal expression");
                return ParseFailure.Literal;
            }
            _ = try self.getNextToken(); // eat ,
        }

        if (values_al.items.len == 0) {
            self.printErrMsg("<something>", "to fill literal expression");
            return ParseFailure.Literal;
        }
        _ = try self.getNextToken(); // eat ]

        try dims_al.append(@intCast(values_al.items.len));

        const hasLit = struct {
            fn func(inputs: []ast.ExprAST) bool {
                for (inputs) |v| {
                    switch (v) {
                        .Literal => return true,
                        else => {},
                    }
                }
                return false;
            }
        }.func;

        if (hasLit(values_al.items)) {
            if (values_al.items[0] != .Literal) {
                self.printErrMsg("uniform well-nested dimensions", "inside literal expression");
                return ParseFailure.Literal;
            }
            const first_literal = values_al.items[0].Literal;
            const first_dims = first_literal.getDims();
            try dims_al.appendSlice(first_dims.shape);

            // Sanity check
            for (values_al.items) |v| {
                const expr_literal = v.Literal;
                const _dims = expr_literal.getDims();
                if (v != .Literal or !std.mem.eql(i64, _dims.shape, first_dims.shape)) {
                    self.printErrMsg("uniform well-nested dimensions", "inside literal expression");
                    return ParseFailure.Literal;
                }
            }
        }

        var values = try ast.ExprASTListType.fromArrayList(&values_al);
        errdefer values.deinit();
        var dims = try ast.VarType.fromArrayList(&dims_al);
        errdefer dims.deinit();

        const lit = try ast.LiteralExprAST.init(self.allocator, loc, values, dims);
        return lit.tagged();
    }

    /// Parse parenthesized expression.
    /// parenexpr ::= '(' expression ')'
    fn parseParenExpr(self: Self) ParseError!ast.ExprAST {
        _ = try self.getNextToken();

        var expr = try self.parseExpression();
        errdefer expr.deinit();

        if (self._lexer.getCurToken() != .tok_parenthese_close) {
            self.printErrMsg(")", "to close expression with parentheses");
            return ParseFailure.ExprAST;
        }
        try self.consumeToken(.tok_parenthese_close);
        return expr;
    }

    /// identifierexpr
    ///   ::= identifier
    ///   ::= identifier '(' expression ')'
    fn parseIdentifierExpr(self: Self) ParseError!ast.ExprAST {
        const name = self._lexer.getId();
        const loc = self._lexer.getLastLocation();
        _ = try self.getNextToken();

        if (self._lexer.getCurToken() != .tok_parenthese_open) {
            const ret = try ast.VariableExprAST.init(self.allocator, loc, name);
            return ret.tagged();
        }
        try self.consumeToken(.tok_parenthese_open);

        var args_al = ast.ExprASTListType.ArrayList.init(self.allocator);
        errdefer {
            for (args_al.items) |v| {
                v.deinit();
            }
            args_al.deinit();
        }

        if (self._lexer.getCurToken() != .tok_parenthese_close) {
            while (true) {
                const expr = try self.parseExpression();
                args_al.append(expr) catch |err| {
                    expr.deinit();
                    return err;
                };

                const cur_tok = self._lexer.getCurToken();
                if (cur_tok == .tok_parenthese_close) {
                    break;
                }

                if (!std.meta.eql(cur_tok, lexer.Token{ .tok_other = ',' })) {
                    self.printErrMsg(", or )", "in argument list");
                    return ParseFailure.Call;
                }
                _ = try self.getNextToken();
            }
        }
        try self.consumeToken(.tok_parenthese_close);

        var args = try ast.ExprASTListType.fromArrayList(&args_al);
        errdefer {
            for (args.slice) |v| {
                v.deinit();
            }
            args.deinit();
        }

        if (std.mem.eql(u8, name, "print")) {
            if (args.slice.len != 1) {
                self.printErrMsg("<single arg>", "as argument to print()");
                return ParseFailure.Call;
            }
            const ret = try ast.PrintExprAST.init(self.allocator, loc, args.slice[0]);

            // NOTE: if we are processing a print statement, we have to release
            // memory allocated for `args` since we are not handing over the
            // ownership of `args` itself to others.
            args.deinit();

            return ret.tagged();
        }

        const ret = try ast.CallExprAST.init(self.allocator, loc, name, args);
        return ret.tagged();
    }

    /// primary
    ///   ::= identifierexpr
    ///   ::= numberexpr
    ///   ::= parenexpr
    ///   ::= tensorexpr
    fn parsePrimary(self: Self) ParseError!ast.ExprAST {
        const cur_tok = self._lexer.getCurToken();
        switch (cur_tok) {
            .tok_ident => return try self.parseIdentifierExpr(),
            .tok_num => return try self.parseNumberExpr(),
            .tok_parenthese_open => return try self.parseParenExpr(),
            .tok_sbracket_open => return try self.parseTensorLiteralExpr(),
            .tok_semicolon, .tok_cbracket_close, .tok_other => return ParseFailure.ExprAST,
            else => {
                const loc = self._lexer.getLastLocation();
                std.debug.print(
                    "Parse error ({d}, {d}): unknown token '{s}' when expecting an expression\n",
                    .{ loc.line, loc.col, @tagName(cur_tok) },
                );
                return ParseFailure.ExprAST;
            },
        }
    }

    /// binoprhs ::= ('+' primary)*
    fn parseBinOpRHS(self: Self, expr_prec: i32, lhs: ast.ExprAST) ParseError!ast.ExprAST {
        var lhs_ = lhs;

        // NOTE: since `lhs_` may represent a nested BinOp, calling
        // `lhs_.deinit()` will recursively release all memory allocated for
        // its `lhs` and `rhs`. This behavior is useful to cleanup, but it can
        // be tricky to understand at first.
        errdefer lhs_.deinit();

        while (true) {
            const tok_prec = self.getTokPrecedence();
            if (tok_prec < expr_prec) {
                return lhs_;
            }

            const cur_tok = self._lexer.getCurToken();
            const bin_op = cur_tok.tok_other;
            try self.consumeToken(cur_tok);
            const loc = self._lexer.getLastLocation();

            var rhs = self.parsePrimary() catch {
                self.printErrMsg("expression", "to complete binary operator");
                return ParseFailure.ExprAST;
            };

            // NOTE: we don't need to add `errdefer rhs.deinit()` here because:
            // 1. if `rhs` is passed to the next `parseBinOpRHS()`, it will be
            //    handled by the existing `errdefer lhs_.deinit()` above. So
            //    adding `errdefer rhs.deinit()` here would acutally result in
            //    a double-free.
            // 2. if the following if-block is not entered, `rhs` will be used
            //    to create a new BinOp and update the top-level `lhs_`.
            //    In this case, the cleanup process for this `rhs` is already
            //    covered by the `errdefer lhs_.deinit()`.

            const next_prec = self.getTokPrecedence();
            if (tok_prec < next_prec) {
                rhs = try self.parseBinOpRHS(tok_prec + 1, rhs);
            }

            const bin_expr = try ast.BinaryExprAST.init(self.allocator, loc, bin_op, lhs_, rhs);
            lhs_ = bin_expr.tagged();
        }
    }

    /// expression ::= primary binop rhs
    fn parseExpression(self: Self) ParseError!ast.ExprAST {
        const lhs = try self.parsePrimary();
        return try self.parseBinOpRHS(0, lhs);
    }

    /// type ::= < shape_list >
    fn parseType(self: Self) ParseError!ast.VarType {
        if (self._lexer.getCurToken() != .tok_abracket_open) {
            self.printErrMsg("<", "to begin type");
            return ParseFailure.VarType;
        }
        try self.consumeToken(.tok_abracket_open);

        var shape_al = ast.VarType.ArrayList.init(self.allocator);
        errdefer shape_al.deinit();

        while (self._lexer.getCurToken() == .tok_num) {
            try shape_al.append(@intFromFloat(self._lexer.getValue()));
            const next_tok = try self.getNextToken();
            const tok_comma = lexer.Token{ .tok_other = ',' };
            if (std.meta.eql(next_tok, tok_comma)) {
                _ = try self.getNextToken();
            }
        }

        var var_type = try ast.VarType.fromArrayList(&shape_al);
        errdefer var_type.deinit();

        if (self._lexer.getCurToken() != .tok_abracket_close) {
            self.printErrMsg(">", "to end type");
            return ParseFailure.VarType;
        }
        _ = try self.getNextToken(); // eat >
        return var_type;
    }

    /// decl ::= var identifier [ type ] = expr
    fn parseDeclaration(self: Self) ParseError!*ast.VarDeclExprAST {
        if (self._lexer.getCurToken() != .tok_var) {
            self.printErrMsg("var", "to begin declaration");
            return ParseFailure.VarDeclExprAST;
        }

        const loc = self._lexer.getLastLocation();
        _ = try self.getNextToken();

        if (self._lexer.getCurToken() != .tok_ident) {
            self.printErrMsg("identifier", "after 'var' declaration");
            return ParseFailure.VarDeclExprAST;
        }

        const var_name = self._lexer.getId();
        _ = try self.getNextToken();

        // Type is optional because it can be inferred
        var var_type: ast.VarType = blk: {
            if (self._lexer.getCurToken() == .tok_abracket_open) {
                break :blk try self.parseType();
            } else {
                // Create a VarType with 0-sized shape
                var shape_al = ast.VarType.ArrayList.init(self.allocator);
                break :blk try ast.VarType.fromArrayList(&shape_al);
            }
        };
        errdefer var_type.deinit();

        var expr: ?ast.ExprAST = null;

        // NOTE: we have to check whether the next token is '=' before parsing
        // it as an expression. Otherwise, we would consume token incorrectly.
        const tok_assign = lexer.Token{ .tok_other = '=' };
        if (std.meta.eql(self._lexer.getCurToken(), tok_assign)) {
            try self.consumeToken(lexer.Token{ .tok_other = '=' });

            expr = try self.parseExpression();
            errdefer if (expr) |v| v.deinit();

            if (expr.?.getKind() == .Literal) {
                const lit = expr.?.Literal;
                try self.checkTensorShape(var_type, lit);
            }
        }

        const var_decl = try ast.VarDeclExprAST.init(self.allocator, loc, var_name, var_type, expr);
        return var_decl;
    }

    /// block ::= { expression_list }
    /// expression_list ::= block_expr; expression_list
    /// block_expr ::= decl | "return" | expr
    fn parseBlock(self: Self) ParseError!ast.FunctionAST.BodyType {
        if (self._lexer.getCurToken() != .tok_cbracket_open) {
            self.printErrMsg("{", "to begin block");
            return ParseFailure.Block;
        }
        try self.consumeToken(.tok_cbracket_open);

        var body_al = ast.FunctionAST.BodyType.ArrayList.init(self.allocator);
        errdefer {
            for (body_al.items) |v| {
                v.deinit();
            }
            body_al.deinit();
        }

        // Ignore empty expressions: swallow sequences of semicolons.
        while (self._lexer.getCurToken() == .tok_semicolon) {
            try self.consumeToken(.tok_semicolon);
        }

        var cur_tok = self._lexer.getCurToken();
        while (cur_tok != .tok_cbracket_close and cur_tok != .tok_eof) {
            switch (cur_tok) {
                .tok_var => {
                    var var_decl = try self.parseDeclaration();
                    body_al.append(var_decl.tagged()) catch |err| {
                        var_decl.deinit();
                        return err;
                    };
                },
                .tok_return => {
                    var ret_tagged = try self.parseReturn();
                    body_al.append(ret_tagged) catch |err| {
                        ret_tagged.deinit();
                        return err;
                    };
                },
                else => {
                    var expr = try self.parseExpression();
                    body_al.append(expr) catch |err| {
                        expr.deinit();
                        return err;
                    };
                },
            }

            if (self._lexer.getCurToken() != .tok_semicolon) {
                self.printErrMsg(";", "after expression");
                return ParseFailure.Block;
            }

            while (self._lexer.getCurToken() == .tok_semicolon) {
                try self.consumeToken(.tok_semicolon);
            }
            cur_tok = self._lexer.getCurToken();
        }

        var body = try ast.FunctionAST.BodyType.fromArrayList(&body_al);
        errdefer body.deinit();

        if (self._lexer.getCurToken() != .tok_cbracket_close) {
            self.printErrMsg("}", "to close block");
            return ParseFailure.Block;
        }
        try self.consumeToken(.tok_cbracket_close);

        return body;
    }

    /// prototype ::= def id '(' decl_list ')'
    /// decl_list ::= identifier | identifier, decl_list
    fn parsePrototype(self: Self) ParseError!*ast.PrototypeAST {
        const loc = self._lexer.getLastLocation();

        if (self._lexer.getCurToken() != .tok_def) {
            self.printErrMsg("def", "in prototype");
            return ParseFailure.Prototype;
        }
        try self.consumeToken(.tok_def);

        if (self._lexer.getCurToken() != .tok_ident) {
            self.printErrMsg("function name", "in prototype");
            return ParseFailure.Prototype;
        }

        const fn_name = self._lexer.getId();
        try self.consumeToken(.{ .tok_ident = fn_name });

        if (self._lexer.getCurToken() != .tok_parenthese_open) {
            self.printErrMsg("(", "in prototype");
            return ParseFailure.Prototype;
        }
        try self.consumeToken(.tok_parenthese_open);

        var args_al = ast.PrototypeAST.ArgsType.ArrayList.init(self.allocator);
        errdefer {
            for (args_al.items) |v| {
                v.deinit();
            }
            args_al.deinit();
        }

        while (self._lexer.getCurToken() != .tok_parenthese_close) {
            const arg_name = self._lexer.getId();
            const arg_loc = self._lexer.getLastLocation();
            try self.consumeToken(.{ .tok_ident = arg_name });

            var decl = try ast.VariableExprAST.init(self.allocator, arg_loc, arg_name);
            args_al.append(decl) catch |err| {
                decl.deinit();
                return err;
            };

            const tok_comma = lexer.Token{ .tok_other = ',' };
            if (!std.meta.eql(self._lexer.getCurToken(), tok_comma)) {
                break;
            }
            try self.consumeToken(tok_comma);

            if (self._lexer.getCurToken() != .tok_ident) {
                self.printErrMsg("identifier", "after ',' in function parameter list");
                return ParseFailure.Prototype;
            }
        }

        if (self._lexer.getCurToken() != .tok_parenthese_close) {
            self.printErrMsg(")", "to end function prototype");
            return ParseFailure.Prototype;
        }
        try self.consumeToken(.tok_parenthese_close);

        var args = try ast.PrototypeAST.ArgsType.fromArrayList(&args_al);
        errdefer args.deinit();
        const proto = try ast.PrototypeAST.init(self.allocator, loc, fn_name, args);
        return proto;
    }

    /// definition ::= prototype block
    fn parseDefinition(self: Self) ParseError!*ast.FunctionAST {
        const proto = try self.parsePrototype();
        errdefer proto.deinit();

        var body = try self.parseBlock();
        errdefer {
            for (body.slice) |v| {
                v.deinit();
            }
            body.deinit();
        }

        const func = try ast.FunctionAST.init(self.allocator, proto, body);
        return func;
    }

    fn getTokPrecedence(self: Self) i32 {
        switch (self._lexer.getCurToken()) {
            .tok_other => |c| {
                return switch (c) {
                    '-', '+' => 20,
                    '*' => 40,
                    else => -1,
                };
            },
            else => return -1,
        }
    }

    fn printErrMsg(self: Self, expected: []const u8, context: []const u8) void {
        const cur_tok = self._lexer.getCurToken();
        const loc = self._lexer.getLastLocation();
        var buf = [1]u8{0};
        std.debug.print(
            "Parse error ({d}, {d}): expected '{s}' {s} but has Token '{s}'\n",
            .{ loc.line, loc.col, expected, context, cur_tok.str(&buf) },
        );
    }

    fn checkTensorShape(self: Self, dims: ast.VarType, lit: *ast.LiteralExprAST) !void {
        const countEls = struct {
            fn count(_lit: *ast.LiteralExprAST) usize {
                const vals = _lit.getValues();
                if (vals.len == 0) {
                    return 0;
                }
                return switch (vals[0].getKind()) {
                    .Num => vals.len,
                    .Literal => {
                        var total: usize = 0;
                        for (vals) |v| {
                            total += count(v.Literal);
                        }
                        return total;
                    },
                    // Any non-number/non-literal expr will be found out in the
                    // earlier parsing loop, so we safely assume this branch is
                    // not possible to reach.
                    else => unreachable,
                };
            }

            fn func(_lit: *ast.LiteralExprAST) usize {
                return count(_lit);
            }
        }.func;

        // Shape can be inferred, so it might be a 0-length slice.
        // In this case, we don't need to check it with the content of tensor.
        if (dims.shape.len == 0) return;

        const num_els = countEls(lit);
        const expected_num_els = blk: {
            var total: usize = 1;
            for (dims.shape) |dim| {
                total *= @intCast(dim);
            }
            break :blk total;
        };
        if (num_els != expected_num_els) {
            const _loc = self._lexer.getLastLocation();
            std.debug.print(
                \\Parse error ({d}, {d}): number of tensor elements and shape mismatched.
                \\Specified shape: {any}, number of elements: {d}
                \\
            ,
                .{ _loc.line, _loc.col, dims.shape, num_els },
            );
            return ParseError.Literal;
        }
    }
};

const test_alloc = std.testing.allocator;

const ParserTestHelper = struct {
    const builtin = @import("builtin");

    pub fn shouldSkipTest() !void {
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
    }

    // Just a combination of existing tricks done in "lexer.zig" to create a
    // lexer with string as content without loading file from disk.
    pub fn createLexer(filename: []const u8, content: []const u8) !lexer.Lexer {
        const memfd = try std.posix.memfd_create(filename, std.posix.MFD.CLOEXEC);

        try std.posix.ftruncate(memfd, content.len);

        const write_len = try std.posix.write(memfd, content);
        try std.testing.expect(write_len == content.len);

        const memfile = std.fs.File{ .handle = memfd };
        defer memfile.close();

        return .{
            .last_location = .{
                .file = filename,
                .line = 0,
                .col = 0,
            },
            .identifier_str = "",
            .cur_line_buf = lexer.StringRef.init("\n"),
            .buffer = try lexer.LexerBuffer.init(memfile),
        };
    }

    // A modified `std.testing.expectError()` to make us able to clean up
    // memory allocated for returned `ast.ModuleAST` if the input error union
    // is not an error.
    pub fn expectParseError(
        comptime expected_error: anyerror,
        actual_error_union: ParseError!*ast.ModuleAST,
    ) !void {
        if (actual_error_union) |actual_payload| {
            std.debug.print("expected 'error.{s}', found a non-error value of type '{s}'\n", .{
                @errorName(expected_error),
                @typeName(@TypeOf(actual_payload)),
            });

            actual_payload.deinit();
            return error.TestUnexpectedError;
        } else |actual_error| {
            if (expected_error != actual_error) {
                std.debug.print("expected 'error.{s}', found 'error.{s}'\n", .{
                    @errorName(expected_error),
                    @errorName(actual_error),
                });
                return error.TestExpectedError;
            }
        }
    }

    pub fn runAndExpectParseError(
        allocator: std.mem.Allocator,
        test_src: std.builtin.SourceLocation,
        comptime expected_error: anyerror,
        content: []const u8,
    ) !void {
        try ParserTestHelper.shouldSkipTest();

        // We use the name of test case here if case we need debugging
        const fname = test_src.fn_name;

        var _lexer = try ParserTestHelper.createLexer(fname, content);
        defer _lexer.deinit();
        var _parser = Parser.init(&_lexer, allocator);
        defer _parser.deinit();

        ParserTestHelper.expectParseError(
            expected_error,
            _parser.parseModule(),
        ) catch |err| {
            std.debug.print("===== Failed with test content =====\n", .{});
            std.debug.print("{s}\n", .{content});
            std.debug.print("====================================\n", .{});
            return err;
        };
    }
};

test "invalid return expr" {
    const content =
        \\def main() {
        \\  return +;
        \\}
    ;
    const exp_err = ParseError.ExprAST;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid tensor literal expr 1" {
    const content =
        \\def main() {
        \\  [1, 2, +, 3]
        \\}
    ;
    const exp_err = ParseError.Literal;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid tensor literal expr 2" {
    const content =
        \\def main() {
        \\  [1 _ 2]
        \\}
    ;
    const exp_err = ParseError.Literal;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid tensor literal expr - non-uniform" {
    const content =
        \\def main() {
        \\  [[1, 2], [3]];
        \\}
    ;
    const exp_err = ParseError.Literal;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid parenthesized expr" {
    const content =
        \\def main() {
        \\  (1
        \\}
    ;
    const exp_err = ParseError.ExprAST;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid call" {
    const content =
        \\def main() {
        \\  foo(1 2)
        \\}
    ;
    const exp_err = ParseError.Call;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid print" {
    // print() takes only 1 argument
    const content =
        \\def main() {
        \\  print(1, 2)
        \\}
    ;
    const exp_err = ParseError.Call;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid expr in nested binary expr" {
    const content =
        \\def main() {
        \\  return (1 + (3 * 2) * (2 + (4 * 5) * (6 * (2 - (1 + 3 -)) - 1))
        \\}
    ;
    const exp_err = ParseError.ExprAST;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid binary expr" {
    const content =
        \\def main() {
        \\  var a = + 1
        \\}
    ;
    const exp_err = ParseError.ExprAST;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid var type" {
    // Shape should contain a comma/space-separated list of numbers
    const content =
        \\def main() {
        \\  var a<1,,>;
        \\}
    ;
    const exp_err = ParseError.VarType;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid declaration" {
    const content =
        \\def main() {
        \\  var 1 = 2;
        \\}
    ;
    const exp_err = ParseError.VarDeclExprAST;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid block - 1" {
    const content =
        \\def main()
    ;
    const exp_err = ParseError.Block;
    try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
}

test "invalid block - 2" {
    // Variable/Declaration/Literal should end with a semicolon
    const content_list = [_][]const u8{
        \\def main() {
        \\  a
        \\}
        ,
        \\def main() {
        \\  var b = 1
        \\}
        ,
        \\def main() {
        \\  [1, 2, 3]
        \\}
    };
    const exp_err = ParseError.Block;

    for (content_list) |content| {
        try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
    }
}

test "invalid prototype" {
    const content_list = [_][]const u8{
        \\def {}
        ,
        \\def add {}
        ,
        \\def
    };
    const exp_err = ParseError.Prototype;

    for (content_list) |content| {
        try ParserTestHelper.runAndExpectParseError(test_alloc, @src(), exp_err, content);
    }
}
