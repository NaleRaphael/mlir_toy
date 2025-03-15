const std = @import("std");
const ast = @import("ast.zig");
const lexer = @import("lexer.zig");

pub const ParseFailure = error{
    Module,
    Definition,
    Prototype,
    ExprASTList,
    ExprAST,
    Block,
    VarDeclExprAST,
    VarType,
    Literal,
    Return,
    Unexpected,
    InvalidToken,
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

    pub fn parseModule(self: Self) ParseError!*ast.ModuleAST {
        _ = try self.getNextToken();

        var functions_al = ast.FunctionASTListType.ArrayList.init(self.allocator);
        while (true) {
            const func_ast = try self.parseDefinition();
            try functions_al.append(func_ast);
            if (self._lexer.getCurToken() == .tok_eof) break;
        }

        if (self._lexer.getCurToken() != .tok_eof) {
            self.printErrMsg("nothing", "at end of module");
            return ParseFailure.Module;
        }

        const functions = try ast.FunctionASTListType.fromArrayList(&functions_al);
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
            const expr = try self.parseExpression();
            if (expr) |v| {
                ret = try ast.ReturnExprAST.init(self.allocator, loc, v);
            } else {
                std.debug.print("Invalid expression for ReturnExpr", .{});
                return ParseFailure.Return;
            }
        } else {
            ret = try ast.ReturnExprAST.init(self.allocator, loc, null);
        }
        return ret.tagged();
    }

    /// numberexpr ::= number
    fn parseNumberExpr(self: Self) ParseError!ast.ExprAST {
        const loc = self._lexer.getLastLocation();
        const val = self._lexer.getValue();
        const num = try ast.NumberExprAST.init(self.allocator, loc, val);
        try self.consumeToken(lexer.Token{ .tok_num = val });
        return num.tagged();
    }

    /// tensorLiteral ::= [ literalList ] | number
    /// literalList ::= tensorLiteral | tensorLiteral, literalList
    fn parseTensorLiteralExpr(self: Self) ParseError!?ast.ExprAST {
        const loc = self._lexer.getLastLocation();
        try self.consumeToken(.tok_sbracket_open);

        var values_al = ast.ExprASTListType.ArrayList.init(self.allocator);
        var dims_al = ast.VarType.ArrayList.init(self.allocator);

        const tok_comma = lexer.Token{ .tok_other = ',' };
        while (true) {
            if (self._lexer.getCurToken() == .tok_sbracket_open) {
                const literal = try self.parseTensorLiteralExpr();
                if (literal) |v| {
                    try values_al.append(v);
                } else {
                    return null; // prase error in the nested arrary
                }
            } else {
                if (self._lexer.getCurToken() != .tok_num) {
                    self.printErrMsg("<num> or [", "in literal expression");
                    return ParseFailure.ExprAST;
                }
                const num = try self.parseNumberExpr();
                try values_al.append(num);
            }

            // End of this list on ']'
            if (self._lexer.getCurToken() == .tok_sbracket_close) {
                break;
            }

            // Elements are separated by a comma
            if (!std.meta.eql(self._lexer.getCurToken(), tok_comma)) {
                self.printErrMsg("] or ,", "in literal expression");
                return ParseFailure.ExprAST;
            }
            _ = try self.getNextToken(); // eat ,
        }

        if (values_al.items.len == 0) {
            self.printErrMsg("<something>", "to fill literal expression");
            return ParseFailure.ExprAST;
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
                return ParseFailure.ExprAST;
            }
            var first_literal = values_al.items[0].asPtr(*ast.LiteralExprAST);
            const first_dims = first_literal.getDims();
            try dims_al.appendSlice(first_dims.shape);

            // Sanity check
            for (values_al.items) |v| {
                const expr_literal = v.asPtr(*ast.LiteralExprAST);
                const _dims = expr_literal.getDims();
                if (v != .Literal or !std.mem.eql(i64, _dims.shape, first_dims.shape)) {
                    self.printErrMsg("uniform well-nested dimensions", "inside literal expression");
                    return ParseFailure.ExprAST;
                }
            }
        }

        const values = try ast.ExprASTListType.fromArrayList(&values_al);
        const dims = try ast.VarType.fromArrayList(&dims_al);
        const lit = try ast.LiteralExprAST.init(self.allocator, loc, values, dims);
        return lit.tagged();
    }

    /// Parse parenthesized expression.
    /// parenexpr ::= '(' expression ')'
    fn parseParenExpr(self: Self) ParseError!?ast.ExprAST {
        _ = try self.getNextToken();

        const expr = try self.parseExpression();
        if (expr) |v| {
            if (self._lexer.getCurToken() != .tok_parenthese_close) {
                self.printErrMsg(")", "to close expression with parentheses");
                return ParseFailure.ExprAST;
            }
            try self.consumeToken(.tok_parenthese_close);
            return v;
        } else {
            return null;
        }
    }

    /// identifierexpr
    ///   ::= identifier
    ///   ::= identifier '(' expression ')'
    fn parseIdentifierExpr(self: Self) ParseError!?ast.ExprAST {
        const name = self._lexer.getId();
        const loc = self._lexer.getLastLocation();
        _ = try self.getNextToken();

        if (self._lexer.getCurToken() != .tok_parenthese_open) {
            const ret = try ast.VariableExprAST.init(self.allocator, loc, name);
            return ret.tagged();
        }
        try self.consumeToken(.tok_parenthese_open);

        var args_al = ast.ExprASTListType.ArrayList.init(self.allocator);
        if (self._lexer.getCurToken() != .tok_parenthese_close) {
            while (true) {
                const expr = try self.parseExpression();
                if (expr) |arg| {
                    try args_al.append(arg);
                } else {
                    return null;
                }

                const cur_tok = self._lexer.getCurToken();
                if (cur_tok == .tok_parenthese_close) {
                    break;
                }

                if (!std.meta.eql(cur_tok, lexer.Token{ .tok_other = ',' })) {
                    self.printErrMsg(", or )", "in argument list");
                    return ParseFailure.ExprAST;
                }
                _ = try self.getNextToken();
            }
        }
        try self.consumeToken(.tok_parenthese_close);

        const args = try ast.ExprASTListType.fromArrayList(&args_al);

        if (std.mem.eql(u8, name, "print")) {
            if (args.slice.len != 1) {
                self.printErrMsg("<single arg>", "as argument to print()");
                return ParseFailure.ExprAST;
            }
            const ret = try ast.PrintExprAST.init(self.allocator, loc, args.slice[0]);
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
    fn parsePrimary(self: Self) ParseError!?ast.ExprAST {
        const cur_tok = self._lexer.getCurToken();
        switch (cur_tok) {
            .tok_ident => return self.parseIdentifierExpr(),
            .tok_num => return self.parseNumberExpr() catch null,
            .tok_parenthese_open => return self.parseParenExpr(),
            .tok_sbracket_open => return self.parseTensorLiteralExpr(),
            .tok_semicolon => return null,
            .tok_cbracket_close => return null,
            .tok_other => return null,
            else => {
                const loc = self._lexer.getLastLocation();
                std.debug.print(
                    "Parse error ({d}, {d}): unknown token '{s}' when expecting an expression\n",
                    .{ loc.line, loc.col, @tagName(cur_tok) },
                );
                return ParseFailure.Unexpected;
            },
        }
    }

    /// binoprhs ::= ('+' primary)*
    fn parseBinOpRHS(self: Self, expr_prec: i32, lhs: ast.ExprAST) ParseError!?ast.ExprAST {
        var lhs_ = lhs;

        while (true) {
            const tok_prec = self.getTokPrecedence();
            if (tok_prec < expr_prec) {
                return lhs_;
            }

            const cur_tok = self._lexer.getCurToken();
            const bin_op = cur_tok.tok_other;
            try self.consumeToken(cur_tok);
            const loc = self._lexer.getLastLocation();

            var rhs = try self.parsePrimary();
            if (rhs == null) {
                self.printErrMsg("expression", "to complete binary operator");
                return ParseFailure.ExprAST;
            }

            const next_prec = self.getTokPrecedence();
            if (tok_prec < next_prec) {
                rhs = try self.parseBinOpRHS(tok_prec + 1, rhs.?);
                if (rhs == null) {
                    return null;
                }
            }

            const bin_expr = try ast.BinaryExprAST.init(self.allocator, loc, bin_op, lhs_, rhs.?);
            lhs_ = bin_expr.tagged();
        }
    }

    /// expression ::= primary binop rhs
    fn parseExpression(self: Self) ParseError!?ast.ExprAST {
        const lhs = try self.parsePrimary();
        if (lhs) |v| {
            return try self.parseBinOpRHS(0, v);
        } else {
            return null;
        }
    }

    /// type ::= < shape_list >
    fn parseType(self: Self) ParseError!ast.VarType {
        if (self._lexer.getCurToken() != .tok_abracket_open) {
            self.printErrMsg("<", "to begin type");
            return ParseFailure.VarType;
        }
        try self.consumeToken(.tok_abracket_open);

        var shape_al = ast.VarType.ArrayList.init(self.allocator);
        while (self._lexer.getCurToken() == .tok_num) {
            try shape_al.append(@intFromFloat(self._lexer.getValue()));
            const next_tok = try self.getNextToken();
            const tok_comma = lexer.Token{ .tok_other = ',' };
            if (std.meta.eql(next_tok, tok_comma)) {
                _ = try self.getNextToken();
            }
        }

        const var_type = try ast.VarType.fromArrayList(&shape_al);

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

        const var_name = self._lexer.getId();
        _ = try self.getNextToken();

        // Type is optional because it can be inferred
        const var_type: ast.VarType = blk: {
            if (self._lexer.getCurToken() == .tok_abracket_open) {
                break :blk try self.parseType();
            } else {
                // Create a VarType with 0-sized shape
                var shape_al = ast.VarType.ArrayList.init(self.allocator);
                break :blk try ast.VarType.fromArrayList(&shape_al);
            }
        };
        try self.consumeToken(lexer.Token{ .tok_other = '=' });

        const expr = try self.parseExpression();

        if (expr != null and expr.?.getKind() == .Literal) {
            const lit = expr.?.asPtr(*ast.LiteralExprAST);
            try self.checkTensorShape(var_type, lit);
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
            return ParseFailure.ExprASTList;
        }
        try self.consumeToken(.tok_cbracket_open);

        var body_al = ast.FunctionAST.BodyType.ArrayList.init(self.allocator);

        // Ignore empty expressions: swallow sequences of semicolons.
        while (self._lexer.getCurToken() == .tok_semicolon) {
            try self.consumeToken(.tok_semicolon);
        }

        var cur_tok = self._lexer.getCurToken();
        while (cur_tok != .tok_cbracket_close and cur_tok != .tok_eof) {
            switch (cur_tok) {
                .tok_var => {
                    const var_decl = try self.parseDeclaration();
                    try body_al.append(var_decl.tagged());
                },
                .tok_return => {
                    const ret_tagged = try self.parseReturn();
                    try body_al.append(ret_tagged);
                },
                else => {
                    const expr = try self.parseExpression();
                    if (expr) |v| {
                        try body_al.append(v);
                    } else {
                        // In C++ impl, it returns a nullptr here. But we want
                        // to avoid abusing it.
                        const loc = self._lexer.getLastLocation();
                        std.debug.print(
                            "Parse error ({d}, {d}): Invalid declaration/expression while parsing body\n",
                            .{ loc.line, loc.col },
                        );
                    }
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

        const body = try ast.FunctionAST.BodyType.fromArrayList(&body_al);

        if (self._lexer.getCurToken() != .tok_cbracket_close) {
            self.printErrMsg("}", "to close block");
            return ParseFailure.ExprASTList;
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
        while (self._lexer.getCurToken() != .tok_parenthese_close) {
            const arg_name = self._lexer.getId();
            const arg_loc = self._lexer.getLastLocation();
            try self.consumeToken(.{ .tok_ident = arg_name });

            const decl = try ast.VariableExprAST.init(self.allocator, arg_loc, arg_name);
            try args_al.append(decl);

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

        const args = try ast.PrototypeAST.ArgsType.fromArrayList(&args_al);
        const proto = try ast.PrototypeAST.init(self.allocator, loc, fn_name, args);
        return proto;
    }

    /// definition ::= prototype block
    fn parseDefinition(self: Self) ParseError!*ast.FunctionAST {
        const proto = try self.parsePrototype();
        const body = try self.parseBlock();
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
        std.debug.print(
            "Parse error ({d}, {d}): expected '{s}' {s} but has Token {s}\n",
            .{ loc.line, loc.col, expected, context, cur_tok.str() },
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
                            total += count(v.asPtr(*ast.LiteralExprAST));
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
