const std = @import("std");
const ast = @import("toy/ast.zig");
const lexer = @import("toy/lexer.zig");
const parser = @import("toy/parser.zig");
const MLIRGen = @import("toy/MLIRGen.zig");
const c_api = @import("toy/c_api.zig");
const argparse = @import("argparse.zig");
const common_options = @import("common_options.zig");

const c = c_api.c;
const Allocator = std.mem.Allocator;
const ArgType = argparse.ArgType;
const ArgParseError = argparse.ArgParseError;

pub const InputType = enum { toy, mlir };
pub const Action = enum { none, ast, mlir };

pub const AsmPrinterOptions = struct {
    // TODO: set this flag if we are using LLVM > 17.
    // mlir_print_elementsattrs_with_hex_if_larger: i64 = -1,
    mlir_elide_elementsattrs_if_larger: i64 = -1,
    mlir_print_debuginfo: bool = false,
    mlir_pretty_debuginfo: bool = false,
    mlir_print_op_generic: bool = false,
    mlir_print_assume_verified: bool = false,
    mlir_print_local_scope: bool = false,
    mlir_print_value_users: bool = false,

    pub fn config(self: @This(), flags: c.MlirOpPrintingFlags) void {
        // if (self.mlir_print_elementsattrs_with_hex_if_larger > 0) {
        //     // TODO: set this flag if we are using LLVM > 17.
        // }
        if (self.mlir_elide_elementsattrs_if_larger > 0) {
            c.mlirOpPrintingFlagsElideLargeElementsAttrs(
                flags,
                self.mlir_elide_elementsattrs_if_larger,
            );
        }
        if (self.mlir_print_op_generic) {
            c.mlirOpPrintingFlagsPrintGenericOpForm(flags);
        }
        c.mlirOpPrintingFlagsEnableDebugInfo(
            flags,
            self.mlir_print_debuginfo,
            self.mlir_pretty_debuginfo,
        );
        if (self.mlir_print_assume_verified) {
            c.mlirOpPrintingFlagsAssumeVerified(flags);
        }
        if (self.mlir_print_local_scope) {
            c.mlirOpPrintingFlagsUseLocalScope(flags);
        }
        if (self.mlir_print_value_users) {
            c.mlirExtOpPrintingFlagsPrintValueUsers(flags);
        }
    }
};

pub const MLIRContextOptions = struct {
    mlir_disable_threading: bool = false,
    mlir_print_op_on_diagnostic: bool = true,
    mlir_print_stacktrace_on_diagnostic: bool = false,

    pub fn config(self: @This(), ctx: c.MlirContext) void {
        c.mlirExtContextPrintOpOnDiagnostic(
            ctx,
            self.mlir_print_op_on_diagnostic,
        );
        c.mlirExtContextSetPrintStackTraceOnDiagnostic(
            ctx,
            self.mlir_print_stacktrace_on_diagnostic,
        );
    }
};

pub const PassManagerOptions = struct {
    // XXX: these 2 options are not used for now. But we can implement a C-API
    // to enable related features with `enableCrashReproducerGeneration()`.
    mlir_pass_pipeline_crash_reproducer: []const u8 = "",
    mlir_pass_pipeline_local_reproducer: bool = false,

    // mlir_print_ir_before: []const u8 = "",
    // mlir_print_ir_after: []const u8 = "",

    // NOTE: when enabling these flags, make sure `--mlir-disable-threading=true` is also set.
    mlir_print_ir_before_all: bool = false,
    mlir_print_ir_after_all: bool = false,
    mlir_print_ir_after_change: bool = false,
    mlir_print_ir_after_failure: bool = false,
    mlir_print_ir_module_scope: bool = false,

    pub fn config(self: @This(), pm: c.MlirPassManager, flags: c.MlirOpPrintingFlags) void {
        c.mlirExtPassManagerEnableIRPrinting(
            pm,
            self.mlir_print_ir_before_all,
            self.mlir_print_ir_after_all,
            self.mlir_print_ir_module_scope,
            self.mlir_print_ir_after_change,
            self.mlir_print_ir_after_failure,
            flags,
        );
    }
};

pub const ArgTmpl = struct {
    file_path: ArgType("file_path", []const u8, "", "Input file"),
    input_type: ArgType("--input_type", InputType, InputType.toy, "Input type"),
    emit_action: ArgType("--emit", Action, Action.none, "Output kind"),
    enable_opt: ArgType("--opt", bool, false, "Enable optimizations"),
};

pub const CLIOptions: type = mergeOptions(&.{
    ArgTmpl,
    common_options.ArgAsmPrinterOptions,
    common_options.ArgMLIRContextOptions,
    common_options.ArgPassManagerOptions,
});

pub fn mergeOptions(comptime opt_types: []const type) type {
    return comptime blk: {
        var num_fields: usize = 0;
        for (opt_types) |t| {
            const ti = @typeInfo(t);
            num_fields += ti.Struct.fields.len;
        }

        var fields: [num_fields]std.builtin.Type.StructField = undefined;
        var i: usize = 0;
        for (opt_types) |t| {
            const ti = @typeInfo(t);
            for (ti.Struct.fields) |f| {
                fields[i] = f;
                i += 1;
            }
        }

        const merged_type = @Type(std.builtin.Type{ .Struct = .{
            .layout = .auto,
            .fields = &fields,
            .decls = &.{},
            .is_tuple = false,
            .backing_integer = null,
        } });
        break :blk merged_type;
    };
}

pub fn initOptions(comptime OptionType: type, args: CLIOptions) OptionType {
    var opts = OptionType{};
    inline for (std.meta.fields(OptionType)) |f| {
        const arg = @field(args, f.name);
        @field(opts, f.name) = arg.value;
    }
    return opts;
}

pub fn parseInputFile(file_path: []const u8, allocator: Allocator) !*ast.ModuleAST {
    var _lexer = try lexer.Lexer.init(file_path);
    var _parser = parser.Parser.init(&_lexer, allocator);
    return try _parser.parseModule();
}

pub fn dumpAST(file_path: []const u8, allocator: Allocator) !void {
    var module_ast = try parseInputFile(file_path, allocator);
    defer module_ast.deinit();

    var ast_dumper = try ast.ASTDumper.init(allocator, 1024);
    defer ast_dumper.deinit();

    try ast_dumper.dump(module_ast);
}

pub const MLIRContextHolder = struct {
    ctx: c.MlirContext,
    opflags: c.MlirOpPrintingFlags,

    const Self = @This();

    pub fn init(
        mlir_context_opts: MLIRContextOptions,
        asm_printer_opts: AsmPrinterOptions,
    ) Self {
        // NOTE: multithreading supports is enabled by default
        const ctx = c.mlirContextCreateWithThreading(!mlir_context_opts.mlir_disable_threading);
        const opflags = c.mlirOpPrintingFlagsCreate();

        mlir_context_opts.config(ctx);
        asm_printer_opts.config(opflags);

        return Self{ .ctx = ctx, .opflags = opflags };
    }

    pub fn deinit(self: *Self) void {
        c.mlirContextDestroy(self.ctx);
        c.mlirOpPrintingFlagsDestroy(self.opflags);
    }
};

pub fn dumpMLIRFromToy(
    file_path: []const u8,
    allocator: Allocator,
    holder: MLIRContextHolder,
    enable_opt: bool,
    pm_opts: PassManagerOptions,
) !void {
    const ctx = holder.ctx;
    const opflags = holder.opflags;

    // Remember to load Toy dialect
    try c_api.loadToyDialect(ctx);

    var module_ast = try parseInputFile(file_path, allocator);
    defer module_ast.deinit();

    var mlirgen = MLIRGen.init(ctx, allocator);
    defer mlirgen.deinit();

    const module = try mlirgen.fromModule(module_ast) orelse {
        return error.FailedToGenMLIR;
    };

    const module_op = c.mlirModuleGetOperation(module);

    if (enable_opt) {
        const name = c.mlirIdentifierStr(c.mlirOperationGetName(module_op));
        const pm = c.mlirPassManagerCreateOnOperation(ctx, name);
        defer c.mlirPassManagerDestroy(pm);

        const opm_toyfunc = c.mlirPassManagerGetNestedUnder(
            pm,
            c.mlirStringRefCreateFromCString("toy.func"),
        );

        pm_opts.config(pm, opflags);

        // `mlirCreateTransformsCanonicalizer` is defined in "mlir-c/Transform.h"
        c.mlirOpPassManagerAddOwnedPass(
            opm_toyfunc,
            c.mlirCreateTransformsCanonicalizer(),
        );

        const result = c.mlirPassManagerRunOnOp(pm, module_op);
        if (c.mlirLogicalResultIsFailure(result)) {
            std.debug.print("failed to run canonicalizer pass\n", .{});
        }
    }

    c.mlirOperationPrintWithFlags(module_op, opflags, c_api.printToStderr, null);
}

pub fn dumpMLIRFromMLIR(
    file_path: []const u8,
    allocator: Allocator,
    holder: MLIRContextHolder,
    enable_opt: bool,
    pm_opts: PassManagerOptions,
) !void {
    _ = allocator;

    const ctx = holder.ctx;
    const opflags = holder.opflags;

    try c_api.loadToyDialect(ctx);

    const fp_strref = c.mlirStringRefCreate(file_path.ptr, file_path.len);
    const module_op = c.mlirExtParseSourceFileAsModuleOp(ctx, fp_strref);
    if (c.mlirOperationIsNull(module_op)) {
        return error.FailedToParseMLIR;
    }

    if (enable_opt) {
        const name = c.mlirIdentifierStr(c.mlirOperationGetName(module_op));
        const pm = c.mlirPassManagerCreateOnOperation(ctx, name);
        defer c.mlirPassManagerDestroy(pm);

        const opm_toyfunc = c.mlirPassManagerGetNestedUnder(
            pm,
            c.mlirStringRefCreateFromCString("toy.func"),
        );

        pm_opts.config(pm, opflags);

        c.mlirOpPassManagerAddOwnedPass(
            opm_toyfunc,
            c.mlirCreateTransformsCanonicalizer(),
        );

        const result = c.mlirPassManagerRunOnOp(pm, module_op);
        if (c.mlirLogicalResultIsFailure(result)) {
            std.debug.print("failed to run canonicalizer pass\n", .{});
        }
    }

    c.mlirOperationPrintWithFlags(module_op, opflags, c_api.printToStderr, null);
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);

    var arg_parser = argparse.ArgumentParser(CLIOptions).init("toyc-ch3");
    const args = arg_parser.parse(argv) catch |err| switch (err) {
        ArgParseError.EndWithPrintingHelp => std.process.exit(0),
        else => {
            arg_parser.printHelp();
            std.process.exit(1);
        },
    };

    const file_path = args.file_path.value;
    const input_type = args.input_type.value;
    const action = args.emit_action.value;
    const enable_opt = args.enable_opt.value;

    const asm_printer_opts = initOptions(AsmPrinterOptions, args);
    const mlir_context_opts = initOptions(MLIRContextOptions, args);
    const pass_manager_opts = initOptions(PassManagerOptions, args);

    switch (action) {
        Action.ast => try dumpAST(file_path, allocator),
        Action.mlir => {
            var ctx_holder = MLIRContextHolder.init(mlir_context_opts, asm_printer_opts);
            defer ctx_holder.deinit();

            if (input_type != InputType.mlir and !std.mem.endsWith(u8, file_path, ".mlir")) {
                try dumpMLIRFromToy(file_path, allocator, ctx_holder, enable_opt, pass_manager_opts);
            } else {
                try dumpMLIRFromMLIR(file_path, allocator, ctx_holder, enable_opt, pass_manager_opts);
            }
        },
        Action.none => {
            std.debug.print("No action specified (parsing only?), use -emit=<action>\n", .{});
            std.process.exit(1);
        },
    }
}
