const std = @import("std");
const ast = @import("toy/ast.zig");
const lexer = @import("toy/lexer.zig");
const parser = @import("toy/parser.zig");
const MLIRGen = @import("toy/MLIRGen.zig");
const c_api = @import("toy/c_api.zig");
const argparse = @import("argparse.zig");
const com_opts = @import("common_options.zig");

const c = c_api.c;
const Allocator = std.mem.Allocator;
const ArgType = argparse.ArgType;
const ArgParseError = argparse.ArgParseError;

const AsmPrinterOptions = com_opts.AsmPrinterOptions;
const MLIRContextOptions = com_opts.MLIRContextOptions;
const PassManagerOptions = com_opts.PassManagerOptions;

const strref = c.mlirStringRefCreateFromCString;

fn z2strref(src: []const u8) c.MlirStringRef {
    return c.mlirStringRefCreate(src.ptr, src.len);
}

pub const InputType = enum { toy, mlir };
pub const Action = enum { none, ast, mlir };

pub const ArgTmpl = struct {
    file_path: ArgType("file_path", []const u8, "", "Input file"),
    input_type: ArgType("--input_type", InputType, InputType.toy, "Input type"),
    emit_action: ArgType("--emit", Action, Action.none, "Output kind"),
    enable_opt: ArgType("--opt", bool, false, "Enable optimizations"),
};

pub const CLIOptions: type = com_opts.mergeOptions(&.{
    ArgTmpl,
    com_opts.ArgAsmPrinterOptions,
    com_opts.ArgMLIRContextOptions,
    com_opts.ArgPassManagerOptions,
});

pub const MLIRContextManager = struct {
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

pub fn dumpMLIRFromToy(
    allocator: Allocator,
    file_path: []const u8,
    manager: MLIRContextManager,
    enable_opt: bool,
    pm_opts: PassManagerOptions,
) !void {
    const ctx = manager.ctx;
    const opflags = manager.opflags;

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
        try processMLIR(manager, module_op, pm_opts);
    }

    c.mlirOperationPrintWithFlags(module_op, opflags, c_api.printToStderr, null);
}

pub fn dumpMLIRFromMLIR(
    _: Allocator,
    file_path: []const u8,
    manager: MLIRContextManager,
    enable_opt: bool,
    pm_opts: PassManagerOptions,
) !void {
    const ctx = manager.ctx;
    const opflags = manager.opflags;

    try c_api.loadToyDialect(ctx);

    const fp_strref = z2strref(file_path);
    const module_op = c.mlirExtParseSourceFileAsModuleOp(ctx, fp_strref);
    if (c.mlirOperationIsNull(module_op)) {
        return error.FailedToParseMLIR;
    }

    if (enable_opt) {
        try processMLIR(manager, module_op, pm_opts);
    }

    c.mlirOperationPrintWithFlags(module_op, opflags, c_api.printToStderr, null);
}

pub fn processMLIR(
    manager: MLIRContextManager,
    module_op: c.MlirOperation,
    pm_opts: PassManagerOptions,
) !void {
    const ctx = manager.ctx;
    const opflags = manager.opflags;

    const name = c.mlirIdentifierStr(c.mlirOperationGetName(module_op));
    const pm = c.mlirPassManagerCreateOnOperation(ctx, name);
    defer c.mlirPassManagerDestroy(pm);

    pm_opts.config(pm, opflags);

    const opm_toyfunc = c.mlirPassManagerGetNestedUnder(pm, strref("toy.func"));

    // Note: passes like `mlirCreateTransformsXXX` are defined in
    // -> "mlir-c/Transform.h"
    // -> "mlir/Transforms/Transforms.capi.h.inc" (generated after build)
    c.mlirOpPassManagerAddOwnedPass(
        opm_toyfunc,
        c.mlirCreateTransformsCanonicalizer(),
    );

    const result = c.mlirPassManagerRunOnOp(pm, module_op);
    if (c.mlirLogicalResultIsFailure(result)) {
        std.debug.print("failed to run canonicalizer pass\n", .{});
        return error.FailedToProcessMLIR;
    }
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

    const asm_printer_opts = com_opts.initOptions(AsmPrinterOptions, args);
    const mlir_context_opts = com_opts.initOptions(MLIRContextOptions, args);
    const pass_manager_opts = com_opts.initOptions(PassManagerOptions, args);

    switch (action) {
        Action.ast => try dumpAST(file_path, allocator),
        Action.mlir => {
            var manager = MLIRContextManager.init(mlir_context_opts, asm_printer_opts);
            defer manager.deinit();

            if (input_type != InputType.mlir and !std.mem.endsWith(u8, file_path, ".mlir")) {
                try dumpMLIRFromToy(allocator, file_path, manager, enable_opt, pass_manager_opts);
            } else {
                try dumpMLIRFromMLIR(allocator, file_path, manager, enable_opt, pass_manager_opts);
            }
        },
        Action.none => {
            std.debug.print("No action specified (parsing only?), use -emit=<action>\n", .{});
            std.process.exit(1);
        },
    }
}
