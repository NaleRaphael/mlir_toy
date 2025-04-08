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

pub const InputType = enum { toy, mlir };
pub const Action = enum { none, ast, mlir };

pub const ArgTmpl = struct {
    file_path: ArgType("file_path", []const u8, "", "Input file"),
    input_type: ArgType("--input_type", InputType, InputType.toy, "Input type"),
    emit_action: ArgType("--emit", Action, Action.none, "Output kind"),
};

pub const CLIOptions: type = com_opts.mergeOptions(&.{
    ArgTmpl,
    com_opts.ArgAsmPrinterOptions,
    com_opts.ArgMLIRContextOptions,
});

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
    mlir_context_opts: MLIRContextOptions,
    asm_printer_opts: AsmPrinterOptions,
) !void {
    // NOTE: multithreading supports is enabled by default
    const ctx = c.mlirContextCreateWithThreading(!mlir_context_opts.mlir_disable_threading);
    defer c.mlirContextDestroy(ctx);

    const opflags = c.mlirOpPrintingFlagsCreate();
    defer c.mlirOpPrintingFlagsDestroy(opflags);

    mlir_context_opts.config(ctx);
    asm_printer_opts.config(opflags);

    // Remember to load Toy dialect
    try c_api.loadToyDialect(ctx);

    var module_ast = try parseInputFile(file_path, allocator);
    defer module_ast.deinit();

    var mlirgen = MLIRGen.init(ctx, allocator);
    defer mlirgen.deinit();

    const module = mlirgen.fromModule(module_ast) catch {
        return error.FailedToGenMLIR;
    };

    const module_op = c.mlirModuleGetOperation(module);
    c.mlirOperationPrintWithFlags(module_op, opflags, c_api.printToStderr, null);
}

pub fn dumpMLIRFromMLIR(
    allocator: Allocator,
    file_path: []const u8,
    mlir_context_opts: MLIRContextOptions,
    asm_printer_opts: AsmPrinterOptions,
) !void {
    _ = allocator;
    const ctx = c.mlirContextCreateWithThreading(!mlir_context_opts.mlir_disable_threading);
    defer c.mlirContextDestroy(ctx);

    const opflags = c.mlirOpPrintingFlagsCreate();
    defer c.mlirOpPrintingFlagsDestroy(opflags);

    mlir_context_opts.config(ctx);
    asm_printer_opts.config(opflags);

    try c_api.loadToyDialect(ctx);

    const fp_strref = c.mlirStringRefCreate(file_path.ptr, file_path.len);
    const module_op = c.mlirExtParseSourceFileAsModuleOp(ctx, fp_strref);
    if (c.mlirOperationIsNull(module_op)) {
        return error.FailedToParseMLIR;
    }

    c.mlirOperationPrintWithFlags(module_op, opflags, c_api.printToStderr, null);
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);

    var arg_parser = argparse.ArgumentParser(CLIOptions).init("toyc-ch2");
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

    const asm_printer_opts = com_opts.initOptions(AsmPrinterOptions, args);
    const mlir_context_opts = com_opts.initOptions(MLIRContextOptions, args);

    switch (action) {
        Action.ast => try dumpAST(file_path, allocator),
        Action.mlir => {
            if (input_type != InputType.mlir and !std.mem.endsWith(u8, file_path, ".mlir")) {
                try dumpMLIRFromToy(allocator, file_path, mlir_context_opts, asm_printer_opts);
            } else {
                try dumpMLIRFromMLIR(allocator, file_path, mlir_context_opts, asm_printer_opts);
            }
        },
        Action.none => {
            std.debug.print("No action specified (parsing only?), use -emit=<action>\n", .{});
            std.process.exit(1);
        },
    }
}
