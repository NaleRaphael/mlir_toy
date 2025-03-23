const std = @import("std");
const ast = @import("toy/ast.zig");
const lexer = @import("toy/lexer.zig");
const parser = @import("toy/parser.zig");
const MLIRGen = @import("toy/MLIRGen.zig");
const c_api = @import("toy/c_api.zig");
const argparse = @import("argparse.zig");

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

pub const ArgTmpl = struct {
    file_path: ArgType("file_path", []const u8, "", "Input file"),
    input_type: ArgType("--input_type", InputType, InputType.toy, "Input type"),
    emit_action: ArgType("--emit", Action, Action.none, "Output kind"),

    // Some CLI options to help debugging (see also [1])
    // [1]: https://mlir.llvm.org/getting_started/Debugging/

    // ---------- AsmPrinterOptions ----------
    // Note that there are some misalignments between the following 2 impls.
    // [1]: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/IR/AsmPrinter.cpp#L131-L171
    // [2]: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/include/mlir/IR/OperationSupport.h#L1100-L1186

    // XXX: this one is not configurable via `mlir::OpPrintingFlags` in LLVM 17.
    // https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/IR/AsmPrinter.cpp#L298-L311
    // mlir_print_elementsattrs_with_hex_if_larger: ArgType("--mlir-print-elementsattrs-with-hex-if-larger", i64, -1,
    //     \\Print DenseElementsAttrs with a hex string that have
    //     \\more elements than the given upper limit (use -1 to disable)
    // ),

    // XXX: we use -1 to disable this feature
    mlir_elide_elementsattrs_if_larger: ArgType("--mlir-elide-elementsattrs-if-larger", i64, -1,
        \\Elide ElementsAttrs with "..." that have
        \\more elements than the given upper limit (use -1 to disable)
    ),
    mlir_print_debuginfo: ArgType("--mlir-print-debuginfo", bool, false,
        \\Print pretty debug info in MLIR output
    ),
    mlir_pretty_debuginfo: ArgType("--mlir-pretty-debuginfo", bool, false,
        \\Print pretty debug info in MLIR output
    ),
    mlir_print_op_generic: ArgType("--mlir-print-op-generic", bool, false,
        \\Print the generic op form
    ),
    mlir_print_assume_verified: ArgType("--mlir-print-assume-verified", bool, false,
        \\Skip op verification when using custom printers
    ),
    mlir_print_local_scope: ArgType("--mlir-print-local-scope", bool, false,
        \\Print with local scope and inline information (eliding
        \\aliases for attributes, types, and locations
    ),
    mlir_print_value_users: ArgType("--mlir-print-value-users", bool, false,
        \\Print users of operation results and block arguments as a comment
    ),

    // ---------- MLIRContextOptions ----------
    // ref: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/IR/MLIRContext.cpp#L58-L74
    mlir_disable_threading: ArgType("--mlir-disable-threading", bool, false,
        \\Disable multi-threading within MLIR, overrides any
        \\further call to MLIRContext::enableMultiThreading()
    ),
    mlir_print_op_on_diagnostic: ArgType("--mlir-print-op-on-diagnostic", bool, true,
        \\When a diagnostic is emitted on an operation, also print
        \\the operation as an attached note
    ),
    mlir_print_stacktrace_on_diagnostic: ArgType("--mlir-print-stacktrace-on-diagnostic", bool, false,
        \\When a diagnostic is emitted, also print the stack trace
        \\as an attached note
    ),
};

pub fn initOptions(comptime OptionType: type, args: ArgTmpl) OptionType {
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

pub fn dumpMLIRFromToy(
    file_path: []const u8,
    allocator: Allocator,
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

    const module = try mlirgen.fromModule(module_ast) orelse {
        return error.FailedToGenMLIR;
    };

    const module_op = c.mlirModuleGetOperation(module);
    c.mlirOperationPrintWithFlags(module_op, opflags, c_api.printToStderr, null);
}

pub fn dumpMLIRFromMLIR(
    file_path: []const u8,
    allocator: Allocator,
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

    var arg_parser = argparse.ArgumentParser(ArgTmpl).init("toyc-ch2");
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

    const asm_printer_opts = initOptions(AsmPrinterOptions, args);
    const mlir_context_opts = initOptions(MLIRContextOptions, args);

    switch (action) {
        Action.ast => try dumpAST(file_path, allocator),
        Action.mlir => {
            if (input_type != InputType.mlir and !std.mem.endsWith(u8, file_path, ".mlir")) {
                try dumpMLIRFromToy(file_path, allocator, mlir_context_opts, asm_printer_opts);
            } else {
                try dumpMLIRFromMLIR(file_path, allocator, mlir_context_opts, asm_printer_opts);
            }
        },
        Action.none => {
            std.debug.print("No action specified (parsing only?), use -emit=<action>\n", .{});
            std.process.exit(1);
        },
    }
}
