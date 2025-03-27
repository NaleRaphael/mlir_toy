const std = @import("std");
const c = @import("toy/c_api.zig").c;
const ArgType = @import("argparse.zig").ArgType;

// Note that there are some misalignments between the following 2 impls.
// [1]: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/IR/AsmPrinter.cpp#L131-L171
// [2]: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/include/mlir/IR/OperationSupport.h#L1100-L1186
pub const ArgAsmPrinterOptions = struct {
    // XXX: this one is not configurable via `mlir::OpPrintingFlags` in LLVM 17.
    // https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/IR/AsmPrinter.cpp#L298-L311
    // mlir_print_elementsattrs_with_hex_if_larger: ArgType("--mlir-print-elementsattrs-with-hex-if-larger", i64, -1,
    //     \\Print DenseElementsAttrs with a hex string that have
    //     \\more elements than the given upper limit (use -1 to disable)
    // ),

    // XXX: we use -1 to disable this feature
    mlir_elide_elementsattrs_if_larger: ArgType(
        "--mlir-elide-elementsattrs-if-larger",
        i64,
        -1,
        "Elide ElementsAttrs with \"...\" that have" ++
            "more elements than the given upper limit (use -1 to disable)",
    ),
    mlir_print_debuginfo: ArgType(
        "--mlir-print-debuginfo",
        bool,
        false,
        "Print pretty debug info in MLIR output",
    ),
    mlir_pretty_debuginfo: ArgType(
        "--mlir-pretty-debuginfo",
        bool,
        false,
        "Print pretty debug info in MLIR output",
    ),
    mlir_print_op_generic: ArgType(
        "--mlir-print-op-generic",
        bool,
        false,
        "Print the generic op form",
    ),
    mlir_print_assume_verified: ArgType(
        "--mlir-print-assume-verified",
        bool,
        false,
        "Skip op verification when using custom printers",
    ),
    mlir_print_local_scope: ArgType(
        "--mlir-print-local-scope",
        bool,
        false,
        "Print with local scope and inline information (eliding " ++
            "aliases for attributes, types, and locations",
    ),
    mlir_print_value_users: ArgType(
        "--mlir-print-value-users",
        bool,
        false,
        "Print users of operation results and block arguments as a comment",
    ),
};

// ref: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/IR/MLIRContext.cpp#L58-L74
pub const ArgMLIRContextOptions = struct {
    mlir_disable_threading: ArgType(
        "--mlir-disable-threading",
        bool,
        false,
        "Disable multi-threading within MLIR, overrides any " ++
            "further call to MLIRContext::enableMultiThreading()",
    ),
    mlir_print_op_on_diagnostic: ArgType(
        "--mlir-print-op-on-diagnostic",
        bool,
        true,
        "When a diagnostic is emitted on an operation, also print " ++
            "the operation as an attached note",
    ),
    mlir_print_stacktrace_on_diagnostic: ArgType(
        "--mlir-print-stacktrace-on-diagnostic",
        bool,
        false,
        "When a diagnostic is emitted, also print the stack trace " ++
            "as an attached note",
    ),
};

// ref: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/Pass/PassManagerOptions.cpp#L19
pub const ArgPassManagerOptions = struct {
    mlir_pass_pipeline_crash_reproducer: ArgType(
        "--mlir-pass-pipeline-crash-reproducer",
        []const u8,
        "",
        "Generate a .mlir reproducer file at the given output path " ++
            "if the pass manager crashes or fails",
    ),
    mlir_pass_pipeline_local_reproducer: ArgType(
        "--mlir-pass-pipeline-local-reproducer",
        bool,
        false,
        "When generating a crash reproducer, attempt to generated " ++
            "a reproducer with the smallest pipeline.",
    ),
    // mlir_print_ir_before,
    // mlir_print_ir_after,

    mlir_print_ir_before_all: ArgType(
        "--mlir-print-ir-before-all",
        bool,
        false,
        "Print IR before each pass",
    ),
    mlir_print_ir_after_all: ArgType(
        "--mlir-print-ir-after-all",
        bool,
        false,
        "Print IR after each pass",
    ),
    mlir_print_ir_after_change: ArgType(
        "--mlir-print-ir-after-change",
        bool,
        false,
        "When printing the IR after a pass, only print if the IR changed",
    ),
    mlir_print_ir_after_failure: ArgType(
        "--mlir-print-ir-after-failure",
        bool,
        false,
        "When printing the IR after a pass, only print if the pass failed",
    ),
    mlir_print_ir_module_scope: ArgType(
        "--mlir-print-ir-module-scope",
        bool,
        false,
        "When printing IR for print-ir-[before|after]{-all} " ++
            "always print the top-level operation",
    ),
};

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

pub fn initOptions(comptime OptionType: type, args: anytype) OptionType {
    var opts = OptionType{};
    inline for (std.meta.fields(OptionType)) |f| {
        const arg = @field(args, f.name);
        @field(opts, f.name) = arg.value;
    }
    return opts;
}
