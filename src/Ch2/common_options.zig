const ArgType = @import("argparse.zig").ArgType;

// ---------- AsmPrinterOptions ----------
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
};

// ---------- MLIRContextOptions ----------
// ref: https://github.com/llvm/llvm-project/blob/release/17.x/mlir/lib/IR/MLIRContext.cpp#L58-L74
pub const ArgMLIRContextOptions = struct {
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
