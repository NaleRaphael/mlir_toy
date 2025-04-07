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
pub const Action = enum { none, ast, mlir, mlir_affine, mlir_llvm, llvm, jit };

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
    registry: c.MlirDialectRegistry,
    opflags: c.MlirOpPrintingFlags,

    const Self = @This();

    pub fn init(
        mlir_context_opts: MLIRContextOptions,
        asm_printer_opts: AsmPrinterOptions,
    ) Self {
        // NOTE: multithreading supports is enabled by default
        const ctx = c.mlirContextCreateWithThreading(!mlir_context_opts.mlir_disable_threading);
        const registry = c.mlirDialectRegistryCreate();
        const opflags = c.mlirOpPrintingFlagsCreate();

        mlir_context_opts.config(ctx);
        asm_printer_opts.config(opflags);

        return Self{ .ctx = ctx, .registry = registry, .opflags = opflags };
    }

    pub fn deinit(self: *Self) void {
        c.mlirContextDestroy(self.ctx);
        c.mlirDialectRegistryDestroy(self.registry);
        c.mlirOpPrintingFlagsDestroy(self.opflags);
    }

    pub fn registerAllUpstreamDialects(self: *Self) void {
        c.mlirRegisterAllDialects(self.registry);
        c.mlirContextAppendDialectRegistry(self.ctx, self.registry);
    }

    pub fn registerAndLoadDialect(self: *Self, comptime name: []const u8) void {
        const func = @field(c, "mlirGetDialectHandle__" ++ name ++ "__");
        const handle = @call(.auto, func, .{});
        c.mlirDialectHandleRegisterDialect(handle, self.ctx);
        const dialect = c.mlirContextGetOrLoadDialect(self.ctx, z2strref(name));
        std.debug.assert(!c.mlirDialectIsNull(dialect));
    }
};

pub fn parseInputFile(file_path: []const u8, allocator: Allocator) !*ast.ModuleAST {
    var _lexer = try lexer.Lexer.init(file_path);
    var _parser = parser.Parser.init(&_lexer, allocator);
    return try _parser.parseModule();
}

pub fn readMLIRFromToy(
    allocator: Allocator,
    mlirgen: *MLIRGen,
    file_path: []const u8,
) !c.MlirOperation {
    var module_ast = try parseInputFile(file_path, allocator);
    defer module_ast.deinit();

    const module = try mlirgen.fromModule(module_ast) orelse {
        return error.FailedToGenMLIR;
    };

    const module_op = c.mlirModuleGetOperation(module);
    std.debug.assert(!c.mlirOperationIsNull(module_op));
    return module_op;
}

pub fn readMLIRFromMLIR(
    file_path: []const u8,
    manager: MLIRContextManager,
) !c.MlirOperation {
    const ctx = manager.ctx;
    const fp_strref = z2strref(file_path);
    const module_op = c.mlirExtParseSourceFileAsModuleOp(ctx, fp_strref);
    if (c.mlirOperationIsNull(module_op)) {
        return error.FailedToParseMLIR;
    }

    return module_op;
}

pub fn processMLIR(
    manager: MLIRContextManager,
    module_op: c.MlirOperation,
    pm_opts: PassManagerOptions,
    enable_opt: bool,
    action: Action,
) !void {
    const ctx = manager.ctx;
    const opflags = manager.opflags;

    const name = c.mlirIdentifierStr(c.mlirOperationGetName(module_op));
    const pm = c.mlirPassManagerCreateOnOperation(ctx, name);
    defer c.mlirPassManagerDestroy(pm);

    pm_opts.config(pm, opflags);

    const is_lowering_to_affine = @intFromEnum(action) >= @intFromEnum(Action.mlir_affine);
    const is_lowering_to_llvm = @intFromEnum(action) >= @intFromEnum(Action.mlir_llvm);

    // NOTE: we only need to emit C interface when we need to invoke functions
    // via execution engine's C-API. Otherwise, we just need to compare the
    // emitted IR.
    const emit_c_interface = @intFromEnum(action) >= @intFromEnum(Action.jit);

    if (enable_opt or is_lowering_to_affine) {
        // Inliner pass (added root module)
        c.mlirPassManagerAddOwnedPass(
            pm,
            c.mlirCreateTransformsInliner(),
        );

        const opm_toyfunc = c.mlirPassManagerGetNestedUnder(pm, strref("toy.func"));

        // ShapeInference, canonicalizer, CSE pass (for toy.funcs)
        // Note: passes like `mlirCreateTransformsXXX` are defined in
        // -> "mlir-c/Transform.h"
        // -> "mlir/Transforms/Transforms.capi.h.inc" (generated after build)
        c.mlirOpPassManagerAddOwnedPass(
            opm_toyfunc,
            c.mlirToyCreateShapeInferencePass(),
        );
        c.mlirOpPassManagerAddOwnedPass(
            opm_toyfunc,
            c.mlirCreateTransformsCanonicalizer(),
        );
        c.mlirOpPassManagerAddOwnedPass(
            opm_toyfunc,
            c.mlirCreateTransformsCSE(),
        );
    }

    if (is_lowering_to_affine) {
        // Partially lower the toy dialect.
        c.mlirPassManagerAddOwnedPass(pm, c.mlirToyCreateLowerToAffinePass());

        // Add a few cleanups post lowering
        const opm_funcfunc = c.mlirPassManagerGetNestedUnder(pm, strref("func.func"));
        c.mlirOpPassManagerAddOwnedPass(
            opm_funcfunc,
            c.mlirCreateTransformsCanonicalizer(),
        );
        c.mlirOpPassManagerAddOwnedPass(
            opm_funcfunc,
            c.mlirCreateTransformsCSE(),
        );

        // NOTE: this is required only if we are going to invoke function via
        // execution engine's C-API.
        if (emit_c_interface) {
            c.mlirOpPassManagerAddOwnedPass(
                opm_funcfunc,
                c.mlirExtLLVMCreateRequestCWrappersPass(),
            );
        }

        if (enable_opt) {
            c.mlirOpPassManagerAddOwnedPass(
                opm_funcfunc,
                c.mlirExtAffineCreateLoopFusionPass(),
            );
            c.mlirOpPassManagerAddOwnedPass(
                opm_funcfunc,
                c.mlirExtAffineCreateAffineScalarReplacementPass(),
            );
        }
    }

    if (is_lowering_to_llvm) {
        c.mlirPassManagerAddOwnedPass(pm, c.mlirToyCreateLowerToLLVMPass());

        const opm_llvmfunc = c.mlirPassManagerGetNestedUnder(pm, strref("llvm.func"));
        c.mlirOpPassManagerAddOwnedPass(
            opm_llvmfunc,
            c.mlirExtLLVMCreateDIScopeForLLVMFuncOpPass(),
        );
    }

    const result = c.mlirPassManagerRunOnOp(pm, module_op);
    if (c.mlirLogicalResultIsFailure(result)) {
        return error.FailedToProcessMLIR;
    }
}

pub fn dumpAST(file_path: []const u8, allocator: Allocator) !void {
    var module_ast = try parseInputFile(file_path, allocator);
    defer module_ast.deinit();

    var ast_dumper = try ast.ASTDumper.init(allocator, 1024);
    defer ast_dumper.deinit();

    try ast_dumper.dump(module_ast);
}

pub fn dumpLLVMIR(
    module_op: c.MlirOperation,
    manager: MLIRContextManager,
    enable_opt: bool,
) !void {
    const ctx = manager.ctx;

    c.mlirExtRegisterBuiltinDialectTranslation(ctx);
    c.mlirExtRegisterLLVMDialectTranslation(ctx);

    const llvm_ctx = c.LLVMContextCreate();
    defer c.LLVMContextDispose(llvm_ctx);

    const llvm_module = c.mlirExtTranslateModuleToLLVMIR(module_op, llvm_ctx);
    if (c.llvmExtModuleIsNull(llvm_module)) {
        return error.FailedToEmitLLVMIR;
    }

    // Initialize LLVM targets.
    _ = c.LLVMInitializeNativeTarget();
    _ = c.LLVMInitializeNativeAsmPrinter();

    // ----- Get data layout and target triple by existing ORC JIT APIs -----
    // 1. Get target triple
    // 1-1. Create a target machine builder by `llvm::orc::JITTargetMachineBuilder::detectHost()`
    // We want to make use of it to determine the target, so that we don't
    // have to configure it manually. See also:
    // https://github.com/llvm/llvm-project/blob/release/17.x/llvm/lib/ExecutionEngine/Orc/JITTargetMachineBuilder.cpp#L24-L38
    var tm_builder: c.LLVMOrcJITTargetMachineBuilderRef = null;
    if (c.LLVMOrcJITTargetMachineBuilderDetectHost(&tm_builder)) |err| {
        // NOTE: the returned value `LLVMErrorRef` would be null if the
        // execution is successful. Otherwise, we have to consume the error
        // properly to prevent memory leak.
        // https://llvm.org/docs/doxygen/group__LLVMCError.html#gad81d81a316ef38888533a24b786a6605
        const msg = c.LLVMGetErrorMessage(err);
        defer c.LLVMDisposeMessage(msg);
        std.debug.print("Could not create JITTargetMachineBuilder, reason: {s}\n", .{msg});
        return error.FailedToPrepareOptimizationPipeline;
    }
    defer c.LLVMOrcDisposeJITTargetMachineBuilder(tm_builder);

    // 1-2. Get target tripe from the target machine builder
    const triple = c.LLVMOrcJITTargetMachineBuilderGetTargetTriple(tm_builder);

    // 2. Get data layout
    // 2-1. Create a `llvm::Target` from triple
    var target: c.LLVMTargetRef = null;
    var err_msg: [*c]u8 = null;
    if (c.LLVMGetTargetFromTriple(triple, &target, &err_msg) != 0) {
        std.debug.print("Failed to get target from triple, reason: {s}\n", .{err_msg});
        c.LLVMDisposeMessage(err_msg);
        return error.FailedToPrepareOptimizationPipeline;
    }

    // 2-2. Get CPU and CPU features for later use
    // (but these can be omitted without altering the generated IR)
    const cpu = c.LLVMGetHostCPUName();
    defer c.LLVMDisposeMessage(cpu);
    const cpu_features = c.LLVMGetHostCPUFeatures();
    defer c.LLVMDisposeMessage(cpu_features);

    // 2-3. Create a `llvm::TargetMachine`, and we will get data layout from it
    // later. Note that all configurations except CPU and CPU features will be
    // kept to default values if we didn't set it explicitly. (it's mentioned
    // in [1])
    // See also [3] for details of this API and related arguments.
    // [1]: https://github.com/llvm/llvm-project/blob/release/17.x/llvm/lib/ExecutionEngine/Orc/JITTargetMachineBuilder.cpp#L40-L60
    // [2]: https://github.com/llvm/llvm-project/blob/release/17.x/llvm/unittests/ExecutionEngine/Orc/JITTargetMachineBuilderTest.cpp
    // [3]: https://github.com/llvm/llvm-project/blob/release/17.x/llvm/include/llvm-c/TargetMachine.h
    const tm: c.LLVMTargetMachineRef = c.LLVMCreateTargetMachine(
        target,
        triple,
        cpu,
        cpu_features,
        0, // LLVMCodeGenOptLevel (0: Default)
        0, // LLVMRelocMode (0: Default)
        0, // LLVMCodeModel (0: Default)
    );
    defer c.LLVMDisposeTargetMachine(tm);

    // 2-4. Get data layout as string
    const data_layout = c.LLVMCreateTargetDataLayout(tm);
    defer c.LLVMDisposeTargetData(data_layout);
    const data_layout_str = c.LLVMCopyStringRepOfTargetData(data_layout);
    defer c.LLVMDisposeMessage(data_layout_str);

    // ----- Set data layout and target to the `llvm::Module` -----
    // These are also what `mlir::ExecutionEngine::setupTargetTripleAndDataLayout()` does:
    c.LLVMSetDataLayout(llvm_module, data_layout_str);
    c.LLVMSetTarget(llvm_module, triple);

    // ----- Run the optimization pipeline -----
    // Since the pipeline is created as a C++ function, we cannot do it from
    // Zig side. So we add a custom C-API for it.
    const result = c.mlirExtOptimizeLLVMModule(llvm_module, enable_opt);
    if (result == -1) {
        return error.FailedToOptimizeLLVMModule;
    }

    // XXX: if we use `LLVMDumpModule()`, there would be some placeholder like
    // "%0" showing up in the IR, so we use this approach instead.
    const module_str = c.LLVMPrintModuleToString(llvm_module);
    defer c.LLVMDisposeMessage(module_str);
    std.debug.print("{s}\n", .{module_str});
}

// NOTE: to invoke function via execution engine's C-API, please make sure
// every builtin function ("func.func") has "llvm.emit_c_interface" in
// attribute before lowering to LLVM IR. The attribute can be attached via
// adding the pass `mlir::LLVM::createRequestCWrappersPass()`.
pub fn runJit(
    module_op: c.MlirOperation,
    manager: MLIRContextManager,
    enable_opt: bool,
) !void {
    const ctx = manager.ctx;
    const module = c.mlirModuleFromOperation(module_op);

    _ = c.LLVMInitializeNativeTarget();
    _ = c.LLVMInitializeNativeAsmPrinter();

    c.mlirExtRegisterBuiltinDialectTranslation(ctx);
    c.mlirExtRegisterLLVMDialectTranslation(ctx);

    const opt_level: c_int = if (enable_opt) 3 else 0;
    const jit = c.mlirExecutionEngineCreate(
        module,
        opt_level,
        0, // numPaths (number of shared libraries to link)
        null, // sharedLibPaths
        false, // enableObjectDump
    );
    defer c.mlirExecutionEngineDestroy(jit);
    std.debug.assert(!c.mlirExecutionEngineIsNull(jit));

    const result = c.mlirExecutionEngineInvokePacked(jit, strref("main"), null);
    std.debug.assert(!c.mlirLogicalResultIsFailure(result));
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);

    var arg_parser = argparse.ArgumentParser(CLIOptions).init("toyc-ch7");
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
        Action.mlir, Action.mlir_affine, Action.mlir_llvm, Action.llvm, Action.jit => {
            var manager = MLIRContextManager.init(mlir_context_opts, asm_printer_opts);
            defer manager.deinit();

            manager.registerAllUpstreamDialects();

            // Remember to load Toy dialect
            try c_api.loadToyDialect(manager.ctx);

            // XXX: if we don't want to register all dialects at once, we need
            // to figure out what else dialects need to load.
            // manager.registerAndLoadDialect("llvm");

            var mlirgen = MLIRGen.init(manager.ctx, allocator);
            defer mlirgen.deinit();

            var module_op: c.MlirOperation = undefined;
            if (input_type != InputType.mlir and !std.mem.endsWith(u8, file_path, ".mlir")) {
                module_op = try readMLIRFromToy(allocator, &mlirgen, file_path);
            } else {
                module_op = try readMLIRFromMLIR(file_path, manager);
            }

            try processMLIR(manager, module_op, pass_manager_opts, enable_opt, action);

            switch (action) {
                Action.mlir, Action.mlir_affine, Action.mlir_llvm => {
                    const opflags = manager.opflags;
                    c.mlirOperationPrintWithFlags(module_op, opflags, c_api.printToStderr, null);
                },
                Action.llvm => {
                    try dumpLLVMIR(module_op, manager, enable_opt);
                },
                Action.jit => {
                    try runJit(module_op, manager, enable_opt);
                },
                else => unreachable,
            }
        },
        Action.none => {
            std.debug.print("No action specified (parsing only?), use -emit=<action>\n", .{});
            std.process.exit(1);
        },
    }
}
