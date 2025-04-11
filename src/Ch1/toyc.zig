const std = @import("std");
const ast = @import("toy/ast.zig");
const lexer = @import("toy/lexer.zig");
const parser = @import("toy/parser.zig");
const argparse = @import("argparse.zig");

const ArgType = argparse.ArgType;
const ArgParseError = argparse.ArgParseError;

pub const Action = enum { none, ast };

pub fn createParser(file_path: []const u8, allocator: std.mem.Allocator) !*parser.Parser {
    const _lexer = try lexer.Lexer.init(file_path, allocator);
    return try parser.Parser.init(_lexer, allocator);
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);

    const ArgTmpl = struct {
        file_path: ArgType("file_path", []const u8, "", "Input file"),
        emit_action: ArgType("--emit", Action, Action.none, "Ouput kind"),
    };

    var arg_parser = argparse.ArgumentParser(ArgTmpl).init("toyc-ch1");
    const args = arg_parser.parse(argv) catch |err| switch (err) {
        ArgParseError.EndWithPrintingHelp => std.process.exit(0),
        else => {
            arg_parser.printHelp();
            std.process.exit(1);
        },
    };

    const file_path = args.file_path.value;
    const action = args.emit_action.value;

    // XXX: check this earlier to avoid unnecessary work.
    if (action == Action.none) {
        std.debug.print("No action specified (parsing only?), use -emit=<action>\n", .{});
        std.process.exit(1);
    }

    var _parser = try createParser(file_path, allocator);
    defer _parser.deinit();

    var module_ast = try _parser.parseModule();
    defer module_ast.deinit();

    var ast_dumper = try ast.ASTDumper.init(allocator, 1024);
    defer ast_dumper.deinit();

    try ast_dumper.dump(module_ast);
}
