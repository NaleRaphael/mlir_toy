const std = @import("std");

test {
    _ = @import("execution_engine.zig");
    _ = @import("ir.zig");
    _ = @import("_llvm.zig");
    _ = @import("pass.zig");
    _ = @import("pdl.zig");
    _ = @import("quant.zig");
    _ = @import("sparse_tensor.zig");
    _ = @import("transform.zig");
}
