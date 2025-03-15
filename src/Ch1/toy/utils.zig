const std = @import("std");

/// Check whether 2 floating numbers are close enought (like `numpy.isclose()`).
/// NaN is considered not to be the same as any value.
pub fn isCloseWithTolerence(comptime T: type, a: T, b: T, rtol: T, atol: T) bool {
    std.debug.assert(@typeInfo(T) == .Float);
    std.debug.assert(rtol > 0);
    std.debug.assert(atol > 0);

    // Fast path for equal values (and signed zeros and infinites)
    // ref: `std.math.approxEqRel()`
    if (a == b) {
        return true;
    }

    if (std.math.isNan(a) or std.math.isNan(b)) {
        return false;
    }

    const delta = a - b;
    const delta_abs = if (delta > 0) delta else -delta;
    const b_abs = if (b > 0) b else -b;
    return delta_abs <= (atol + rtol * b_abs);
}

/// A shortcut to call `isCloseWithTolerence()` with default tolerences.
pub inline fn isClose(comptime T: type, a: T, b: T) bool {
    return isCloseWithTolerence(T, a, b, 1e-5, 1e-8);
}
