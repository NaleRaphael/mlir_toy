/// This example is trying to re-implement the `ScopedHashTable` in LLVM/ADT.
///
/// It is basically a hash map in which values are wrapped by `ScopedValue` (a
/// linked list) and managed with `Scope`.
///
/// The `ScopedValue` is used to trace:
/// - `prev_in_scope`: the last value defined in a scope. This will be used to
///   traverse all values in that scope.
/// - `prev_in_map: the previous value defined with an identical key.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn ScopedHashMap(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime max_load_percentage: u64,
) type {
    return struct {
        top_level_map: map_t,
        cur_scope: ?*Scope,
        allocator: Allocator,

        const Self = @This();
        const val_t = ScopedValue;
        const map_t = std.HashMap(K, *ScopedValue, Context, max_load_percentage);
        const scoped_map_t = Self;

        const ScopedValue = struct {
            prev_in_scope: ?*ScopedValue,
            prev_in_map: ?*ScopedValue,
            key: K,
            value: V,
            allocator: Allocator,

            pub fn init(
                prev_in_scope: ?*ScopedValue,
                prev_in_map: ?*ScopedValue,
                key: K,
                value: V,
                allocator: Allocator,
            ) Allocator.Error!*ScopedValue {
                const inst = try allocator.create(ScopedValue);
                inst.* = .{
                    .prev_in_scope = prev_in_scope,
                    .prev_in_map = prev_in_map,
                    .key = key,
                    .value = value,
                    .allocator = allocator,
                };
                return inst;
            }

            pub fn deinit(self: *ScopedValue) void {
                self.allocator.destroy(self);
            }
        };

        const Scope = struct {
            scoped_map: *scoped_map_t,
            parent: ?*Scope,
            last_value_in_scope: ?*ScopedValue,
            allocator: Allocator,

            pub fn init(
                scoped_map: *scoped_map_t,
                allocator: Allocator,
            ) Allocator.Error!*Scope {
                const inst = try allocator.create(Scope);
                inst.* = .{
                    .scoped_map = scoped_map,
                    .parent = scoped_map.cur_scope,
                    .last_value_in_scope = null,
                    .allocator = allocator,
                };
                scoped_map.cur_scope = inst;
                return inst;
            }

            pub fn deinit(self: *Scope) void {
                std.debug.assert(self.scoped_map.cur_scope != null);
                std.debug.assert(self.scoped_map.cur_scope.? == self);

                self.scoped_map.cur_scope = self.parent;

                while (self.last_value_in_scope) |this_entry| {
                    if (this_entry.prev_in_map) |sv| {
                        // There is existing entry added before this scope,
                        // so we just need to pop it.
                        const entry = self.scoped_map.top_level_map.getEntry(this_entry.key);
                        std.debug.assert(std.meta.eql(entry.?.value_ptr.*, this_entry));
                        entry.?.value_ptr.* = sv;
                    } else {
                        // Entry exists in this scope only, so we can remove it
                        const key = this_entry.key;
                        std.debug.assert(std.meta.eql(
                            self.scoped_map.top_level_map.get(key).?,
                            this_entry,
                        ));
                        _ = self.scoped_map.top_level_map.remove(key);
                    }

                    self.last_value_in_scope = this_entry.prev_in_scope;
                    this_entry.deinit();
                }

                self.allocator.destroy(self);
            }
        };

        pub fn init(allocator: Allocator) Self {
            return .{
                .top_level_map = map_t.init(allocator),
                .cur_scope = null,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            // TODO: iterate keys and call `ctx.deinitKey(key)`
            self.top_level_map.deinit();
        }

        pub fn createScope(self: *Self) Allocator.Error!void {
            _ = try Scope.init(self, self.allocator);
        }

        pub fn destroyScope(self: *Self) void {
            std.debug.assert(self.cur_scope != null);
            self.cur_scope.?.deinit();
        }

        pub fn put(self: *Self, key: K, value: V) Allocator.Error!void {
            std.debug.assert(self.cur_scope != null);
            const cur_scope = self.cur_scope.?;

            const entry = try self.top_level_map.getOrPut(key);
            if (!entry.found_existing) {
                // TODO: call `ctx.initKey(key)` for those types requiring
                // further management.
                entry.key_ptr.* = key;
            }

            const prev = if (entry.found_existing) entry.value_ptr.* else null;
            const scoped_value = try val_t.init(
                cur_scope.last_value_in_scope,
                prev,
                key,
                value,
                self.allocator,
            );

            entry.value_ptr.* = scoped_value;
            cur_scope.last_value_in_scope = scoped_value;
        }

        pub fn get(self: *Self, key: K) ?V {
            if (self.top_level_map.get(key)) |v| {
                return v.value;
            }
            return null;
        }

        pub fn contains(self: Self, key: K) bool {
            return self.top_level_map.contains(key);
        }
    };
}

test ScopedHashMap {
    const expect = std.testing.expect;
    const allocator = std.testing.allocator;

    const ctx = std.hash_map.StringContext;
    const load_factor = std.hash_map.default_max_load_percentage;
    const scoped_hash_map_t = ScopedHashMap([]const u8, u64, ctx, load_factor);

    var scoped_hash_map = scoped_hash_map_t.init(allocator);
    defer scoped_hash_map.deinit();

    try scoped_hash_map.createScope();
    defer scoped_hash_map.destroyScope();

    try scoped_hash_map.put("v0", 0);
    try scoped_hash_map.put("v1", 1);
    try expect(scoped_hash_map.get("v0").? == 0);
    try expect(scoped_hash_map.get("v1").? == 1);

    {
        try scoped_hash_map.createScope();
        defer scoped_hash_map.destroyScope();

        try scoped_hash_map.put("v0", 42);
        try expect(scoped_hash_map.get("v0").? == 42);
        try expect(scoped_hash_map.get("v1").? == 1);

        try scoped_hash_map.put("v0", 53);
        try expect(scoped_hash_map.get("v0").? == 53);
        try expect(scoped_hash_map.get("v1").? == 1);

        {
            try scoped_hash_map.createScope();
            defer scoped_hash_map.destroyScope();

            try scoped_hash_map.put("v0", 64);
            try expect(scoped_hash_map.get("v0").? == 64);
            try expect(scoped_hash_map.get("v1").? == 1);
        }

        try expect(scoped_hash_map.get("v0").? == 53);
        try expect(scoped_hash_map.get("v1").? == 1);
    }

    try expect(scoped_hash_map.get("v0").? == 0);
    try expect(scoped_hash_map.get("v1").? == 1);
}
