//! A bounded map for captured round stretches. An insertion returns the least recently used value for its owner
//! to free; a replay counts as a use. Reserving before a cross-rank vote never evicts an existing entry.

const std = @import("std");

pub fn Lru(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();
        const Slot = struct { v: V, used: u64 };

        map: std.AutoHashMapUnmanaged(K, Slot) = .empty,
        cap: usize, // 0: no cap
        tick: u64 = 0,
        evicted: u64 = 0,

        pub fn init(cap: usize) Self {
            return .{ .cap = cap };
        }

        pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
            self.map.deinit(gpa);
            self.* = undefined;
        }

        pub fn count(self: *const Self) usize {
            return self.map.count();
        }

        pub fn get(self: *Self, k: K) ?*V {
            const s = self.map.getPtr(k) orelse return null;
            self.tick += 1;
            s.used = self.tick;
            return &s.v;
        }

        /// Reserve a new slot without changing which keys the cache holds. The voted path calls this before the
        /// reserve stage, then inserts only after every rank agreed on the same result.
        pub fn reserve(self: *Self, gpa: std.mem.Allocator) !void {
            try self.map.ensureUnusedCapacity(gpa, 1);
        }

        /// Insert a key not already present after reserve. Eviction and insertion are one allocation-free step.
        pub fn putAssumeCapacity(self: *Self, k: K, v: V) ?V {
            var out: ?V = null;
            if (self.cap != 0 and self.map.count() >= self.cap) {
                // The scan is on a miss only, whose capture already dominates this cost. Each use has a unique
                // tick, so hash-map iteration order cannot change the victim between ranks.
                var it = self.map.iterator();
                var lru = it.next().?;
                while (it.next()) |en| if (en.value_ptr.used < lru.value_ptr.used) {
                    lru = en;
                };
                out = lru.value_ptr.v;
                self.map.removeByPtr(lru.key_ptr);
                self.evicted += 1;
            }
            self.tick += 1;
            self.map.putAssumeCapacityNoClobber(k, .{ .v = v, .used = self.tick });
            return out;
        }

        /// A failed allocation leaves all keys, values and their recency unchanged.
        pub fn put(self: *Self, gpa: std.mem.Allocator, k: K, v: V) !?V {
            try self.reserve(gpa);
            return self.putAssumeCapacity(k, v);
        }

        pub fn valueIterator(self: *const Self) ValueIterator {
            return .{ .it = self.map.valueIterator() };
        }

        pub const ValueIterator = struct {
            it: std.AutoHashMapUnmanaged(K, Slot).ValueIterator,
            pub fn next(i: *ValueIterator) ?*V {
                const s = i.it.next() orelse return null;
                return &s.v;
            }
        };
    };
}

const Key = struct { rows: u32, bucket: u64, stretch: u32, mode: u8 = 0 };

fn key(rows: u32, bucket: u64, stretch: u32) Key {
    return .{ .rows = rows, .bucket = bucket, .stretch = stretch };
}

test "at the cap a put hands back the least recently used entry, a get counting as a use" {
    const gpa = std.testing.allocator;
    var c = Lru(Key, u32).init(3);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, null), try c.put(gpa, key(1, 1024, 0), 10));
    try std.testing.expectEqual(@as(?u32, null), try c.put(gpa, key(1, 1024, 1), 11));
    try std.testing.expectEqual(@as(?u32, null), try c.put(gpa, key(1, 1024, 2), 12));
    try std.testing.expectEqual(@as(u32, 10), c.get(key(1, 1024, 0)).?.*);
    try std.testing.expectEqual(@as(?u32, 11), try c.put(gpa, key(2, 1024, 0), 20));
    try std.testing.expectEqual(@as(?*u32, null), c.get(key(1, 1024, 1)));
    try std.testing.expectEqual(@as(u32, 10), c.get(key(1, 1024, 0)).?.*);
    try std.testing.expectEqual(@as(u32, 12), c.get(key(1, 1024, 2)).?.*);
    try std.testing.expectEqual(@as(u32, 20), c.get(key(2, 1024, 0)).?.*);
    try std.testing.expectEqual(@as(usize, 3), c.count());
    try std.testing.expectEqual(@as(u64, 1), c.evicted);
}

test "cap 0 keeps every entry: the cache before GRAPH_CAP" {
    const gpa = std.testing.allocator;
    var c = Lru(Key, u32).init(0);
    defer c.deinit(gpa);
    for (0..600) |i| try std.testing.expectEqual(@as(?u32, null), try c.put(gpa, key(@intCast(i % 16 + 1), @as(u64, 1024) << @intCast(i / 48), @intCast(i % 3)), @intCast(i)));
    try std.testing.expectEqual(@as(usize, 600), c.count());
    try std.testing.expectEqual(@as(u64, 0), c.evicted);
}

test "over days of new shapes the cache stays at its cap and hands back each oldest graph once" {
    const gpa = std.testing.allocator;
    var c = Lru(Key, u32).init(128);
    defer c.deinit(gpa);
    for (0..600) |i| {
        const out = try c.put(gpa, key(@intCast(i % 16 + 1), @as(u64, 1024) << @intCast(i / 48), @intCast(i % 3)), @intCast(i));
        try std.testing.expectEqual(if (i < 128) null else @as(?u32, @intCast(i - 128)), out);
        try std.testing.expect(c.count() <= 128);
    }
    try std.testing.expectEqual(@as(usize, 128), c.count());
    try std.testing.expectEqual(@as(u64, 472), c.evicted);
}

test "three ranks evict the same key with replayed graphs and voted tombstones" {
    const gpa = std.testing.allocator;
    // Value zero stands for a voted no/tombstone. It occupies a slot and a hit refreshes recency.
    var ranks: [3]Lru(Key, u32) = .{ .init(3), .init(3), .init(3) };
    defer for (&ranks) |*c| c.deinit(gpa);
    for (&ranks) |*c| {
        try c.reserve(gpa);
        try std.testing.expectEqual(@as(?u32, null), c.putAssumeCapacity(key(1, 1024, 0), 10));
        try c.reserve(gpa);
        try std.testing.expectEqual(@as(?u32, null), c.putAssumeCapacity(key(1, 1024, 1), 11));
        try c.reserve(gpa);
        try std.testing.expectEqual(@as(?u32, null), c.putAssumeCapacity(key(1, 1024, 2), 12));
        try std.testing.expectEqual(@as(u32, 10), c.get(key(1, 1024, 0)).?.*);
        try c.reserve(gpa); // a reserve vote alone must not evict key 1
        try std.testing.expectEqual(@as(usize, 3), c.count());
        try std.testing.expectEqual(@as(u64, 0), c.evicted);
        try std.testing.expectEqual(@as(u32, 11), c.get(key(1, 1024, 1)).?.*);
        // key 1 was just replayed; key 2 is now oldest. A voted no installs a tombstone.
        try std.testing.expectEqual(@as(?u32, 12), c.putAssumeCapacity(key(2, 1024, 0), 0));
        try std.testing.expectEqual(@as(u32, 0), c.get(key(2, 1024, 0)).?.*);
        try c.reserve(gpa);
        try std.testing.expectEqual(@as(?u32, 10), c.putAssumeCapacity(key(3, 1024, 0), 30));
        try c.reserve(gpa);
        try std.testing.expectEqual(@as(?u32, 11), c.putAssumeCapacity(key(4, 1024, 0), 40));
        try c.reserve(gpa);
        try std.testing.expectEqual(@as(?u32, 0), c.putAssumeCapacity(key(5, 1024, 0), 50));
        try std.testing.expect(c.get(key(2, 1024, 0)) == null);
        try std.testing.expectEqual(@as(usize, 3), c.count());
        try std.testing.expectEqual(@as(u64, 4), c.evicted);
        // A discarded key is a clean miss: it can be captured again on every rank.
        try c.reserve(gpa);
        try std.testing.expectEqual(@as(?u32, 30), c.putAssumeCapacity(key(1, 1024, 0), 100));
        try std.testing.expectEqual(@as(u32, 100), c.get(key(1, 1024, 0)).?.*);
        try std.testing.expectEqual(@as(usize, 3), c.count());
        try std.testing.expectEqual(@as(u64, 5), c.evicted);
    }
}
