//! Arena-owned append storage. Values are immutable slices, not mutable buffers.
const std = @import("std");

pub fn Buffer(comptime T: type) type {
    return struct {
        const Self = @This();
        storage: []T,
        used: usize,

        /// Only the latest full prefix may append. An older alias forks storage.
        /// Never resize/realloc/free: saved values must survive both growth and forks.
        pub fn concat(a: std.mem.Allocator, owner: ?*Self, left: []const T, right: []const T) error{OutOfMemory}!struct { []const T, *Self } {
            const end = std.math.add(usize, left.len, right.len) catch return error.OutOfMemory;
            const current = if (owner) |b| b.used == left.len and b.storage.ptr == left.ptr else false;
            const b = if (current) owner.? else try a.create(Self);
            if (!current) b.* = .{ .storage = &.{}, .used = 0 };
            if (end > b.storage.len) {
                const floor = @max(b.storage.len, @max(1, 64 / @sizeOf(T)));
                const capacity = @max(end, std.math.mul(usize, floor, 2) catch end);
                const fresh = try a.alloc(T, capacity);
                @memcpy(fresh[0..left.len], left);
                // right may alias left or any retained allocation.
                @memcpy(fresh[left.len..end], right);
                b.storage = fresh;
            } else {
                // Existing immutable slices end at/before used, so the destination
                // is disjoint from them (including a self-concatenation).
                @memcpy(b.storage[left.len..end], right);
            }
            b.used = end;
            return .{ b.storage[0..end], b };
        }
    };
}
