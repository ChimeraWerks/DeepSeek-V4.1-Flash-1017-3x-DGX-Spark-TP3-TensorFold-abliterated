//! Cross-chunk identity. No device state or persisted checkpoint fields.
const std = @import("std");

pub fn agreement(on: bool) u64 {
    return if (on) @as(u64, 1) << 59 else 0;
}
pub fn check(leader: u64, follower: u64) !void {
    if ((leader ^ follower) & agreement(true) != 0) return error.EngramKeepSettingMismatch;
}

pub const Key = struct {
    slot: usize,
    generation: u64,
    start: usize,
    n: usize,
    layers: u8,
    tokens: [32]u8,

    pub fn make(slot: usize, generation: u64, seq: []const i32, start: usize, n: usize, ngram: usize, layers: u8) ?Key {
        if (generation == 0 or n == 0 or ngram == 0 or start > seq.len or n > seq.len - start or layers == 0) return null;
        // Hasher.hashes reads only these tokens, including the left n-gram halo.
        const ids = seq[start -| (ngram - 1) .. start + n];
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(ids), &digest, .{});
        return .{ .slot = slot, .generation = generation, .start = start, .n = n, .layers = layers, .tokens = digest };
    }

    pub fn eql(a: Key, b: Key) bool {
        return std.meta.eql(a, b);
    }
};

pub const State = struct {
    generation: u64 = 0,
    key: ?Key = null,

    pub fn begin(s: *State) !void {
        s.key = null;
        if (s.generation == std.math.maxInt(u64)) return error.EngramGenerationExhausted;
        s.generation += 1;
    }
};

test "prefill slice engram keep identity covers occupant interval layers and ngram halo" {
    var state: State = .{};
    try state.begin();
    var ids = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    const key = Key.make(2, state.generation, &ids, 4, 4, 4, 3).?;
    try std.testing.expect(key.eql(Key.make(2, state.generation, &ids, 4, 4, 4, 3).?));
    try std.testing.expect(!key.eql(Key.make(1, state.generation, &ids, 4, 4, 4, 3).?));
    try std.testing.expect(!key.eql(Key.make(2, state.generation, &ids, 3, 4, 4, 3).?));
    try std.testing.expect(!key.eql(Key.make(2, state.generation, &ids, 4, 3, 4, 3).?));
    try std.testing.expect(!key.eql(Key.make(2, state.generation, &ids, 4, 4, 4, 1).?));
    for (1..8) |i| {
        ids[i] = -1;
        try std.testing.expect(!key.eql(Key.make(2, state.generation, &ids, 4, 4, 4, 3).?));
        ids[i] = @intCast(i);
    }
    ids[0] = 99; // outside every n-gram
    ids[8] = 99; // not in this chunk
    try std.testing.expect(key.eql(Key.make(2, state.generation, &ids, 4, 4, 4, 3).?));
    try state.begin();
    try std.testing.expect(!key.eql(Key.make(2, state.generation, &ids, 4, 4, 4, 3).?));
    state.generation = std.math.maxInt(u64);
    try std.testing.expectError(error.EngramGenerationExhausted, state.begin());
    try std.testing.expect(Key.make(2, 0, &ids, 4, 4, 4, 3) == null);
    try std.testing.expect(Key.make(2, 1, &ids, 8, 4, 4, 3) == null);
}

test "prefill slice engram keep startup bit 59 rejects either follower and off adds zero" {
    const base = @as(u64, 1) << 63 | @as(u64, 1) << 58 | 0x123456789;
    try std.testing.expectEqual(base, base | agreement(false));
    try std.testing.expectEqual(@as(u64, 1) << 59, agreement(true));
    for (0..3) |rank| {
        var words: [3]u64 = @splat(base);
        words[rank] |= agreement(true);
        var mismatched = false;
        for (words[1..]) |w| check(words[0], w) catch {
            mismatched = true;
        };
        try std.testing.expect(mismatched);
    }
    try check(base | agreement(true), base | agreement(true));
}
