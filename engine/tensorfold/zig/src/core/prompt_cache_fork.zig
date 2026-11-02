//! Pure fork evidence. Physical eligibility, namespace and budget remain the Store's responsibility.
const std = @import("std");

/// A fork's state must not read the first differing token. Full-prompt matches are not fork evidence.
/// Picture ids are shortened hashes: without full media digests in Entry.last, never learn across them.
pub fn cut(previous: []const u32, prompt: []const u32, lookahead: u32, history: u32) u32 {
    const n = std.mem.indexOfDiff(u32, previous, prompt) orelse @min(previous.len, prompt.len);
    if (n == 0 or n >= prompt.len or n >= history or n < history / 2) return 0;
    for (prompt[0..n]) |id| if (id >= 0x8000_0000) return 0;
    return @intCast(n -| @as(usize, lookahead));
}

test "fork evidence subtracts lookahead and excludes whole prompts, weak prefixes and unbound media" {
    const a = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]u32{ 1, 2, 3, 4, 8, 9 };
    try std.testing.expectEqual(@as(u32, 3), cut(&a, &b, 1, 5));
    try std.testing.expectEqual(@as(u32, 0), cut(&a, &a, 1, 7));
    try std.testing.expectEqual(@as(u32, 0), cut(&a, &b, 1, 4));
    try std.testing.expectEqual(@as(u32, 0), cut(&a, &b, 1, 10));
    try std.testing.expectEqual(@as(u32, 0), cut(&a, &b, 9, 5));
    try std.testing.expectEqual(@as(u32, 0), cut(&.{ 0x80000001, 2, 3 }, &.{ 0x80000001, 2, 4 }, 0, 3));
}
