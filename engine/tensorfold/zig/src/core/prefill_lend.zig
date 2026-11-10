//! Bounded reply loans through the host's existing prefill debt.
const std = @import("std");

pub const Lend = struct {
    on: bool = false,
    rounds: u32 = 0,
    ns_round: f64 = 0,
    tokens_round: f64 = 0,

    pub fn fromEnv() Lend {
        return .{ .on = if (std.c.getenv("TF_FILL_LEND")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false };
    }

    pub fn rounded(s: *Lend, ns: i64, tokens: u64, runnable: usize) void {
        if (!s.on or ns <= 0 or tokens == 0 or runnable == 0) return;
        s.rounds +|= 1;
        const w = @max(1.0 / 32.0, 1.0 / @as(f64, @floatFromInt(s.rounds)));
        s.ns_round += w * (@as(f64, @floatFromInt(ns)) - s.ns_round);
        s.tokens_round += w * (@as(f64, @floatFromInt(tokens)) / @as(f64, @floatFromInt(runnable)) - s.tokens_round);
    }

    /// Unknown rates and replies near a tie leave the prompt first.
    pub fn replyNs(s: *const Lend, tokens: i64, remaining: usize, completed: usize, work_ns: i64) ?f64 {
        if (!s.on or tokens <= 0 or remaining == 0 or completed == 0 or work_ns <= 0 or s.ns_round <= 0 or s.tokens_round <= 0) return null;
        const reply = @as(f64, @floatFromInt(tokens)) * s.ns_round / s.tokens_round;
        const fill = @as(f64, @floatFromInt(remaining)) * @as(f64, @floatFromInt(work_ns)) / @as(f64, @floatFromInt(completed));
        return if (reply < fill / 4) reply else null;
    }

    /// Credit comes only from completed measured fill occupancy, never from predicted future work.
    pub fn grant(work_ns: i64, lent_ns: i64, reply_ns: f64) i64 {
        const budget: i64 = @divTrunc(@max(0, work_ns), 4);
        const spent: i64 = @max(0, lent_ns);
        const available = if (spent >= budget) 0 else budget - spent;
        if (available == 0 or !std.math.isFinite(reply_ns) or reply_ns <= 0) return 0;
        const wanted: i64 = @intFromFloat(@min(reply_ns, @as(f64, @floatFromInt(std.math.maxInt(i64) / 2))));
        return @min(available, wanted);
    }
};

test "stepped prefill lending waits for samples and requires a quarter-margin reply" {
    var s: Lend = .{ .on = true };
    try std.testing.expect(s.replyNs(10, 900, 100, 1000) == null);
    s.rounded(30, 2, 1);
    s.rounded(30, 4, 1);
    try std.testing.expectEqual(@as(f64, 3), s.tokens_round);
    try std.testing.expectEqual(@as(f64, 100), s.replyNs(10, 900, 100, 1000).?);
    try std.testing.expect(s.replyNs(225, 900, 100, 1000) == null);
    try std.testing.expect(s.replyNs(224, 900, 100, 1000) != null);
    try std.testing.expect(s.replyNs(1, 900, 0, 1000) == null);
    s.on = false;
    try std.testing.expect(s.replyNs(1, 900, 100, 1000) == null);
    const old = s;
    s.rounded(1, 1, 1);
    try std.testing.expectEqualDeep(old, s);
}

test "stepped prefill lending burst average and sustained credit have a hard bound" {
    var s: Lend = .{ .on = true };
    for (0..64) |i| s.rounded(40, if (i % 2 == 0) 1 else 5, 1);
    const before = s.tokens_round;
    s.rounded(40, 5, 1);
    s.rounded(40, 5, 1);
    try std.testing.expect(@abs(s.tokens_round - before) < 0.2);
    var lent: i64 = 0;
    for (1..1001) |i| {
        const work: i64 = @intCast(i * 1000);
        const loan = Lend.grant(work, lent, 1e9);
        lent += loan;
        try std.testing.expect(lent <= @divTrunc(work, 4));
        try std.testing.expectEqual(@as(i64, 0), Lend.grant(work, lent, 1e9));
    }
    try std.testing.expectEqual(@as(i64, 250000), lent);
    try std.testing.expectEqual(@as(i64, 4), Lend.grant(100, 0, 4));
    try std.testing.expectEqual(@as(i64, 0), Lend.grant(100, 0, std.math.nan(f64)));
    try std.testing.expectEqual(@as(i64, 0), Lend.grant(100, 0, std.math.inf(f64)));
    try std.testing.expectEqual(@as(i64, 0), Lend.grant(100, 26, 10));
    try std.testing.expect(Lend.grant(std.math.maxInt(i64), 0, 1e30) <= @divTrunc(std.math.maxInt(i64), 4));
}
