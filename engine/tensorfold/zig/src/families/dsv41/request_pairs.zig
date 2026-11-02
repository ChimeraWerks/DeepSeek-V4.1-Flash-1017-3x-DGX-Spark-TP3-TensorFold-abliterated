//! Host-only validation for paired serial/speculative requests in the standalone runner.
const std = @import("std");

pub fn validate(reqs: anytype, global_serial: bool, drafting: bool) !usize {
    var count: usize = 0;
    for (reqs, 0..) |r, i| {
        if (r.expect_request) |ref| {
            if (ref >= i) return error.ReferenceMustPrecedeRequest;
            const base = reqs[ref];
            if (global_serial or !drafting or r.serial or !base.serial or base.expect_request != null)
                return error.PairMustCompareSerialToSpeculative;
            if (r.temperature != 0 or base.temperature != 0) return error.PairRequiresGreedySampling;
            if (r.images.len != 0 or base.images.len != 0) return error.PairedVisionUnsupported;
            if (r.max_tokens == 0 or base.max_tokens != r.max_tokens or r.prompt.len == 0 or
                !std.mem.eql(u32, r.prompt, base.prompt)) return error.PairInputsDiffer;
            count += 1;
        }
    }
    return count;
}

pub fn ready(serial: bool, reference: ?usize, completed: []const bool, active: usize) bool {
    if (serial and active != 0) return false;
    if (reference) |r| return r < completed.len and completed[r];
    return true;
}

const FakeRequest = struct {
    prompt: []const u32 = &.{ 1, 2, 3 },
    max_tokens: u32 = 64,
    serial: bool = false,
    expect_request: ?usize = null,
    temperature: f64 = 0,
    images: []const u8 = &.{},
};

test "pairs bind identical nonempty greedy input to an earlier serial reference" {
    var requests = [_]FakeRequest{ .{ .serial = true }, .{ .expect_request = 0 } };
    try std.testing.expectEqual(@as(usize, 1), try validate(&requests, false, true));
    requests[1].prompt = &.{ 1, 2, 4 };
    try std.testing.expectError(error.PairInputsDiffer, validate(&requests, false, true));
    requests[1].prompt = requests[0].prompt;
    requests[1].max_tokens = 65;
    try std.testing.expectError(error.PairInputsDiffer, validate(&requests, false, true));
}

test "forward references and accidental serial-vs-serial comparisons fail" {
    var requests = [_]FakeRequest{ .{ .serial = true }, .{ .expect_request = 1 } };
    try std.testing.expectError(error.ReferenceMustPrecedeRequest, validate(&requests, false, true));
    requests[1].expect_request = 0;
    try std.testing.expectError(error.PairMustCompareSerialToSpeculative, validate(&requests, true, true));
    try std.testing.expectError(error.PairMustCompareSerialToSpeculative, validate(&requests, false, false));
    requests[0].serial = false;
    try std.testing.expectError(error.PairMustCompareSerialToSpeculative, validate(&requests, false, true));
}

test "sampling and image paths cannot claim this greedy text qualification" {
    var requests = [_]FakeRequest{ .{ .serial = true }, .{ .expect_request = 0, .temperature = 0.5 } };
    try std.testing.expectError(error.PairRequiresGreedySampling, validate(&requests, false, true));
    requests[1].temperature = 0;
    requests[0].images = &.{1};
    try std.testing.expectError(error.PairedVisionUnsupported, validate(&requests, false, true));
}

test "reference runs alone and dependent waits while independent drafted jobs can share" {
    try std.testing.expect(!ready(true, null, &.{}, 1));
    try std.testing.expect(ready(true, null, &.{}, 0));
    try std.testing.expect(!ready(false, 0, &.{false}, 0));
    try std.testing.expect(!ready(false, 2, &.{true}, 0));
    try std.testing.expect(ready(false, 0, &.{true}, 7));
}
