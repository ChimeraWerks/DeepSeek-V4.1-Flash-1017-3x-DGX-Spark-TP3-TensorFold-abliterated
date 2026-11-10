//! Wire spelling for raw model scores. Token bytes are null when the text adapter cannot expose lossless bytes.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const Text = @import("model_text.zig").Text;
const Value = json.Value;
pub const Row = []const api.LogprobRow;
fn token(a: std.mem.Allocator, text: Text, id: u32, value: f64) !*json.Object {
    const o = try json.newObject(a);
    try o.put(a, "token", .{ .string = try text.decode(a, &.{id}) });
    try o.put(a, "bytes", .null);
    try o.put(a, "logprob", .{ .float = value });
    return o;
}
pub fn rows(a: std.mem.Allocator, text: Text, input: Row) std.mem.Allocator.Error!Value {
    const out = try a.alloc(Value, input.len);
    for (input, out) |row, *item| {
        const o = try token(a, text, row.chosen.id, row.chosen.logprob);
        var top: std.ArrayList(Value) = .empty;
        for (row.top[0..row.count]) |t| {
            if (!std.math.isFinite(t.logprob)) continue;
            try top.append(a, .{ .object = try token(a, text, t.id, t.logprob) });
        }
        try o.put(a, "top_logprobs", .{ .array = top.items });
        item.* = .{ .object = o };
    }
    const o = try json.newObject(a);
    try o.put(a, "content", .{ .array = out });
    return .{ .object = o };
}
