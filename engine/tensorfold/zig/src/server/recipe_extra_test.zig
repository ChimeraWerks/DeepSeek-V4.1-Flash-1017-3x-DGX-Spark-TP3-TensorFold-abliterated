const std = @import("std");
const reply = @import("reply_text.zig");

test "recipe issue 45 quoted terminator stays literal unless emitted as control" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const prefix = "The thinking delimiter is `";
    const text = prefix ++ "</think>` and this is still deliberation.";
    const literal: reply.Markup = .{ .text = text, .token_aware = true };
    for (0..text.len + 1) |end| {
        const split = try reply.splitThinkingWith(a, text[0..end], end == text.len, reply.think_markers, literal);
        try std.testing.expect(std.mem.startsWith(u8, text[0..end], split.reasoning));
        if (end == text.len) try std.testing.expectEqualStrings(text, split.reasoning);
        try std.testing.expectEqualStrings("", split.answer);
    }
    const spans = [_]reply.Span{.{ .start = prefix.len, .end = prefix.len + "</think>".len }};
    const control: reply.Markup = .{ .text = text, .spans = &spans, .token_aware = true };
    const split = try reply.splitThinkingWith(a, text, true, reply.think_markers, control);
    try std.testing.expectEqualStrings(prefix, split.reasoning);
    try std.testing.expectEqualStrings("` and this is still deliberation.", split.answer);
}
