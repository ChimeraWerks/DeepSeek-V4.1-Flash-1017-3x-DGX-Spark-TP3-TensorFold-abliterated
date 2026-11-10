const std = @import("std");
const tokenizer = @import("tokenizer");
const template = @import("template");
const json = @import("json.zig");
const hf = @import("hf_text.zig");

test "recipe issue 63 native image placeholder survives normalization and encoding" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const placeholder = "<｜deepseek_image｜>";
    var h: hf.HfText = .{ .gpa = std.testing.allocator, .arena = .init(std.testing.allocator), .tok = try tokenizer.parse(std.testing.allocator,
        \\{"added_tokens":[{"id":129264,"content":"<｜deepseek_image｜>","special":true}],"model":{"type":"BPE","vocab":{"<｜deepseek_image｜>":129264},"merges":[]}}
    ), .source = "{{ messages[0].content }}" };
    defer h.arena.deinit();
    defer h.tok.deinit();
    h.compiled = try template.compile(std.testing.allocator, h.source, null);
    defer h.compiled.?.deinit();
    const input = (try json.parse(a,
        \\[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,eA=="}}]}]
    )).ok;
    var cx: @import("errors.zig").Cx = .{ .a = a };
    var images: @import("messages.zig").Images = .{ .placeholder = placeholder };
    const normalized = try @import("messages.zig").normalizeImages(&cx, input, "system", false, &images);
    try std.testing.expectEqualStrings("x", images.list.items[0]);
    try std.testing.expectEqualStrings(placeholder, normalized.array[0].get("content").?.string);
    var problem: []const u8 = "";
    try std.testing.expectEqualSlices(u32, &.{129264}, try h.text().renderIds(a, normalized, .{}, &problem));
}

test "recipe 47 native tokenizer ignores padding and truncation metadata" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const model = "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1,\"c\":2},\"merges\":[]}";
    var plain = try tokenizer.parse(std.testing.allocator, "{" ++ model ++ "}");
    defer plain.deinit();
    var configured = try tokenizer.parse(std.testing.allocator, "{\"padding\":{\"strategy\":\"BatchLongest\",\"direction\":\"Right\",\"pad_to_multiple_of\":8,\"pad_id\":0,\"pad_type_id\":0,\"pad_token\":\"a\"}," ++
        "\"truncation\":{\"direction\":\"Right\",\"max_length\":2,\"strategy\":\"LongestFirst\",\"stride\":0}," ++ model ++ "}");
    defer configured.deinit();
    const expected = try plain.encode(a, "abcabc");
    try std.testing.expectEqual(@as(usize, 6), expected.len);
    try std.testing.expectEqualSlices(u32, expected, try configured.encode(a, "abcabc"));
}

test "recipe 60 embedded current-turn template retains reasoning with no active tools" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Only the checkpoint's current-turn rule, without its unsupported **kwargs macro.
    const source = "{% set state = namespace(last_user=-1) %}{% for m in messages %}{% if m.role == 'user' or (m.role == 'system' and loop.index0 > 0) %}{% set state.last_user = loop.index0 %}{% endif %}{% endfor %}{% for m in messages %}{% if m.role == 'assistant' and loop.index0 > state.last_user %}{{ m.reasoning_content }}{% endif %}{% endfor %}";
    var h: hf.HfText = .{ .gpa = std.testing.allocator, .arena = .init(std.testing.allocator), .tok = try tokenizer.parse(std.testing.allocator,
        \\{"model":{"type":"BPE","vocab":{"x":0},"merges":[]}}
    ), .source = source };
    defer h.arena.deinit();
    defer h.tok.deinit();
    h.compiled = try template.compile(std.testing.allocator, source, null);
    defer h.compiled.?.deinit();
    for ([_][]const u8{ "user", "system" }) |boundary| {
        const input = (try json.parse(a, try std.fmt.allocPrint(a,
            \\[{{"role":"user","content":"q1"}},{{"role":"assistant","content":"a1","reasoning_content":"old reasoning"}},{{"role":"{s}","content":"q2"}},{{"role":"assistant","content":"checking","reasoning_content":"current reasoning","tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"lookup","arguments":"{{}}"}}}}]}},{{"role":"tool","content":"observed","tool_call_id":"call_1"}}]
        , .{boundary}))).ok;
        const tools = (try json.parse(a,
            \\[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]
        )).ok;
        var cx: @import("errors.zig").Cx = .{ .a = a };
        const prepared = (try @import("messages.zig").requestMessages(&cx, h.text(), input)).?;
        for ([_]?json.Value{ null, tools }) |offered| {
            const active = try @import("tool_specs.zig").active(&cx, offered, if (offered != null) .{ .string = "none" } else null);
            var problem: []const u8 = "";
            const actual = try h.text().render(a, prepared, .{ .enable_thinking = true, .tools = .{ .array = active } }, &problem);
            try std.testing.expectEqualStrings("current reasoning", actual);
        }
    }
}
