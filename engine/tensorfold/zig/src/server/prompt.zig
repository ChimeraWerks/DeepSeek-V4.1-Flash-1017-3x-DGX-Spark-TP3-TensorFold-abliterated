//! Prompt ids from a request, as ``prepare_prompt`` and ``render_prompt_ids`` make them.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const model_text = @import("model_text.zig");
const chat = @import("chat.zig");
const Server = @import("server.zig").Server;
const api = @import("engine_api");
const Value = json.Value;
const Cx = errors.Cx;

/// With images, `cache_key` is what the prompt cache matches: the ids with each image span spelled by its digest.
pub const Rendered = struct { ids: []const u32, history_len: usize = 0, images: []const api.Image = &.{}, cache_key: []const u32 = &.{}, media: []const api.prompt_cache.Media = &.{} };

pub const isTitle = messages_mod.isTitleRequest;

/// ``render_prompt_ids``: messages normalized for this template, rendered, and a dangling ``<think>`` closed.
pub fn renderIds(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool) errors.Refused![]const u32 {
    return renderRequest(srv, cx, messages, tools, thinking, effort, generation, null);
}

/// The schema a request's ``response_format`` asks the template to state, or null for free text.
pub fn responseFormat(f: Value) ?Value {
    const rf = f.field("response_format") orelse return null;
    if (rf != .object) return null;
    if (rf.typeIs("json_object")) return rf;
    if (rf.typeIs("json_schema")) if (rf.field("json_schema")) |js| return js.field("schema");
    return null;
}

/// The template states tools and a response format in its system block, as deepseek-recipe does: an empty one if need be.
fn withSystem(cx: *Cx, messages: Value) errors.Refused!Value {
    if (messages != .array or messages.array.len == 0) return messages;
    const role = messages.array[0].get("role");
    if (role != null and role.? == .string and std.mem.eql(u8, role.?.string, "system")) return messages;
    const system = try json.newObject(cx.a);
    try system.put(cx.a, "role", .{ .string = "system" });
    try system.put(cx.a, "content", .{ .string = "" });
    return .{ .array = try std.mem.concat(cx.a, Value, &.{ &.{.{ .object = system }}, messages.array }) };
}

/// A last assistant message marked ``prefix`` is continued, so it must not end its turn.
fn continues(messages: Value) bool {
    if (messages != .array or messages.array.len == 0) return false;
    const last = messages.array[messages.array.len - 1];
    const prefix = last.get("prefix") orelse return false;
    return prefix.truthy();
}

pub fn renderRequest(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool, format: ?Value) errors.Refused![]const u32 {
    var normalized = try messages_mod.toolArguments(cx, try messages_mod.normalize(cx, messages, srv.late_system, srv.needs_user_after_tool));
    if (srv.ds41_tool_separator) normalized = try messages_mod.ds41ToolSeparator(cx, normalized);
    var problem: []const u8 = "";
    const options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = @constCast(tools) } else null,
        .add_generation_prompt = generation,
        .enable_thinking = thinking,
        .reasoning_effort = if (thinking) effort else null,
        .reasoning_effort_value = if (thinking) srv.effortValue(effort) else null,
        .response_format = format,
    };
    if (tools.len > 0 or format != null) normalized = try withSystem(cx, normalized);
    var ids = srv.text.renderIds(cx.a, normalized, options, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.server, "{s}", .{problem}),
    };
    if (!thinking and generation and ids.len > 0) {
        const last = try srv.text.decode(cx.a, ids[ids.len - 1 ..]);
        if (std.mem.eql(u8, @import("reply_text.zig").pyStrip(last), "<think>")) {
            const close = srv.text.encode(cx.a, "</think>", false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Template => &[_]u32{},
            };
            if (close.len == 1) ids = try std.mem.concat(cx.a, u32, &.{ ids, close });
        }
    }
    if (continues(messages) and ids.len > 0) {
        const last = try srv.text.decode(cx.a, ids[ids.len - 1 ..]);
        if (std.mem.endsWith(u8, last, "end▁of▁sentence｜>")) ids = ids[0 .. ids.len - 1];
    }
    return ids;
}

/// A request's prompt: raw text or ids for a completion, else the chat template's, with its history length.
pub fn prepare(srv: *Server, cx: *Cx, input: chat.Input, thinking: bool, effort: ?[]const u8) errors.Refused!Rendered {
    if (input.prompt) |p| switch (p) {
        .text => |t| return .{ .ids = srv.text.encode(cx.a, t, true) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => return cx.fail(.server, "the tokenizer cannot encode this prompt", .{}),
        } },
        .ids => |ids| return .{ .ids = ids },
    };
    const format = responseFormat(input.fields);
    const prompt = try renderRequest(srv, cx, input.messages, input.tools, thinking, effort, true, format);
    const history = try renderRequest(srv, cx, input.messages, input.tools, thinking, effort, false, format);
    var history_len: usize = 0;
    if (history.len > 0 and history.len < prompt.len and std.mem.eql(u32, prompt[0..history.len], history)) history_len = history.len;
    if (history_len == 0 and prompt.len > 1 and std.mem.eql(u32, history, prompt)) history_len = prompt.len - 1; // a template with no generation suffix
    if (input.images.len > 0) {
        var rendered = try spans(srv, cx, prompt, history_len, input.images);
        // The one switch for RAM and spill: no content key means neither may restore or keep an image prompt.
        if (!api.prompt_cache.imageCacheOn(if (std.c.getenv("TF_DS_IMAGE_CACHE")) |v| std.mem.span(v) else null)) {
            rendered.cache_key = &.{};
            rendered.media = &.{};
            rendered.history_len = 0;
        }
        return rendered;
    }
    return .{ .ids = prompt, .history_len = history_len };
}

/// Each image placeholder of the rendered prompt as its image's span (the vision family's prepare: the span's length and
/// the payload the engine reads), the images in prompt order.
fn spans(srv: *Server, cx: *Cx, ids: []const u32, history_len: usize, images: []const []const u8) errors.Refused!Rendered {
    const vision = srv.info.vision orelse return cx.refuse("this server accepts text only; image, audio and video inputs are unsupported");
    var out: std.ArrayList(u32) = .empty;
    var key: std.ArrayList(u32) = .empty;
    var placed: std.ArrayList(api.Image) = .empty;
    var history_extra: usize = 0;
    var media: std.ArrayList(api.prompt_cache.Media) = .empty;
    // Each picture is prepared in this scratch, emptied before the next one, and only its payload is copied into the
    // request's arena: decoding, resizing and patchifying take ~15 MiB a screenshot that an arena would hold until the
    // request ends.
    var scratch: std.heap.ArenaAllocator = .init(srv.gpa);
    defer scratch.deinit();
    for (ids, 0..) |id, i| {
        if (id != vision.token) {
            try out.append(cx.a, id);
            try key.append(cx.a, id);
            continue;
        }
        const k = placed.items.len;
        if (k >= images.len) return cx.refuse("image input: the prompt holds more image placeholders than images");
        var problem: []const u8 = "";
        _ = scratch.reset(.retain_capacity);
        const prep = vision.prepare(vision.ctx, scratch.allocator(), images[k], &problem) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.BadImage => return cx.refuse(try cx.a.dupe(u8, problem)), // the message may live in the scratch
        };
        const payload = try cx.a.alignedAlloc(u8, .@"16", prep.payload.len);
        @memcpy(payload, prep.payload);
        try placed.append(cx.a, .{ .at = @intCast(out.items.len), .tokens = prep.tokens, .bytes = payload });
        try out.appendNTimes(cx.a, vision.token, prep.tokens);
        try spellImage(cx.a, &key, payload, prep.tokens);
        if (i < history_len) history_extra += prep.tokens - 1;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(payload, &digest, .{});
        try media.append(cx.a, .{ .start = @intCast(out.items.len - prep.tokens), .len = prep.tokens, .digest = digest });
    }
    if (placed.items.len != images.len) return cx.refuse("image input: the prompt holds fewer image placeholders than images");
    return .{ .ids = out.items, .images = placed.items, .cache_key = key.items, .media = media.items, .history_len = if (history_len > 0) history_len + history_extra else 0 };
}

/// An image span as the cache key spells it: ids past every vocabulary's, one a position, from the payload's digest.
fn spellImage(a: std.mem.Allocator, key: *std.ArrayList(u32), payload: []const u8, tokens: u32) !void {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(payload, &digest, .{});
    for (0..tokens) |j| try key.append(a, api.prompt_cache.pictureId(&digest, @intCast(j)));
}

test "an image span's key depends on the image and the position, never on the vocabulary" {
    const a = std.testing.allocator;
    var one: std.ArrayList(u32) = .empty;
    defer one.deinit(a);
    var same: std.ArrayList(u32) = .empty;
    defer same.deinit(a);
    var other: std.ArrayList(u32) = .empty;
    defer other.deinit(a);
    try spellImage(a, &one, "the same pixels", 64);
    try spellImage(a, &same, "the same pixels", 64);
    try spellImage(a, &other, "other pixels", 64);
    try std.testing.expectEqualSlices(u32, one.items, same.items);
    try std.testing.expect(!std.mem.eql(u32, one.items, other.items));
    for (one.items) |id| try std.testing.expect(id >= 0x8000_0000);
    try std.testing.expect(one.items[0] != one.items[1]);
}

/// A reusable system prefix, found with a probe in place of the first user message; zero below 512 tokens.
pub fn systemPrefixLen(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, prompt_ids: []const u32, thinking: bool, effort: ?[]const u8) usize {
    if (messages != .array) return 0;
    const first_user = for (messages.array, 0..) |m, i| {
        const role = m.get("role") orelse continue;
        if (role == .string and std.mem.eql(u8, role.string, "user")) break i;
    } else return 0;
    const probe = cx.a.alloc(Value, first_user + 1) catch return 0;
    @memcpy(probe[0..first_user], messages.array[0..first_user]);
    const user = json.newObject(cx.a) catch return 0;
    user.put(cx.a, "role", .{ .string = "user" }) catch return 0;
    user.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return 0;
    probe[first_user] = .{ .object = user };
    var scratch: Cx = .{ .a = cx.a };
    const other = renderIds(srv, &scratch, .{ .array = probe }, tools, thinking, effort, true) catch return 0;
    var shared: usize = 0;
    while (shared < @min(prompt_ids.len, other.len) and prompt_ids[shared] == other[shared]) shared += 1;
    return if (shared >= 512) shared else 0;
}
