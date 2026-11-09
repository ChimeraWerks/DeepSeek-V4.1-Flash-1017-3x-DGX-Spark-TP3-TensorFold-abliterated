//! vLLM's /tokenize and /detokenize: the ids the chat and completion routes would run.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const messages = @import("messages.zig");
const prompt = @import("prompt.zig");
const tool_specs = @import("tool_specs.zig");
const routes = @import("routes.zig");
const http_body = @import("http_body.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const paths = [_][]const u8{ "/tokenize", "/v1/tokenize", "/detokenize", "/v1/detokenize" };

fn flag(cx: *Cx, body: Value, name: []const u8, default: bool) errors.Refused!bool {
    const v = body.get(name) orelse return default;
    if (v != .bool) return cx.fail(.request, "{s} must be a boolean", .{name});
    return v.bool;
}

/// Reuse the native Responses translator; counting never consults stored history.
fn responseTokenBody(cx: *Cx, body: Value) errors.Refused!Value {
    const NoHistory = struct {
        pub fn conversation(_: @This(), context: *Cx, _: Value) errors.Refused![]Value {
            return context.refuse("Responses token counting requires full input, without previous_response_id");
        }
    };
    return (try @import("responses_translate.zig").translate(cx, body, NoHistory{})).chat;
}

fn tokenize(srv: *Server, cx: *Cx, raw_body: Value) errors.Refused!Value {
    const a = cx.a;
    const body = if (std.mem.eql(u8, raw_body.strField("api") orelse "", "responses")) try responseTokenBody(cx, raw_body) else raw_body;
    if (body != .object) return cx.refuse("the request body must be a JSON object");
    const strings = try flag(cx, body, "return_token_strs", false);
    var ids: []const u32 = undefined;
    var image_tokens: usize = 0;
    if (body.has("messages")) {
        const generation = try flag(cx, body, "add_generation_prompt", true);
        // images as the chat route takes them: placeholders in the text, each spanned to its image's length below
        var images: messages.Images = .{ .placeholder = if (srv.info.vision) |v| v.placeholder else "" };
        const take: ?*messages.Images = if (srv.info.vision != null) &images else null;
        const msgs = try messages.normalizeImages(cx, try messages.requestMessages(cx, srv.text, body.get("messages")), "system", srv.needs_user_after_tool, take);
        const tools = tool_specs.active(cx, try messages.maskControls(cx.a, srv.text, body.get("tools")), body.get("tool_choice")) catch |e| switch (e) {
            error.Refused => return cx.refuse(cx.message), // a ValueError here is a RequestError
            else => |x| return x,
        };
        const thinking = try fields.thinkingFields(cx, body, srv.effort_levels);
        const on = thinking.enable orelse srv.config.enable_thinking;
        ids = prompt.renderRequest(srv, cx, msgs, tools, on, srv.effortFor(thinking.effort), generation, prompt.responseFormat(body)) catch |e| switch (e) {
            error.Refused => return cx.other(cx.message),
            else => |x| return x,
        };
        const counted = try prompt.countedIds(cx, srv.gpa, srv.info.vision, ids, images.list.items);
        ids = counted.ids;
        image_tokens = counted.image_tokens;
    } else {
        const text = body.get("prompt");
        if (text == null or text.? != .string) return cx.refuse("prompt must be a string (or send messages)");
        const special = try flag(cx, body, "add_special_tokens", true);
        ids = srv.text.encode(a, text.?.string, special) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => return cx.other("the tokenizer cannot encode this prompt"),
        };
    }
    const o = try json.newObject(a);
    try o.put(a, "count", try json.intValue(a, ids.len));
    // with images, their share of count, each span as generation lays it out (text-only replies keep vLLM's shape)
    if (image_tokens > 0) try o.put(a, "image_tokens", try json.intValue(a, image_tokens));
    try o.put(a, "max_model_len", if (srv.info.context_window > 0) try json.intValue(a, srv.info.context_window) else .null);
    const list = try a.alloc(Value, ids.len);
    for (ids, list) |t, *slot| slot.* = try json.intValue(a, t);
    try o.put(a, "tokens", .{ .array = list });
    if (strings) {
        const strs = try a.alloc(Value, ids.len);
        for (ids, strs) |t, *slot| slot.* = .{ .string = srv.text.tokenString(a, t) catch "" };
        try o.put(a, "token_strs", .{ .array = strs });
    }
    return .{ .object = o };
}

fn detokenize(srv: *Server, cx: *Cx, body: Value) errors.Refused!Value {
    if (body != .object) return cx.refuse("the request body must be a JSON object");
    var value = body.get("tokens") orelse Value.null;
    if (value == .array and value.array.len == 1 and value.array[0] == .array) value = value.array[0];
    if (value != .array) return cx.refuse("tokens must be a list of integer token ids (one prompt a request)");
    for (value.array) |t| if (t != .int) return cx.refuse("tokens must be a list of integer token ids (one prompt a request)");
    const vocab = srv.text.vocabSize();
    const out = try cx.a.alloc(u32, value.array.len);
    for (value.array, out) |t, *slot| {
        const n = t.int64() orelse -1;
        if (n < 0 or n >= vocab) return cx.fail(.request, "tokens token ids must be in the vocabulary's range 0 to {d}", .{@as(i64, vocab) - 1});
        slot.* = @intCast(n);
    }
    const text = srv.text.decode(cx.a, out) catch return error.OutOfMemory;
    const o = try json.newObject(cx.a);
    try o.put(cx.a, "prompt", .{ .string = text });
    return .{ .object = o };
}

fn tokenRequest(srv: *Server, cx: *Cx, body: Value, detok: bool, gone: anytype) errors.Refused!Value {
    const ticket = try srv.gate.request(srv.io, cx, body, gone);
    defer srv.gate.leave(srv.io, ticket);
    return if (detok) detokenize(srv, cx, body) else tokenize(srv, cx, body);
}

pub fn post(srv: *Server, conn: *Conn, a: Allocator, detok: bool) void {
    var cx: Cx = .{ .a = a };
    const reply = blk: {
        const body = http_body.readJson(conn, &cx) catch break :blk null;
        break :blk tokenRequest(srv, &cx, body, detok, @import("openai.zig").Gone{ .conn = conn }) catch null;
    };
    if (reply) |r| return routes.sendValue(conn, a, 200, r);
    const o = json.newObject(a) catch return;
    o.put(a, "message", .{ .string = cx.message }) catch return;
    o.put(a, "type", .{ .string = "invalid_request_error" }) catch return;
    const wrapped = json.newObject(a) catch return;
    wrapped.put(a, "error", .{ .object = o }) catch return;
    routes.sendValue(conn, a, if (cx.kind == .capacity) 503 else 400, .{ .object = wrapped });
}

test "prepare floor covers chat tokenize raw tokenize and detokenize before allocation" {
    const Test = struct {
        fn memory(_: std.Io) ?u64 {
            return 0;
        }
        pub fn check(_: @This()) bool {
            return true;
        }
    };
    var srv: Server = undefined;
    srv.io = std.testing.io;
    srv.gate = .init(3);
    srv.gate.avail = Test.memory;
    srv.gate.inflight = 1;
    srv.gate.reserved = 24;
    for ([_]bool{ false, true }) |detok| {
        var cx: Cx = .{ .a = std.testing.allocator };
        try std.testing.expectError(error.Refused, tokenRequest(&srv, &cx, .null, detok, Test{}));
        try std.testing.expectEqual(errors.Kind.capacity, cx.kind);
    }
    try std.testing.expectEqual(@as(u32, 1), srv.gate.inflight);
    try std.testing.expectEqual(@as(u64, 24), srv.gate.reserved);
}

test "Responses token counting preserves native translation and refuses hidden history" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const parsed = try json.parse(cx.a,
        \\{ "input": [{"role":"user","content":"hello"}], "instructions":"keep constraints" }
    );
    const raw = parsed.ok;
    const translated = try responseTokenBody(&cx, raw);
    const msgs = translated.get("messages").?.array;
    try std.testing.expectEqual(@as(usize, 2), msgs.len);
    try std.testing.expectEqualStrings("system", msgs[0].strField("role").?);
    try std.testing.expectEqualStrings("user", msgs[1].strField("role").?);
    const bad = (try json.parse(cx.a,
        \\{"input":"hello","previous_response_id":"hidden"}
    )).ok;
    try std.testing.expectError(error.Refused, responseTokenBody(&cx, bad));
}
