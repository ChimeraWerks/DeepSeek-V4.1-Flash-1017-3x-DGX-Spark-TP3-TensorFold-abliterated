//! A streamed context-window refusal is an error event, and every other refusal is a 400 before the stream opens.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const anthropic_translate = @import("anthropic_translate.zig");
const responses_translate = @import("responses_translate.zig");
const errors = @import("errors.zig");
const Conn = @import("http_conn.zig").Conn;

/// Bytes as tokens; every prompt renders as "prompt" (6 tokens).
const TestText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .special = special } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var decoded: std.ArrayList(u8) = .empty;
        for (ids) |id| {
            if (id == 256) try decoded.appendSlice(a, "</think>") else if (id == 257) try decoded.appendSlice(a, "｜DSML｜") else try decoded.append(a, @intCast(id));
        }
        var at: usize = 0;
        while (at < decoded.items.len) {
            const n = std.unicode.utf8ByteSequenceLength(decoded.items[at]) catch 1;
            if (at + n > decoded.items.len) return std.mem.concat(a, u8, &.{ decoded.items[0..at], "\u{fffd}" });
            at += n;
        }
        return decoded.items;
    }

    fn tokenId(_: *anyopaque, piece: []const u8) ?u32 {
        return if (std.mem.eql(u8, piece, "</think>")) 256 else null;
    }

    fn special(_: *anyopaque, id: u32) bool {
        return id == 256 or id == 257;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 258;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An 8-token window whose engine answers "hi" at once.
const HiEngine = struct {
    tokens: []const u32 = &.{ 'h', 'i' },
    window: u32 = 8,
    submissions: usize = 0,
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *HiEngine = @ptrCast(@alignCast(ctx));
        return .{ .context_window = e.window };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *HiEngine = @ptrCast(@alignCast(ctx));
        e.submissions += 1;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.tokens) |token| sink.event(sink.ctx, id, &.{ .tokens = &.{token} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// What the route's output saw: a whole reply's status, error code and sampling label, or a stream and its first role.
const Seen = struct {
    tokens: ?[]const u32 = null,
    submissions: usize = 0,
    answer: [2048]u8 = undefined,
    answer_len: usize = 0,
    reasoning: [2048]u8 = undefined,
    reasoning_len: usize = 0,
    calls: usize = 0,

    gone_at_open: bool = false, // the client is gone when the stream opens
    opened: bool = false,
    events: usize = 0,
    role_first: bool = false,
    status: ?u16 = null,
    context_code: bool = false,
    label: ?enum { exact, greedy } = null,
    preparing: i64 = 0,

    fn capture(c: *Seen, message: json.Value) void {
        if (message.strField("content")) |value| {
            const n = @min(value.len, c.answer.len - c.answer_len);
            @memcpy(c.answer[c.answer_len..][0..n], value[0..n]);
            c.answer_len += n;
        }
        if (message.strField("reasoning_content")) |value| {
            const n = @min(value.len, c.reasoning.len - c.reasoning_len);
            @memcpy(c.reasoning[c.reasoning_len..][0..n], value[0..n]);
            c.reasoning_len += n;
        }
        if (message.get("tool_calls")) |calls| if (calls == .array) {
            c.calls += calls.array.len;
        };
    }

    fn out(c: *Seen) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(ctx: *anyopaque) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        if (c.gone_at_open) return error.Closed;
        c.opened = true;
    }

    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.events += 1;
        const p = payload orelse return;
        if (p.get("error")) |err| {
            const code = err.get("code") orelse return;
            c.context_code = code == .string and std.mem.eql(u8, code.string, "context_length_exceeded");
            return;
        }
        const choices = p.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const delta = choices.array[0].get("delta") orelse return;
        c.capture(delta);
        if (c.events != 1) return;
        const role = delta.get("role") orelse return;
        c.role_first = role == .string and std.mem.eql(u8, role.string, "assistant");
    }

    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.status = status;
        if (payload.get("choices")) |choices| if (choices == .array and choices.array.len > 0) if (choices.array[0].get("message")) |message| c.capture(message);
        if (payload.get("tensorfold")) |t| if (t.get("sampling")) |l| if (l == .string) {
            c.label = if (std.mem.eql(u8, l.string, "exact")) .exact else if (std.mem.eql(u8, l.string, "greedy")) .greedy else null;
        };
        const code = (payload.get("error") orelse return).get("code") orelse return;
        c.context_code = code == .string and std.mem.eql(u8, code.string, "context_length_exceeded");
    }
};

const Route = enum { chat, messages, responses };

/// A Responses request with no stored conversation to read.
const NoHistory = struct {
    pub fn conversation(_: NoHistory, cx: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return cx.refuse("no stored responses in this test");
    }
};

/// `body` through `route`'s translation and the chat handler, as the route runs it; the preparing count after it.
fn send(route: Route, body: []const u8) !Seen {
    return sendTo(route, body, .{});
}

fn sendTo(route: Route, body: []const u8, start: Seen) !Seen {
    var text: TestText = .{};
    var backend: HiEngine = .{};
    if (start.tokens) |tokens| {
        backend.tokens = tokens;
        backend.window = 8192;
    }
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, body)).ok;
    var cx: errors.Cx = .{ .a = a };
    const chat = switch (route) {
        .chat => raw,
        .messages => try anthropic_translate.translate(&cx, raw, false),
        .responses => (try responses_translate.translate(&cx, raw, NoHistory{})).chat,
    };
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen = start;
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, chat);
    seen.preparing = srv.preparing.load(.acquire);
    seen.submissions = backend.submissions;
    return seen;
}

fn expectContextInStream(seen: Seen, role: bool) !void {
    try std.testing.expect(seen.opened);
    try std.testing.expectEqual(role, seen.role_first);
    try std.testing.expect(seen.context_code);
    try std.testing.expectEqual(@as(?u16, null), seen.status);
    try std.testing.expect(seen.events >= 2);
    try std.testing.expectEqual(@as(i64, 0), seen.preparing);
}

test "a streamed chat over the window reports context_length_exceeded after the role chunk" {
    try expectContextInStream(try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":100}"), true);
}

test "a streamed chat with tools over the window reports the context error with no role chunk" {
    try expectContextInStream(try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":100," ++
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}]}"), false);
}

test "a streamed Messages request over the window reports the context error in the stream" {
    try expectContextInStream(try send(.messages, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":100}"), true);
}

test "a streamed Responses request over the window reports the context error in the stream" {
    try expectContextInStream(try send(.responses, "{\"model\":\"test-model\",\"input\":\"x\",\"stream\":true,\"max_output_tokens\":100}"), true);
}

test "a valid streamed chat still opens with the role chunk and gives the preparing count back" {
    const seen = try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":2}");
    try std.testing.expect(seen.opened);
    try std.testing.expect(seen.role_first);
    try std.testing.expect(seen.events >= 3); // the role, the text, the last chunk
    try std.testing.expectEqual(@as(?u16, null), seen.status);
    try std.testing.expectEqual(@as(i64, 0), seen.preparing);
}

test "a stream whose client is gone when it opens gives the preparing count back" {
    const body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":2}";
    const seen = try sendTo(.chat, body, .{ .gone_at_open = true });
    try std.testing.expect(!seen.opened);
    try std.testing.expectEqual(@as(i64, 0), seen.preparing);
}

test "a reply labels a greedy request greedy and a sampled one exact" {
    const greedy = try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2,\"temperature\":0}");
    try std.testing.expectEqual(@as(?u16, 200), greedy.status);
    try std.testing.expectEqual(.greedy, greedy.label.?);
    const sampled = try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2,\"temperature\":0.7,\"seed\":1}");
    try std.testing.expectEqual(.exact, sampled.label.?);
    try std.testing.expectEqual(@as(i64, 0), sampled.preparing);
}

// Streamed tool calls on the real server route, with a CPU fake engine (no model or GPU).
const SpaceEngine = struct {
    answer: []const u8,
    chunk_size: usize,
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = HiEngine.cancel, .status = HiEngine.status, .memory = HiEngine.memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *SpaceEngine = @ptrCast(@alignCast(ctx));
        var tokens: [2048]u32 = undefined;
        var len: usize = 0;
        var byte: usize = 0;
        while (byte < e.answer.len) {
            if (std.mem.startsWith(u8, e.answer[byte..], "｜DSML｜")) {
                tokens[len] = 257;
                byte += "｜DSML｜".len;
            } else {
                tokens[len] = e.answer[byte];
                byte += 1;
            }
            len += 1;
        }
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var pos: usize = 0;
        while (pos < len) {
            const end = @min(len, pos + e.chunk_size);
            sink.event(sink.ctx, id, &.{ .tokens = tokens[pos..end] });
            pos = end;
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
};

const SpaceSeen = struct {
    a: std.mem.Allocator,
    content: std.ArrayList(u8) = .empty,
    calls: usize = 0,
    call_finish: bool = false,
    failed: bool = false,
    fn open(_: *anyopaque) error{Closed}!void {}
    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const s: *SpaceSeen = @ptrCast(@alignCast(ctx));
        const value = payload orelse return;
        if (value.get("error") != null) s.failed = true;
        const choices = value.get("choices") orelse return;
        for (choices.array) |choice| {
            if (choice.get("finish_reason")) |f| {
                if (f == .string and std.mem.eql(u8, f.string, "tool_calls")) s.call_finish = true;
            }
            const delta = choice.get("delta") orelse continue;
            if (delta.get("content")) |c| if (c == .string) {
                s.content.appendSlice(s.a, c.string) catch return error.Closed;
            };
            if (delta.get("tool_calls")) |c| s.calls += c.array.len;
        }
    }
    fn reply(ctx: *anyopaque, _: u16, _: json.Value) void {
        const s: *SpaceSeen = @ptrCast(@alignCast(ctx));
        s.failed = true; // every case below requests a stream
    }
};

fn exerciseSpace(answer: []const u8, expected: []const u8, called: bool, tools: bool, chunk_size: usize) !void {
    var text: TestText = .{};
    var backend: SpaceEngine = .{ .answer = answer, .chunk_size = chunk_size };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tool_list = if (tools) ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"}}}}}]" else "";
    const body = try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":2048", tool_list, "}" });
    const raw = (try json.parse(a, body)).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: SpaceSeen = .{ .a = a };
    openai.run(srv, a, .{ .ctx = &seen, .vt = &.{ .open = SpaceSeen.open, .event = SpaceSeen.event, .reply = SpaceSeen.reply } }, .{ .conn = &conn }, true, raw);
    try std.testing.expect(!seen.failed);
    try std.testing.expectEqual(called, seen.call_finish);
    try std.testing.expectEqual(called, seen.calls > 0);
    try std.testing.expectEqualStrings(expected, seen.content.items);
}

const space_call = "<\u{ff5c}DSML\u{ff5c} calls><\u{ff5c}DSML\u{ff5c} invoke name=\"f\"><\u{ff5c}DSML\u{ff5c} parameter name=\"x\" string=\"false\">7</\u{ff5c}DSML\u{ff5c} parameter></\u{ff5c}DSML\u{ff5c} invoke></\u{ff5c}DSML\u{ff5c} calls>";

test "pure DSML call never streams framing whitespace" {
    for ([_]usize{ 1, 2, 7, 2048 }) |n| try exerciseSpace("\n\n" ++ space_call, "", true, true, n);
}
test "leading indentation in ordinary prose is retained with tools" {
    for ([_]usize{ 1, 2, 7, 2048 }) |n| try exerciseSpace("\n\n    indented\n", "\n\n    indented\n", false, true, n);
}
test "whitespace only answer is retained when no call is parsed" {
    try exerciseSpace("\n \t", "\n \t", false, true, 1);
}
test "whitespace with tools disabled is unchanged" {
    try exerciseSpace("\n\n    indented\n", "\n\n    indented\n", false, false, 1);
}
test "prose before a call retains its leading whitespace" {
    try exerciseSpace("\n\nLook up.\n" ++ space_call, "\n\nLook up.\n", true, true, 1);
}

test "invalid tool histories and duplicate names refuse before opening Chat and Messages streams" {
    for ([_][]const u8{
        "{\"model\":\"test-model\",\"messages\":[{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"a\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"wrong\",\"content\":\"bad\"}],\"stream\":true,\"max_tokens\":1}",
        "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[{\"function\":{\"name\":\"f\"}},{\"function\":{\"name\":\"f\"}}],\"stream\":true,\"max_tokens\":1}",
    }) |body| {
        const seen = try send(.chat, body);
        try std.testing.expectEqual(@as(?u16, 400), seen.status);
        try std.testing.expect(!seen.opened);
        try std.testing.expectEqual(@as(usize, 0), seen.events);
        try std.testing.expectEqual(@as(usize, 0), seen.submissions);
    }
    const seen = try send(.messages, "{\"model\":\"test-model\",\"max_tokens\":1,\"stream\":true,\"messages\":[{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"a\",\"name\":\"f\",\"input\":{}}]},{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"wrong\",\"content\":\"bad\"}]}]}");
    try std.testing.expectEqual(@as(?u16, 400), seen.status);
    try std.testing.expect(!seen.opened);
    try std.testing.expectEqual(@as(usize, 0), seen.events);
}

test "token markers stay literal through Chat Messages and Responses in streamed and final replies" {
    const thought = "literal </think> and <｜DSML｜ calls> remain thoughts.";
    const answer = "literal <｜DSML｜ calls><｜DSML｜ invoke name=\"f\"></｜DSML｜ invoke></｜DSML｜ calls> remains text.";
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokens: std.ArrayList(u32) = .empty;
    for (thought) |byte| try tokens.append(a, byte);
    try tokens.append(a, 256);
    for (answer) |byte| try tokens.append(a, byte);
    for ([_]bool{ false, true }) |stream| for ([_]Route{ .chat, .messages, .responses }) |route| {
        const body = switch (route) {
            .chat => "{\"model\":\"test-model\",\"max_tokens\":1024,\"thinking\":{\"type\":\"enabled\"},\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}",
            .messages => "{\"model\":\"test-model\",\"max_tokens\":1024,\"thinking\":{\"type\":\"enabled\"},\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[{\"name\":\"f\",\"input_schema\":{\"type\":\"object\"}}]}",
            .responses => "{\"model\":\"test-model\",\"max_output_tokens\":1024,\"reasoning\":{\"effort\":\"high\"},\"input\":\"x\",\"tools\":[{\"type\":\"function\",\"name\":\"f\"}]}",
        };
        const request = (try json.parse(a, body)).ok;
        try request.object.put(a, "stream", .{ .bool = stream });
        const seen = try sendTo(route, try json.stringify(a, request, .{}), .{ .tokens = tokens.items });
        try std.testing.expectEqual(@as(usize, 1), seen.submissions);
        try std.testing.expectEqualStrings(thought, seen.reasoning[0..seen.reasoning_len]);
        try std.testing.expectEqualStrings(answer, seen.answer[0..seen.answer_len]);
        try std.testing.expectEqual(@as(usize, 0), seen.calls);
    };
}

/// An engine silent for `quiet_ms` (a long cold prefill), then answering "hi" from its own thread.
const QuietEngine = struct {
    quiet_ms: i64 = 400,
    thread: ?std.Thread = null,
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = HiEngine.cancel, .status = HiEngine.status, .memory = HiEngine.memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *QuietEngine = @ptrCast(@alignCast(ctx));
        e.thread = std.Thread.spawn(.{}, answer, .{ e, id, sink }) catch return error.Busy;
    }
    fn answer(e: *QuietEngine, id: api.Id, sink: api.Sink) void {
        std.Io.sleep(std.testing.io, .fromMilliseconds(e.quiet_ms), .awake) catch {};
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for ("hi") |byte| sink.event(sink.ctx, id, &.{ .tokens = &.{byte} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
};

/// Reads a socket into `out` until its other end closes.
fn drainSocket(fd: std.posix.fd_t, out: *std.ArrayList(u8)) void {
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = std.posix.read(fd, &buf) catch return;
        if (n == 0) return;
        out.appendSlice(std.heap.page_allocator, buf[0..n]) catch return;
    }
}

test "a silent streamed Responses request sends a ping comment before its first delta" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const chat_mod = @import("chat.zig");
    const saved = chat_mod.ping_after_ns;
    defer chat_mod.ping_after_ns = saved;
    chat_mod.ping_after_ns = 100 * std.time.ns_per_ms;
    var text: TestText = .{};
    var backend: QuietEngine = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = "{\"model\":\"test-model\",\"input\":\"x\",\"stream\":true,\"store\":false,\"max_output_tokens\":16}";
    const linux = std.os.linux;
    var fds: [2]i32 = undefined;
    if (linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.STREAM, 0, &fds)) != .SUCCESS) return error.SocketPair;
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(std.heap.page_allocator);
    const reader = try std.Thread.spawn(.{}, drainSocket, .{ fds[1], &got });
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var conn: Conn = .{ .fd = fds[0], .peer = "", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .headers = &headers, .path = "/v1/responses", .method = "POST", .version = "HTTP/1.1" };
    @import("responses.zig").post(srv, &conn, a);
    if (backend.thread) |t| t.join();
    _ = std.posix.system.close(fds[0]);
    reader.join();
    _ = std.posix.system.close(fds[1]);
    const bytes = got.items;
    const ping = std.mem.indexOf(u8, bytes, "\n: ping\n\n") orelse return error.NoPing;
    const created = std.mem.indexOf(u8, bytes, "response.created") orelse return error.NoCreated;
    const delta = std.mem.indexOf(u8, bytes, "response.output_text.delta") orelse return error.NoDelta;
    try std.testing.expect(created < ping and ping < delta);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "response.completed") != null);
}
