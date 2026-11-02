//! An engine's effort numbers (Info.efforts): each named effort passes as asked and the template hears its number;
//! without them the template hears the nearest name it quotes, as before.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;

/// DeepSeek-V4.1's numbers (families/dsv41/text.zig; the served lane's EFFORT_VALUES).
const numbers = [_]api.Effort{
    .{ .name = "minimal", .value = 25 }, .{ .name = "low", .value = 50 },   .{ .name = "medium", .value = 62 },
    .{ .name = "high", .value = 75 },    .{ .name = "xhigh", .value = 88 }, .{ .name = "max", .value = 100 },
};

/// Renders what the template would hear: the thinking switch and the effort as a number, a name or nothing.
const EffortText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, o: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        if (o.reasoning_effort_value) |v| return std.fmt.allocPrint(a, "thinking={} effort={d}\n", .{ o.enable_thinking, v });
        if (o.reasoning_effort) |e| return std.fmt.allocPrint(a, "thinking={} effort={s}\n", .{ o.enable_thinking, e });
        return std.fmt.allocPrint(a, "thinking={} effort=-\n", .{o.enable_thinking});
    }

    /// The names DeepSeek-V4.1's template quotes.
    fn templateSource(_: *anyopaque) []const u8 {
        return "{% if e == \"low\" %}{% elif e == \"high\" %}{% elif e == \"max\" %}{% endif %}";
    }
};

/// Copies the prompt the server submits, then finishes at once; reports the effort numbers when it has them.
const Capture = struct {
    prompt: []u32 = &.{},
    cache_salt: bool = false,
    cache_namespace: ?[32]u8 = null,
    efforts: []const api.Effort = &.{},
    resets: usize = 0,
    reset_error: ?api.ResetCacheError = null,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .reset_cache = reset } };
    }

    fn reset(ctx: *anyopaque) api.ResetCacheError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        if (e.reset_error) |err| return err;
        e.resets += 1;
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        return .{ .efforts = e.efforts, .cache_salt = e.cache_salt };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        std.testing.allocator.free(e.prompt);
        e.prompt = std.testing.allocator.dupe(u32, request.prompt) catch return error.Busy;
        e.cache_namespace = request.cache_namespace;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
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

fn post(srv: *server.Server, body: []const u8) !u16 {
    return exchange(srv, "POST", "/v1/chat/completions", body);
}

fn exchange(srv: *server.Server, method: []const u8, path: []const u8, body: []const u8) !u16 {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = try std.fmt.allocPrint(std.testing.allocator, "{s} {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ method, path, body.len });
    defer std.testing.allocator.free(head);
    for ([_][]const u8{ head, body }) |part| {
        var sent: usize = 0;
        while (sent < part.len) {
            const n = std.c.write(pair[0], part[sent..].ptr, part.len - sent);
            if (n <= 0) return error.Closed;
            sent += @intCast(n);
        }
    }
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    if (try conn.readRequest(arena.allocator(), false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, arena.allocator());
    var tmp: [4096]u8 = undefined;
    var pollfd = [_]std.posix.pollfd{.{ .fd = pair[0], .events = std.posix.POLL.IN, .revents = 0 }};
    if (try std.posix.poll(&pollfd, 500) == 0) return error.NoReply;
    const n = try std.posix.read(pair[0], &tmp);
    if (n < "HTTP/1.1 200".len) return error.BadReply;
    return std.fmt.parseInt(u16, tmp["HTTP/1.1 ".len..][0..3], 10);
}

fn heard(backend: *const Capture) ![]u8 {
    const text = try std.testing.allocator.alloc(u8, backend.prompt.len);
    for (backend.prompt, text) |id, *byte| byte.* = @intCast(id);
    return text;
}

fn expectHeard(srv: *server.Server, backend: *const Capture, body: []const u8, want: []const u8) !void {
    try std.testing.expectEqual(@as(u16, 200), try post(srv, body));
    const text = try heard(backend);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings(want, text);
}

test "an engine's effort numbers: every name passes as asked and the template hears its number" {
    var text: EffortText = .{};
    var backend: Capture = .{ .efforts = &numbers };
    defer std.testing.allocator.free(backend.prompt);
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    defer srv.deinit();
    try std.testing.expectEqual(@as(usize, 6), srv.effort_levels.len);
    const ask = "{{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"reasoning_effort\":\"{s}\"}}";
    for (numbers) |e| {
        const body = try std.fmt.allocPrint(std.testing.allocator, ask, .{e.name});
        defer std.testing.allocator.free(body);
        const want = try std.fmt.allocPrint(std.testing.allocator, "thinking=true effort={d}\n", .{e.value});
        defer std.testing.allocator.free(want);
        try expectHeard(srv, &backend, body, want);
    }
    // no effort: the template's own default; none: thinking off
    try expectHeard(srv, &backend, "{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", "thinking=true effort=-\n");
    try expectHeard(srv, &backend, "{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"none\"}", "thinking=false effort=-\n");
}

test "a request's own effort number from 1 to 100 reaches the template as is; names keep their numbers" {
    var text: EffortText = .{};
    var backend: Capture = .{ .efforts = &numbers };
    defer std.testing.allocator.free(backend.prompt);
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    defer srv.deinit();
    const ask = "{{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"reasoning_effort\":{s}}}";
    for ([_][2][]const u8{ .{ "40", "40" }, .{ "\"40\"", "40" }, .{ "1", "1" }, .{ "100", "100" }, .{ "\"high\"", "75" } }) |case| {
        const body = try std.fmt.allocPrint(std.testing.allocator, ask, .{case[0]});
        defer std.testing.allocator.free(body);
        const want = try std.fmt.allocPrint(std.testing.allocator, "thinking=true effort={s}\n", .{case[1]});
        defer std.testing.allocator.free(want);
        try expectHeard(srv, &backend, body, want);
    }
    for ([_][]const u8{ "0", "101", "-5", "\"40.5\"", "40.5", "\"loud\"", "true" }) |bad| {
        const body = try std.fmt.allocPrint(std.testing.allocator, ask, .{bad});
        defer std.testing.allocator.free(body);
        try std.testing.expectEqual(@as(u16, 400), try post(srv, body));
    }
    try std.testing.expect(@import("fields.zig").numericEffort("62") == 62);
    try std.testing.expect(@import("fields.zig").numericEffort("high") == null);
}

test "benchmark cache reset is opt-in, authenticated and reports backend refusals" {
    var text: EffortText = .{};
    var backend: Capture = .{ .efforts = &numbers };
    defer std.testing.allocator.free(backend.prompt);
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .reasoning_effort = "max" }, null);
    defer srv.deinit();
    try expectHeard(srv, &backend, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", "thinking=true effort=100\n");
    try std.testing.expectEqual(@as(u16, 404), try exchange(srv, "POST", "/reset_prefix_cache", "{}"));
    try std.testing.expectEqual(@as(u16, 404), try exchange(srv, "GET", "/benchmark/status", ""));
    try std.testing.expectEqual(@as(usize, 0), backend.resets);
    srv.config.benchmark_control = true;
    srv.preparing.store(1, .release);
    try std.testing.expectEqual(@as(u16, 409), try exchange(srv, "POST", "/reset_prefix_cache", "{}"));
    srv.preparing.store(0, .release);
    for ([_]struct { err: api.ResetCacheError, code: u16 }{
        .{ .err = error.Busy, .code = 409 }, .{ .err = error.Closed, .code = 503 }, .{ .err = error.Unsupported, .code = 501 },
    }) |case| {
        backend.reset_error = case.err;
        try std.testing.expectEqual(case.code, try exchange(srv, "POST", "/reset_prefix_cache", "{}"));
    }
    try std.testing.expectEqual(@as(usize, 0), backend.resets);
    backend.reset_error = null;
    try std.testing.expectEqual(@as(u16, 200), try exchange(srv, "POST", "/reset_prefix_cache", "{}"));
    try std.testing.expectEqual(@as(usize, 1), backend.resets);
    try std.testing.expectEqual(@as(u16, 200), try exchange(srv, "GET", "/benchmark/status", ""));
    try std.testing.expect(server.auth.gated("/reset_prefix_cache", true));
    try std.testing.expect(server.auth.gated("/benchmark/status", true));
}

test "cache salt reaches engine request without changing rendered tokens and unsupported engines refuse it" {
    var text: EffortText = .{};
    var backend: Capture = .{ .efforts = &numbers, .cache_salt = true };
    defer std.testing.allocator.free(backend.prompt);
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .reasoning_effort = "max" }, null);
    defer srv.deinit();
    const salted = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"cache_salt\":\"bench-A\"}";
    try expectHeard(srv, &backend, salted, "thinking=true effort=100\n");
    try std.testing.expect(std.meta.eql(api.prompt_cache.namespaceFor("bench-A"), backend.cache_namespace));
    try expectHeard(srv, &backend, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", "thinking=true effort=100\n");
    try std.testing.expect(backend.cache_namespace == null);
    // Legacy completion prompts take the same namespace, without inserting it
    // into prompt bytes or accidentally enabling chat reasoning.
    try std.testing.expectEqual(@as(u16, 200), try exchange(srv, "POST", "/v1/completions", "{\"model\":\"m\",\"prompt\":\"hello\",\"cache_salt\":\"bench-B\"}"));
    try std.testing.expect(std.meta.eql(api.prompt_cache.namespaceFor("bench-B"), backend.cache_namespace));
    try std.testing.expectEqualSlices(u32, &.{ 'h', 'e', 'l', 'l', 'o' }, backend.prompt);
    for ([_][]const u8{ "\"\"", "42", "[]", "true" }) |bad| {
        const body = try std.fmt.allocPrint(std.testing.allocator, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"cache_salt\":{s}}}", .{bad});
        defer std.testing.allocator.free(body);
        try std.testing.expectEqual(@as(u16, 400), try post(srv, body));
    }
    srv.info.cache_salt = false;
    try std.testing.expectEqual(@as(u16, 400), try post(srv, salted));
}

test "without effort numbers the template hears the nearest name it quotes, as before" {
    var text: EffortText = .{};
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    defer srv.deinit();
    try std.testing.expectEqual(@as(usize, 3), srv.effort_levels.len);
    try expectHeard(srv, &backend, "{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"high\"}", "thinking=true effort=high\n");
    const coerced = srv.effortFor("minimal").?;
    const body = try std.fmt.allocPrint(std.testing.allocator, "{{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"reasoning_effort\":\"minimal\"}}", .{});
    defer std.testing.allocator.free(body);
    const want = try std.fmt.allocPrint(std.testing.allocator, "thinking=true effort={s}\n", .{coerced});
    defer std.testing.allocator.free(want);
    try expectHeard(srv, &backend, body, want);
}
