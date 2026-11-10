//! A Responses request as the chat completion that runs it (``responses_translate.translate``).
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const pyrepr = @import("pyrepr.zig");
const tool_specs = @import("tool_specs.zig");
const response_tools = @import("responses_tools.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Fields a chat completion reads as they are: OpenAI's and this server's own.
const passed = [_][]const u8{ "model", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty", "repetition_penalty_window", "penalty_include_prompt", "seed", "stream", "parallel_tool_calls", "stop", "draft", "thinking_budget", "ignore_eos", "priority", "return_token_ids", "chat_template_kwargs" };
const refused = [_][2][]const u8{
    .{ "background", "background responses are not supported: send the request and wait for it" },
    .{ "conversation", "conversations are not supported: send previous_response_id or the items" },
    .{ "prompt", "prompt templates are not supported: send input and instructions" },
    .{ "context_management", "context management is not supported" },
};

pub const Request = struct {
    chat: Value,
    echo: *json.Object,
    added: []Value,
    store: bool,
    stream: bool,
    tools: response_tools.Set,
};

fn kindOf(v: Value) ?Value {
    return if (v == .object) v.get("type") else null;
}

/// A message's content as chat content: text parts as text, images as image_url parts.
fn content(cx: *Cx, c: ?Value) errors.Refused!Value {
    const a = cx.a;
    const value = c orelse Value.null;
    if (value == .string) return value;
    if (value != .array) return cx.refuse("message content must be a string or a list of content parts");
    var parts: std.ArrayList(Value) = .empty;
    for (value.array) |part| {
        const kind = kindOf(part);
        const k = if (kind != null and kind.? == .string) kind.?.string else "";
        const is = struct {
            fn any(s: []const u8, names: []const []const u8) bool {
                for (names) |n| if (std.mem.eql(u8, s, n)) return true;
                return false;
            }
        };
        if (kind != null and kind.? == .string and is.any(k, &.{ "input_text", "output_text", "text" }) and part.strField("text") != null) {
            try parts.append(a, try textPart(a, part.strField("text").?));
        } else if (kind != null and kind.? == .string and std.mem.eql(u8, k, "refusal") and part.strField("refusal") != null) {
            try parts.append(a, try textPart(a, part.strField("refusal").?));
        } else if (kind != null and kind.? == .string and std.mem.eql(u8, k, "input_image")) {
            const url_text = part.strField("image_url") orelse return cx.refuse("input_image needs an image_url (a URL or data URL); file ids are not supported");
            const url = try json.newObject(a);
            try url.put(a, "url", .{ .string = url_text });
            if (json.truthyField(part, "detail")) try url.put(a, "detail", part.get("detail").?);
            const p = try json.newObject(a);
            try p.put(a, "type", .{ .string = "image_url" });
            try p.put(a, "image_url", .{ .object = url });
            try parts.append(a, .{ .object = p });
        } else if (kind != null and kind.? == .string and (std.mem.eql(u8, k, "input_file") or std.mem.eql(u8, k, "file"))) {
            // a PDF as the chat file part pdf.zig renders, from inline file_data only (TensorFold upstream, PR #1820)
            const inner = if (part.get("file")) |f| (if (f == .object) f.object else return cx.refuse("file must be an object")) else blk: {
                const f = try json.newObject(a);
                if (part.get("filename")) |v| try f.put(a, "filename", v);
                if (part.get("file_data")) |v| try f.put(a, "file_data", v);
                if (part.get("file_url")) |v| try f.put(a, "url", v);
                if (part.get("file_id")) |v| try f.put(a, "file_id", v);
                break :blk f;
            };
            const iv: Value = .{ .object = inner };
            if (iv.get("file_id") != null and iv.get("file_data") == null and iv.get("data") == null and iv.get("url") == null and iv.get("file_url") == null)
                return cx.refuse("file parts require inline file_data (base64 or data:application/pdf;base64,...); file_id is not supported");
            if (iv.strField("url") orelse iv.strField("file_url")) |u| {
                const trimmed = std.mem.trim(u8, u, " \t\r\n");
                if (trimmed.len > 0 and !std.ascii.startsWithIgnoreCase(trimmed, "data:")) return cx.refuse(@import("pdf.zig").url_refusal);
            }
            if ((iv.strField("file_data") orelse iv.strField("data") orelse iv.strField("url") orelse iv.strField("file_url")) == null)
                return cx.fail(.request, "content parts of type {s} are not supported: send input_text, input_image or input_file", .{try pyrepr.repr(a, kind)});
            const p = try json.newObject(a);
            try p.put(a, "type", .{ .string = "file" });
            try p.put(a, "file", .{ .object = inner });
            try parts.append(a, .{ .object = p });
        } else return cx.fail(.request, "content parts of type {s} are not supported: send input_text or input_image", .{try pyrepr.repr(a, kind)});
    }
    return .{ .array = parts.items };
}

const textPart = @import("messages.zig").textPart;

/// A function_call_output's output as a tool message's content.
fn output(cx: *Cx, o: ?Value) errors.Refused!Value {
    const value = o orelse Value.null;
    if (value == .string) return value;
    if (value == .array) {
        var texts_only = true;
        var parts_only = true;
        for (value.array) |p| {
            const k = kindOf(p);
            const name = if (k != null and k.? == .string) k.?.string else "";
            if (!std.mem.eql(u8, name, "input_text")) texts_only = false;
            if (!std.mem.eql(u8, name, "input_text") and !std.mem.eql(u8, name, "input_image") and !std.mem.eql(u8, name, "input_file")) parts_only = false;
        }
        if (texts_only) {
            var joined: std.ArrayList(u8) = .empty;
            for (value.array) |p| if (p.get("text")) |t| if (t.truthy()) try joined.appendSlice(cx.a, try tool_specs.pyStr(cx.a, t));
            return .{ .string = joined.items };
        }
        if (parts_only) return content(cx, value);
    }
    return cx.refuse("a function_call_output's output must be a string or input_text and input_image parts");
}

/// Input items (output items too) as chat messages: a turn's reasoning, text and calls are one assistant message.
pub fn messages(cx: *Cx, items: []const Value) errors.Refused![]Value {
    const a = cx.a;
    var out: std.ArrayList(*json.Object) = .empty;
    var reasoning_open = false;
    for (items) |item| {
        if (item != .object) return cx.refuse("each input item must be an object");
        var kind: ?Value = item.get("type");
        if (kind == null or !kind.?.truthy()) kind = if (item.has("role")) Value{ .string = "message" } else null;
        const k = if (kind != null and kind.? == .string) kind.?.string else "";
        if (std.mem.eql(u8, k, "message")) {
            const role = item.strField("role") orelse "";
            const ok = for ([_][]const u8{ "user", "assistant", "system", "developer" }) |r| {
                if (std.mem.eql(u8, r, role)) break true;
            } else false;
            if (!ok or item.get("role").? != .string) return cx.refuse("a message's role must be user, assistant, system or developer");
            const reuse = reasoning_open and std.mem.eql(u8, role, "assistant");
            const m = if (reuse) out.items[out.items.len - 1] else try json.newObject(a);
            try m.put(a, "role", .{ .string = role });
            try m.put(a, "content", try content(cx, item.get("content")));
            if (!reuse) try out.append(a, m);
            reasoning_open = false;
        } else if (std.mem.eql(u8, k, "function_call") or std.mem.eql(u8, k, "custom_tool_call")) {
            const call_id = item.strField("call_id");
            const name = item.strField("name");
            if (call_id == null or name == null) return cx.refuse("a function_call item needs a call_id and a name");
            const function = try json.newObject(a);
            try function.put(a, "name", .{ .string = name.? });
            if (std.mem.eql(u8, k, "custom_tool_call")) {
                const input = item.strField("input") orelse return cx.refuse("a custom_tool_call needs string input");
                const args = try json.newObject(a);
                try args.put(a, "input", .{ .string = input });
                try function.put(a, "arguments", .{ .string = try json.stringify(a, .{ .object = args }, .{ .ascii = false }) });
            } else {
                const args = item.get("arguments");
                try function.put(a, "arguments", if (args != null and args.?.truthy()) args.? else .{ .string = "{}" });
                if (item.field("namespace")) |ns| {
                    if (ns != .string) return cx.refuse("a function_call namespace must be a string");
                    try function.put(a, "namespace", ns);
                }
            }
            const call = try json.newObject(a);
            try call.put(a, "id", .{ .string = call_id.? });
            try call.put(a, "type", .{ .string = "function" });
            try call.put(a, "function", .{ .object = function });
            const last_role = if (out.items.len > 0) (Value{ .object = out.items[out.items.len - 1] }).strField("role") orelse "" else "";
            if (out.items.len == 0 or !std.mem.eql(u8, last_role, "assistant")) {
                const m = try json.newObject(a);
                try m.put(a, "role", .{ .string = "assistant" });
                try m.put(a, "content", .{ .string = "" });
                try out.append(a, m);
            }
            const last = out.items[out.items.len - 1];
            reasoning_open = false;
            const existing = last.get("tool_calls");
            const before: []const Value = if (existing != null and existing.? == .array) existing.?.array else &.{};
            try last.put(a, "tool_calls", .{ .array = try std.mem.concat(a, Value, &.{ before, &.{.{ .object = call }} }) });
        } else if (std.mem.eql(u8, k, "function_call_output") or std.mem.eql(u8, k, "custom_tool_call_output")) {
            const call_id = item.strField("call_id") orelse return cx.refuse("a function_call_output item needs the call_id of its call");
            const m = try json.newObject(a);
            try m.put(a, "role", .{ .string = "tool" });
            try m.put(a, "tool_call_id", .{ .string = call_id });
            try m.put(a, "content", try output(cx, if (std.mem.eql(u8, k, "custom_tool_call_output")) item.field("output") orelse .{ .string = "" } else item.get("output")));
            try out.append(a, m);
            reasoning_open = false;
        } else if (std.mem.eql(u8, k, "reasoning")) {
            var texts: std.ArrayList([]const u8) = .empty;
            if (item.get("content")) |c| if (c == .array) for (c.array) |p| {
                if (p.strField("text")) |t| try texts.append(a, t);
            };
            const joined = try std.mem.join(a, "\n", texts.items);
            if (joined.len > 0) {
                try flushThought(a, &out, joined);
                reasoning_open = true;
            }
        } else return cx.fail(.request, "input items of type {s} are not supported: send messages, function or custom tool calls and outputs, and reasoning items", .{try pyrepr.repr(a, kind)});
    }
    const result = try a.alloc(Value, out.items.len);
    for (out.items, result) |m, *slot| slot.* = .{ .object = m };
    return result;
}

fn emptyContent(value: Value) bool {
    if (value == .string) return value.string.len == 0;
    if (value != .array) return false;
    for (value.array) |part| {
        const t = part.strField("text") orelse return false;
        if (t.len != 0) return false;
    }
    return true;
}

fn flushThought(a: Allocator, out: *std.ArrayList(*json.Object), thought: []const u8) Allocator.Error!void {
    const last = if (out.items.len > 0) out.items[out.items.len - 1] else null;
    const reuse = if (last) |m| blk: {
        const v: Value = .{ .object = m };
        break :blk std.mem.eql(u8, v.strField("role") orelse "", "assistant") and emptyContent(v.get("content") orelse .null);
    } else false;
    const m = if (reuse) last.? else try json.newObject(a);
    if (!reuse) {
        try m.put(a, "role", .{ .string = "assistant" });
        try m.put(a, "content", .{ .string = "" });
        try out.append(a, m);
    }
    const before = (Value{ .object = m }).strField("reasoning_content");
    try m.put(a, "reasoning_content", .{ .string = if (before) |b| try std.mem.join(a, "\n", &.{ b, thought }) else thought });
}

const Choice = struct { choice: ?Value, offered: ?[]Value };

fn toolChoice(cx: *Cx, raw: ?Value, offered: ?[]Value, set: *response_tools.Set) errors.Refused!Choice {
    const choice = raw orelse return .{ .choice = null, .offered = offered };
    if (choice == .null) return .{ .choice = null, .offered = offered };
    if (choice == .string) for ([_][]const u8{ "none", "auto", "required" }) |s| if (std.mem.eql(u8, s, choice.string)) return .{ .choice = choice, .offered = offered };
    const kind = kindOf(choice);
    const k = if (kind != null and kind.? == .string) kind.?.string else "";
    if ((std.mem.eql(u8, k, "function") or std.mem.eql(u8, k, "custom")) and choice.strField("name") != null) {
        const function = try json.newObject(cx.a);
        try function.put(cx.a, "name", .{ .string = try set.name(cx, choice.strField("namespace"), choice.strField("name").?) });
        const c = try json.newObject(cx.a);
        try c.put(cx.a, "type", .{ .string = "function" });
        try c.put(cx.a, "function", .{ .object = function });
        return .{ .choice = .{ .object = c }, .offered = offered };
    }
    const mode: Value = choice.get("mode") orelse .{ .string = "auto" };
    const mode_ok = mode == .string and (std.mem.eql(u8, mode.string, "auto") or std.mem.eql(u8, mode.string, "required"));
    if (std.mem.eql(u8, k, "allowed_tools") and mode_ok) {
        const allowed = choice.get("tools");
        const list: []const Value = if (allowed != null and allowed.? == .array) allowed.?.array else &.{};
        if (allowed != null and allowed.?.truthy() and allowed.? != .array) return cx.refuse("allowed_tools may name function tools only");
        for (list) |t| {
            const tk = kindOf(t);
            if (tk == null or tk.? != .string or (!std.mem.eql(u8, tk.?.string, "function") and !std.mem.eql(u8, tk.?.string, "custom"))) return cx.refuse("allowed_tools may name function tools only");
        }
        var kept: std.ArrayList(Value) = .empty;
        for (offered orelse &.{}) |t| {
            const name = t.get("function").?.get("name").?;
            for (list) |allow| if (allow.get("name")) |n| if (n == .string and std.mem.eql(u8, try set.name(cx, allow.strField("namespace"), n.string), name.string)) {
                try kept.append(cx.a, t);
                break;
            };
        }
        return .{ .choice = mode, .offered = kept.items };
    }
    return cx.refuse("tool_choice must be none, auto, required, a function or allowed_tools of functions");
}

/// ``text.format`` as a chat ``response_format`` (null for plain text).
fn format(cx: *Cx, text: ?Value) errors.Refused!?Value {
    const t = text orelse return null;
    if (t == .null) return null;
    if (t != .object) return cx.refuse("text must be an object");
    const fmt = t.get("format") orelse return null;
    if (fmt == .null) return null;
    if (fmt != .object) return cx.other("'str' object has no attribute 'get'");
    const kind = fmt.get("type");
    const k = if (kind != null and kind.? == .string) kind.?.string else "";
    if (std.mem.eql(u8, k, "text")) return null;
    const out = try json.newObject(cx.a);
    if (std.mem.eql(u8, k, "json_object")) {
        try out.put(cx.a, "type", .{ .string = "json_object" });
        return .{ .object = out };
    }
    if (std.mem.eql(u8, k, "json_schema")) {
        const js = try json.newObject(cx.a);
        for ([_][]const u8{ "name", "schema", "strict", "description" }) |key| if (fmt.field(key)) |v| try js.put(cx.a, key, v);
        try out.put(cx.a, "type", .{ .string = "json_schema" });
        try out.put(cx.a, "json_schema", .{ .object = js });
        return .{ .object = out };
    }
    return cx.refuse("text.format must be text, json_object or json_schema");
}

/// A Responses request as the chat completion that runs it; ``history`` reads a stored conversation.
pub fn translate(cx: *Cx, body: Value, history: anytype) errors.Refused!Request {
    return translateWith(cx, body, history, false);
}
pub fn translateWith(cx: *Cx, original: Value, history: anytype, supported: bool) errors.Refused!Request {
    var body = original;
    if (supported and body == .object) {
        var scored = body.field("top_logprobs") != null;
        if (body.get("include")) |inc| if (inc == .array) for (inc.array) |v| {
            if (v == .string and std.mem.eql(u8, v.string, "message.output_text.logprobs")) scored = true;
        };
        if (scored) {
            const copy = try json.copyObject(cx.a, body.object);
            try copy.put(cx.a, "logprobs", .{ .bool = true });
            body = .{ .object = copy };
        }
    }
    const a = cx.a;
    if (body != .object) return cx.refuse("the request body must be a JSON object");
    try fields.probabilityOptionsWith(cx, body, supported);
    for (refused) |r| if (json.truthyField(body, r[0])) return cx.refuse(r[1]);
    if (body.get("include")) |inc| if (inc.truthy()) {
        var parts: std.ArrayList([]const u8) = .empty;
        switch (inc) {
            // reasoning.encrypted_content asks for nothing left out: reasoning comes as text in its reasoning item, and
            // nothing is encrypted. Clients built on the Responses API (Pi and others) send it with every reasoning request.
            .array => |items| for (items) |i| {
                if (i == .string and std.mem.eql(u8, i.string, "reasoning.encrypted_content")) continue;
                if (supported and i == .string and std.mem.eql(u8, i.string, "message.output_text.logprobs")) continue;
                try parts.append(a, try tool_specs.pyStr(a, i));
            },
            .object => |o| for (o.keys()) |k| try parts.append(a, k),
            .string => |s| for (0..s.len) |i| try parts.append(a, s[i .. i + 1]),
            else => try parts.append(a, try tool_specs.pyStr(a, inc)),
        }
        if (parts.items.len > 0) return cx.fail(.request, "include is not supported ({s}): reasoning comes as text in its reasoning item, and nothing is encrypted", .{try std.mem.join(a, ", ", parts.items)});
    };
    if (body.field("truncation")) |t| if (!(t == .string and std.mem.eql(u8, t.string, "disabled"))) return cx.refuse("truncation must be \"disabled\": a prompt too long for the context is refused");
    if (!supported and json.truthyField(body, "top_logprobs")) return cx.refuse("top_logprobs is not supported");
    const meta_raw = body.get("metadata");
    const metadata: Value = if (meta_raw != null and meta_raw.?.truthy()) meta_raw.? else .{ .object = try json.newObject(a) };
    const meta_ok = metadata == .object and metadata.object.count() <= 16 and for (metadata.object.keys(), metadata.object.values()) |k, v| {
        if (@import("reply_text.zig").charCount(k) > 64 or v != .string or @import("reply_text.zig").charCount(v.string) > 512) break false;
    } else true;
    if (!meta_ok) return cx.refuse("metadata must be at most 16 string pairs (keys up to 64 characters, values up to 512)");
    const instructions = body.field("instructions");
    if (instructions != null and instructions.? != .string) return cx.refuse("instructions must be a string");
    var given = body.get("input") orelse Value.null;
    if (given == .string) {
        const m = try json.newObject(a);
        try m.put(a, "role", .{ .string = "user" });
        try m.put(a, "content", given);
        const one = try a.alloc(Value, 1);
        one[0] = .{ .object = m };
        given = .{ .array = one };
    }
    if (given != .array or given.array.len == 0) return cx.refuse("input must be a string or a non-empty list of items");
    const added = try messages(cx, given.array);
    const parent = body.field("previous_response_id");
    const earlier: []Value = if (parent) |p| try history.conversation(cx, p) else &.{};
    var set = try response_tools.Set.initHistory(cx, body.get("tools"), try std.mem.concat(a, Value, &.{ earlier, added }));
    const choice = try toolChoice(cx, body.get("tool_choice"), set.offered, &set);
    const chat = try json.newObject(a);
    for (passed) |k| if (body.get(k)) |v| try chat.put(a, k, v);
    if (json.truthyField(body, "logprobs")) {
        try chat.put(a, "logprobs", .{ .bool = true });
        if (body.get("top_logprobs")) |v| try chat.put(a, "top_logprobs", v);
    }
    var msgs: std.ArrayList(Value) = .empty;
    if (instructions != null and instructions.?.string.len > 0) {
        const sys = try json.newObject(a);
        try sys.put(a, "role", .{ .string = "system" });
        try sys.put(a, "content", instructions.?);
        try msgs.append(a, .{ .object = sys });
    }
    try msgs.appendSlice(a, earlier);
    try msgs.appendSlice(a, added);
    try chat.put(a, "messages", .{ .array = try set.mapMessages(cx, msgs.items) });
    if (choice.offered) |o| try chat.put(a, "tools", .{ .array = o });
    if (choice.choice) |c| try chat.put(a, "tool_choice", c);
    if (body.field("max_output_tokens")) |m| try chat.put(a, "max_tokens", m);
    const reasoning_raw = body.get("reasoning");
    const reasoning: Value = if (reasoning_raw != null and reasoning_raw.?.truthy()) reasoning_raw.? else .{ .object = try json.newObject(a) };
    if (reasoning != .object) return cx.refuse("reasoning must be an object");
    if (reasoning.field("effort")) |e| try chat.put(a, "reasoning_effort", e);
    if (try format(cx, body.get("text"))) |f| try chat.put(a, "response_format", f);
    const echo = try json.newObject(a);
    try echo.put(a, "instructions", instructions orelse .null);
    try echo.put(a, "max_output_tokens", body.get("max_output_tokens") orelse .null);
    try echo.put(a, "metadata", metadata);
    try echo.put(a, "parallel_tool_calls", body.get("parallel_tool_calls") orelse .{ .bool = true });
    try echo.put(a, "previous_response_id", parent orelse .null);
    const r = try json.newObject(a);
    try r.put(a, "effort", reasoning.get("effort") orelse .null);
    try r.put(a, "summary", reasoning.get("summary") orelse .null);
    try echo.put(a, "reasoning", .{ .object = r });
    const store_field = body.get("store");
    const store = !(store_field != null and store_field.? == .bool and !store_field.?.bool);
    try echo.put(a, "store", .{ .bool = store });
    try echo.put(a, "temperature", body.get("temperature") orelse .null);
    const text_obj = try json.newObject(a);
    const text_raw = body.get("text");
    const fmt_raw: ?Value = if (text_raw != null and text_raw.?.truthy()) text_raw.?.get("format") else null;
    if (fmt_raw != null and fmt_raw.?.truthy()) try text_obj.put(a, "format", fmt_raw.?) else {
        const plain = try json.newObject(a);
        try plain.put(a, "type", .{ .string = "text" });
        try text_obj.put(a, "format", .{ .object = plain });
    }
    try echo.put(a, "text", .{ .object = text_obj });
    const tc = body.get("tool_choice");
    try echo.put(a, "tool_choice", if (tc != null and tc.?.truthy()) tc.? else .{ .string = "auto" });
    const ts = body.get("tools");
    try echo.put(a, "tools", if (ts != null and ts.?.truthy()) ts.? else .{ .array = &.{} });
    try echo.put(a, "top_p", body.get("top_p") orelse .null);
    try echo.put(a, "truncation", .{ .string = "disabled" });
    try echo.put(a, "user", body.get("user") orelse .null);
    try echo.put(a, "background", .{ .bool = false });
    return .{ .chat = .{ .object = chat }, .echo = echo, .added = added, .store = store, .stream = json.truthyField(body, "stream"), .tools = set };
}

test "reasoning joins parts and items, keeps empty and encrypted items, and survives at end" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const input = (try json.parse(a,
        \\[{"type":"reasoning","content":[{"text":"alpha"},{"text":"beta"}]},{"type":"reasoning","content":[]},{"type":"reasoning","encrypted_content":"opaque"},{"type":"reasoning","content":[{"text":"gamma"}]}]
    )).ok;
    const terminal = try messages(&cx, input.array);
    try std.testing.expectEqual(@as(usize, 1), terminal.len);
    try std.testing.expectEqualStrings("alpha\nbeta\ngamma", terminal[0].get("reasoning_content").?.string);
    try std.testing.expectEqualStrings("", terminal[0].get("content").?.string);
    const answer = (try json.parse(a, "[{\"role\":\"assistant\",\"content\":\"answer\"}]")).ok;
    const answered = try messages(&cx, try std.mem.concat(a, Value, &.{ input.array, answer.array }));
    try std.testing.expectEqual(@as(usize, 1), answered.len);
    try std.testing.expectEqualStrings("alpha\nbeta\ngamma", answered[0].get("reasoning_content").?.string);
    const user = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"next\"}]")).ok;
    const next = try messages(&cx, try std.mem.concat(a, Value, &.{ input.array, user.array }));
    try std.testing.expectEqual(@as(usize, 2), next.len);
    try std.testing.expectEqualStrings("alpha\nbeta\ngamma", next[0].get("reasoning_content").?.string);
    const encrypted = (try json.parse(a, "[{\"type\":\"reasoning\",\"encrypted_content\":\"opaque\"}]")).ok;
    try std.testing.expectEqual(@as(usize, 0), (try messages(&cx, encrypted.array)).len);
}

test "custom tool history encodes input and accepts omitted output" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const input = (try json.parse(a,
        \\[{"type":"custom_tool_call","call_id":"c1","name":"apply_patch","input":"*** Begin Patch\n+é\n*** End Patch"},{"type":"custom_tool_call_output","call_id":"c1"}]
    )).ok;
    const out = try messages(&cx, input.array);
    const args = out[0].get("tool_calls").?.array[0].get("function").?.get("arguments").?.string;
    try std.testing.expectEqualStrings(input.array[0].get("input").?.string, (try json.parse(a, args)).ok.get("input").?.string);
    try std.testing.expectEqualStrings("", out[1].get("content").?.string);
    try std.testing.expectEqualStrings("c1", out[1].get("tool_call_id").?.string);
}

test "Responses choices select mapped namespaces and custom tools while regex stays Chat only" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const decl = (try json.parse(a,
        \\[{"type":"custom","name":"apply_patch"},{"type":"namespace","name":"files","tools":[{"type":"function","name":"read"}]}]
    )).ok;
    var set = try response_tools.Set.init(&cx, decl);
    for ([_][2][]const u8{
        .{ "{\"type\":\"custom\",\"name\":\"apply_patch\"}", "apply_patch" },
        .{ "{\"type\":\"function\",\"namespace\":\"files\",\"name\":\"read\"}", "files::read" },
    }) |c| {
        const choice = try toolChoice(&cx, (try json.parse(a, c[0])).ok, set.offered, &set);
        try std.testing.expectEqualStrings(c[1], choice.choice.?.get("function").?.get("name").?.string);
    }
    const allowed = (try json.parse(a, "{\"type\":\"allowed_tools\",\"tools\":[{\"type\":\"function\",\"namespace\":\"files\",\"name\":\"read\"}]}")).ok;
    const filtered = try toolChoice(&cx, allowed, set.offered, &set);
    try std.testing.expectEqual(@as(usize, 1), filtered.offered.?.len);
    try std.testing.expectEqualStrings("files::read", filtered.offered.?[0].get("function").?.get("name").?.string);
    const regex = (try json.parse(a, "{\"format\":{\"type\":\"regex\",\"regex\":\"x\"}}")).ok;
    try std.testing.expectError(error.Refused, format(&cx, regex));
}

test "ordinary Responses requests keep the same chat prompt inputs" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const NoHistory = struct {
        pub fn conversation(_: @This(), _: *Cx, _: Value) errors.Refused![]Value {
            return &.{};
        }
    };
    const raw =
        \\{"model":"test","input":[{"role":"user","content":"hello"},{"type":"function_call","call_id":"c1","name":"read","arguments":"{}"},{"type":"function_call_output","call_id":"c1","output":"ok"}],"tools":[{"type":"function","name":"read","description":"read","parameters":{"type":"object"},"strict":false}],"tool_choice":"auto"}
    ;
    const request = try translate(&cx, (try json.parse(a, raw)).ok, NoHistory{});
    const expected = (try json.parse(a,
        \\{"model":"test","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"ok"}],"tools":[{"type":"function","function":{"name":"read","description":"read","parameters":{"type":"object"},"strict":false}}],"tool_choice":"auto"}
    )).ok;
    try std.testing.expectEqualStrings(try json.stringify(a, expected, .{}), try json.stringify(a, request.chat, .{}));
}

test "reasoning following answered and empty assistant messages preserves order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const raw = (try json.parse(a,
        \\[{"role":"assistant","content":"old answer"},{"type":"reasoning","content":[{"text":"new thought"}]},{"type":"reasoning","content":[]},{"role":"assistant","content":"new answer"},{"type":"reasoning","content":[{"text":"terminal"}]}]
    )).ok;
    const out = try messages(&cx, raw.array);
    try std.testing.expectEqual(@as(usize, 3), out.len);
    try std.testing.expectEqualStrings("old answer", out[0].get("content").?.string);
    try std.testing.expectEqualStrings("new thought", out[1].get("reasoning_content").?.string);
    try std.testing.expectEqualStrings("new answer", out[1].get("content").?.string);
    try std.testing.expectEqualStrings("terminal", out[2].get("reasoning_content").?.string);
}

test "reasoning reuses an assistant with empty content parts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const raw = (try json.parse(a,
        \\[{"role":"assistant","content":[]},{"type":"reasoning","content":[{"text":"thought"}]},{"type":"reasoning","content":[{"text":"more"}]}]
    )).ok;
    const out = try messages(&cx, raw.array);
    try std.testing.expectEqual(@as(usize, 1), out.len);
    try std.testing.expectEqualStrings("thought\nmore", out[0].get("reasoning_content").?.string);
}

test "input_file parts become file parts from inline data; a remote file_url or a bare file_id is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const items = (try json.parse(cx.a,
        \\[{"role": "user", "content": [{"type": "input_text", "text": "read"}, {"type": "input_file", "filename": "a.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}]},
        \\ {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
        \\ {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_file", "file_data": "JVBERi0="}]}]
    )).ok.array;
    const out = try messages(&cx, items);
    const part = out[0].get("content").?.array[1];
    try std.testing.expectEqualStrings("file", part.get("type").?.string);
    try std.testing.expectEqualStrings("a.pdf", part.get("file").?.get("filename").?.string);
    try std.testing.expectEqualStrings("JVBERi0=", out[2].get("content").?.array[0].get("file").?.get("file_data").?.string);
    for ([_][]const u8{
        \\[{"role": "user", "content": [{"type": "input_file", "file_url": "https://example.com/doc.pdf"}]}]
        ,
        \\[{"role": "user", "content": [{"type": "input_file", "file_id": "file-1"}]}]
    }, [_][]const u8{ "URLs are not accepted", "file_id is not supported" }) |raw, want| {
        var bad: Cx = .{ .a = cx.a };
        try std.testing.expectError(error.Refused, messages(&bad, (try json.parse(cx.a, raw)).ok.array));
        try std.testing.expect(std.mem.indexOf(u8, bad.message, want) != null);
    }
}

test "recipe 41 current namespace avoids historical bare names in both orders" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const NoHistory = struct {
        pub fn conversation(_: @This(), _: *Cx, _: Value) errors.Refused![]Value {
            return &.{};
        }
    };
    for ([_]bool{ false, true }) |declared| {
        for ([_]bool{ false, true }) |reversed| {
            const namespaced = "{\"type\":\"function_call\",\"call_id\":\"namespaced\",\"namespace\":\"ns\",\"name\":\"run\",\"arguments\":\"{}\"}";
            const bare = "{\"type\":\"function_call\",\"call_id\":\"bare\",\"name\":\"ns::run\",\"arguments\":\"{}\"}";
            const raw = try std.fmt.allocPrint(a, "{{\"model\":\"test\",\"tools\":{s},\"input\":[{s},{s},{{\"type\":\"function_call_output\",\"call_id\":\"namespaced\",\"output\":\"one\"}},{{\"type\":\"function_call_output\",\"call_id\":\"bare\",\"output\":\"two\"}}]}}", .{
                if (declared) "[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"function\",\"name\":\"run\"}]}]" else "[]",
                if (reversed) bare else namespaced,
                if (reversed) namespaced else bare,
            });
            const request = try translate(&cx, (try json.parse(a, raw)).ok, NoHistory{});
            const calls = request.chat.get("messages").?.array[0].get("tool_calls").?.array;
            const ns_name = calls[@intFromBool(reversed)].get("function").?.get("name").?.string;
            const bare_name = calls[@intFromBool(!reversed)].get("function").?.get("name").?.string;
            try std.testing.expectEqualStrings("ns::run", bare_name);
            try std.testing.expect(!std.mem.eql(u8, ns_name, bare_name));
            if (declared) try std.testing.expectEqualStrings(ns_name, request.chat.get("tools").?.array[0].get("function").?.get("name").?.string);
        }
    }
}
