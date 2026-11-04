//! Offered tools and tool_choice as the template should see them (``tools.active_tool_specs``).
const std = @import("std");
const strict_schema = @import("strict_schema.zig");
const json = @import("json.zig");
const errors = @import("errors.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Python's ``str()`` of a decoded JSON scalar (containers approximate with JSON).
pub fn pyStr(a: Allocator, v: Value) Allocator.Error![]const u8 {
    return switch (v) {
        .string => |s| s,
        .int => |t| t,
        .bool => |b| if (b) "True" else "False",
        .null => "None",
        .float => |f| blk: {
            var buf: [40]u8 = undefined;
            break :blk try a.dupe(u8, json.floatRepr(&buf, f));
        },
        else => try json.stringify(a, v, .{}),
    };
}

/// ``str(value or "").strip()``.
fn nameText(a: Allocator, v: ?Value) Allocator.Error![]const u8 {
    const value = v orelse return "";
    if (!value.truthy()) return "";
    return std.mem.trim(u8, try pyStr(a, value), " \t\r\n\x0b\x0c");
}

pub fn toolName(a: Allocator, tool: Value) Allocator.Error![]const u8 {
    if (tool != .object) return "";
    if (tool.get("function")) |function| if (function == .object) return nameText(a, function.get("name"));
    return nameText(a, tool.get("name"));
}

fn lowerEq(text: []const u8, want: []const u8) bool {
    return std.ascii.eqlIgnoreCase(std.mem.trim(u8, text, " \t\r\n\x0b\x0c"), want);
}

pub fn choiceDisables(choice: ?Value) bool {
    const c = choice orelse return false;
    if (c == .string) return lowerEq(c.string, "none");
    if (c == .object) {
        const value = c.firstTruthy(&.{ "type", "mode" }) orelse return false;
        return value == .string and lowerEq(value.string, "none");
    }
    return false;
}

/// OpenAI's "required" or a named function: the reply must call a tool.
pub fn choiceRequiresCall(a: Allocator, choice: ?Value) Allocator.Error!bool {
    const c = choice orelse return false;
    if (c == .string) return lowerEq(c.string, "required");
    if (c == .object) {
        const value = if (c.firstTruthy(&.{ "type", "mode" })) |v| try pyStr(a, v) else "";
        if (lowerEq(value, "required")) return true;
        const function = c.get("function");
        return lowerEq(value, "function") and function != null and function.? == .object;
    }
    return false;
}

fn isFunctionChoice(a: Allocator, c: Value) Allocator.Error!bool {
    if (c != .object) return false;
    const t = c.get("type") orelse return false;
    if (!t.truthy()) return false;
    return std.ascii.eqlIgnoreCase(try pyStr(a, t), "function");
}

/// The tools the template offers: none for "none", only the named one for a named function.
pub fn active(cx: *Cx, tools: ?Value, choice: ?Value) errors.Refused![]Value {
    var specs: []Value = &.{};
    if (tools) |t| if (t != .null) {
        if (t != .array) return cx.other("tools must be a list");
        var names: std.StringHashMapUnmanaged(void) = .empty;
        for (t.array, 0..) |tool, i| {
            if (tool != .object) return cx.fail(.other, "tools[{d}] must be an object", .{i});
            const name = try toolName(cx.a, tool);
            if (name.len == 0) return cx.fail(.other, "tools[{d}] must include a function name", .{i});
            if ((try names.getOrPut(cx.a, name)).found_existing) return cx.refuse("Tool names must be unique.");
            const function = definition(tool);
            if (try strict_schema.enabled(cx, function)) _ = try strictParameters(cx, function);
        }
        specs = t.array;
    };
    if (choiceDisables(choice)) return &.{};
    if (specs.len == 0) {
        if (try choiceRequiresCall(cx.a, choice)) return cx.other("tool_choice requires a tool call, but the request offers no tools");
        return &.{};
    }
    const c = choice orelse return specs;
    if (!try isFunctionChoice(cx.a, c)) return specs;
    const function = c.get("function") orelse return cx.other("tool_choice function must include a function object");
    if (function != .object) return cx.other("tool_choice function must include a function object");
    const requested = try nameText(cx.a, function.get("name"));
    if (requested.len == 0) return cx.other("tool_choice function must include a name");
    var named: std.ArrayList(Value) = .empty;
    for (specs) |tool| if (std.mem.eql(u8, try toolName(cx.a, tool), requested)) try named.append(cx.a, tool);
    if (named.items.len == 0) return cx.fail(.other, "tool_choice requested unknown tool '{s}'", .{requested});
    return named.items;
}

pub fn definition(tool: Value) Value {
    const function = tool.get("function") orelse return tool;
    return if (function == .object) function else tool;
}

fn strictParameters(cx: *Cx, function: Value) errors.Refused!Value {
    const schema = function.field("parameters") orelse function.field("input_schema") orelse blk: {
        const empty = try json.newObject(cx.a);
        try empty.put(cx.a, "type", .{ .string = "object" });
        break :blk Value{ .object = empty };
    };
    return strict_schema.normalize(cx, schema, true);
}

/// The decoder gets a normalized copy; the template keeps the request as written.
pub fn constrained(cx: *Cx, tools: []const Value, has_format: bool) errors.Refused!?[]const Value {
    var strict: usize = 0;
    for (tools) |tool| if (try strict_schema.enabled(cx, definition(tool))) {
        strict += 1;
    };
    if (strict == 0) return null;
    if (strict != tools.len) return cx.refuse("Cannot use strict tool with non-strict tool in the same request");
    if (has_format) return cx.refuse("response_format should not be used with tools");
    const out = try cx.a.alloc(Value, tools.len);
    for (tools, out) |tool, *slot| {
        const function = try json.copyObject(cx.a, definition(tool).object);
        try function.put(cx.a, "parameters", try strictParameters(cx, .{ .object = function }));
        if (tool.has("function")) {
            const copy = try json.copyObject(cx.a, tool.object);
            try copy.put(cx.a, "function", .{ .object = function });
            slot.* = .{ .object = copy };
        } else slot.* = .{ .object = function };
    }
    return out;
}

test "duplicate offered names and strict constraints are request errors" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const duplicate = (try json.parse(cx.a, "[{\"function\":{\"name\":\"f\"}},{\"function\":{\"name\":\"f\"}}]")).ok;
    for ([_]?Value{ null, .{ .string = "none" } }) |choice| {
        try std.testing.expectError(error.Refused, active(&cx, duplicate, choice));
        try std.testing.expectEqualStrings("Tool names must be unique.", cx.message);
        try std.testing.expectEqual(errors.Kind.request, cx.kind);
    }
    const strict = (try json.parse(cx.a, "[{\"function\":{\"name\":\"f\",\"strict\":true}}]")).ok;
    const normalized = (try constrained(&cx, strict.array, false)).?;
    try std.testing.expect(!definition(normalized[0]).get("parameters").?.get("additionalProperties").?.bool);
    try std.testing.expect(!definition(strict.array[0]).has("parameters"));
    try std.testing.expectError(error.Refused, constrained(&cx, strict.array, true));
    const mixed = (try json.parse(cx.a, "[{\"function\":{\"name\":\"f\",\"strict\":true}},{\"function\":{\"name\":\"g\"}}]")).ok;
    try std.testing.expectError(error.Refused, constrained(&cx, mixed.array, false));
    const disabled = try active(&cx, mixed, .{ .string = "none" });
    try std.testing.expectEqual(@as(?[]const Value, null), try constrained(&cx, disabled, true));
}
