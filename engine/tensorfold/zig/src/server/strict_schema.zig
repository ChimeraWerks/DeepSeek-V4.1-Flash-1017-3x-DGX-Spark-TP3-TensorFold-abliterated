//! Strict schemas for decoding; copies leave the client's prompt schema unchanged.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const Value = json.Value;
const Cx = errors.Cx;

pub fn enabled(cx: *Cx, spec: Value) errors.Refused!bool {
    const strict = spec.field("strict") orelse return false;
    if (strict != .bool) return cx.refuse("strict must be a boolean");
    return strict.bool;
}

pub fn normalize(cx: *Cx, schema: Value, tool: bool) errors.Refused!Value {
    const out = try node(cx, schema, tool, 0);
    if (!std.mem.eql(u8, out.strField("type") orelse "", "object")) return cx.refuse(if (tool)
        "The top level of the function parameters schema must be an object."
    else
        "The top level of the json schema must be an object.");
    return out;
}

fn node(cx: *Cx, schema: Value, allow_empty: bool, depth: usize) errors.Refused!Value {
    if (depth > 128) return cx.refuse("JSON schema nesting is too deep");
    if (schema != .object) return cx.refuse("Invalid json schema: expected a schema object");
    const out = try json.copyObject(cx.a, schema.object);
    const kind = schema.get("type");
    if (kind) |t| {
        if (t == .array) {
            if (t.array.len == 0) return cx.refuse("Invalid json schema: type must not be empty");
            for (t.array) |v| if (v != .string or !simple(v.string)) return cx.refuse("Invalid json schema: type arrays may contain only primitive types");
        } else if (t == .string and std.mem.eql(u8, t.string, "object")) {
            const props = schema.field("properties") orelse Value{ .object = try json.newObject(cx.a) };
            if (props != .object) return cx.refuse("Invalid json schema: properties must be an object");
            if (!allow_empty and props.object.count() == 0) return cx.refuse("An object with no properties is not allowed.");
            if (schema.field("required")) |required| {
                if (required != .array) return cx.refuse("Invalid json schema: required must be an array");
                var seen: std.StringHashMapUnmanaged(void) = .empty;
                for (required.array) |key| {
                    if (key != .string) return cx.refuse("Invalid json schema: required entries must be strings");
                    if (!props.object.contains(key.string)) return cx.fail(.request, "Required property '{s}' not found in object", .{key.string});
                    if ((try seen.getOrPut(cx.a, key.string)).found_existing) return cx.fail(.request, "Duplicate required property '{s}'", .{key.string});
                }
                if (seen.count() != props.object.count()) return cx.refuse("Required properties must match all properties in the object");
            }
            if (schema.field("additionalProperties")) |additional| {
                if (additional != .bool) return cx.refuse("Invalid json schema: additionalProperties must be a boolean");
                if (additional.bool) return cx.refuse("additionalProperties should not be true");
            }
            const normalized = try json.newObject(cx.a);
            const required = try cx.a.alloc(Value, props.object.count());
            for (props.object.keys(), props.object.values(), required) |key, value, *slot| {
                try normalized.put(cx.a, key, try node(cx, value, false, depth + 1));
                slot.* = .{ .string = key };
            }
            try out.put(cx.a, "properties", .{ .object = normalized });
            try out.put(cx.a, "required", .{ .array = required });
            try out.put(cx.a, "additionalProperties", .{ .bool = false });
        } else if (t == .string and std.mem.eql(u8, t.string, "array")) {
            const items = schema.get("items") orelse return cx.refuse("Invalid json schema: array requires items");
            try out.put(cx.a, "items", try node(cx, items, false, depth + 1));
        } else if (t != .string or !simple(t.string)) return cx.refuse("Invalid json schema: unknown type");
    } else if (!schema.has("anyOf") and !schema.has("$ref")) return cx.refuse("Invalid json schema: one of type, anyOf, $ref is required");
    if (schema.get("$ref")) |ref| {
        if (ref != .string or !std.mem.startsWith(u8, ref.string, "#")) return cx.refuse("Invalid json schema: only local references are supported");
    }
    // Retain and normalize standard definitions too, rather than dropping references' targets.
    for ([_][]const u8{ "$def", "$defs", "definitions" }) |key| if (schema.get(key)) |defs| {
        if (defs != .object) return cx.refuse("Invalid json schema: definitions must be an object");
        const normalized = try json.newObject(cx.a);
        for (defs.object.keys(), defs.object.values()) |name, value| try normalized.put(cx.a, name, try node(cx, value, false, depth + 1));
        try out.put(cx.a, key, .{ .object = normalized });
    };
    for ([_][]const u8{ "anyOf", "oneOf", "allOf" }) |key| if (schema.get(key)) |branches| {
        if (branches != .array or branches.array.len == 0) return cx.refuse("Invalid json schema: alternatives must be a non-empty array");
        const normalized = try cx.a.alloc(Value, branches.array.len);
        for (branches.array, normalized) |branch, *slot| slot.* = try node(cx, branch, false, depth + 1);
        try out.put(cx.a, key, .{ .array = normalized });
    };
    return .{ .object = out };
}

fn simple(kind: []const u8) bool {
    for ([_][]const u8{ "string", "number", "integer", "boolean", "null" }) |name| if (std.mem.eql(u8, kind, name)) return true;
    return false;
}

test "strict schemas normalize recursively without changing the prompt schema" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const schema = (try json.parse(cx.a, "{\"type\":\"object\",\"properties\":{\"xs\":{\"type\":\"array\",\"items\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}}}}}}")).ok;
    const before = try json.stringify(cx.a, schema, .{});
    const normalized = try normalize(&cx, schema, true);
    try std.testing.expectEqualStrings(before, try json.stringify(cx.a, schema, .{}));
    try std.testing.expectEqualStrings("xs", normalized.get("required").?.array[0].string);
    const nested = normalized.get("properties").?.get("xs").?.get("items").?;
    try std.testing.expect(!nested.get("additionalProperties").?.bool);
    try std.testing.expectEqualStrings("x", nested.get("required").?.array[0].string);
}

test "strict schemas reject conflicts and non-object roots" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    for ([_][]const u8{
        "{\"type\":\"string\"}",
        "{\"type\":\"object\",\"additionalProperties\":true}",
        "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}},\"required\":[]}",
        "{\"type\":\"object\",\"required\":[\"unknown\"]}",
        "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}},\"required\":[\"x\",\"x\"]}",
        "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"object\"}}}",
    }) |raw| try std.testing.expectError(error.Refused, normalize(&cx, (try json.parse(cx.a, raw)).ok, true));
    const empty = (try json.parse(cx.a, "{\"type\":\"object\"}")).ok;
    _ = try normalize(&cx, empty, true);
    try std.testing.expectError(error.Refused, normalize(&cx, empty, false));
}

test "recipe 31 standard definitions keep normalized targets" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    for ([_][]const u8{ "$defs", "definitions" }) |key| {
        const raw = try std.fmt.allocPrint(cx.a, "{{\"type\":\"object\",\"properties\":{{\"value\":{{\"$ref\":\"#/{s}/Entry\"}}}},\"{s}\":{{\"Entry\":{{\"type\":\"object\",\"properties\":{{\"name\":{{\"type\":\"string\"}}}}}}}}}}", .{ key, key });
        const normalized = try normalize(&cx, (try json.parse(cx.a, raw)).ok, true);
        const entry = normalized.get(key).?.get("Entry").?;
        try std.testing.expectEqualStrings("name", entry.get("required").?.array[0].string);
        try std.testing.expect(!entry.get("additionalProperties").?.bool);
    }
}

test "recipe 33 anyOf keeps references and nested unions" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const raw =
        \\{"type":"object","properties":{"value":{"anyOf":[{"$ref":"#/properties/base"},{"anyOf":[{"type":"string"},{"type":"null"}]}]},"base":{"type":"integer"}}}
    ;
    const normalized = try normalize(&cx, (try json.parse(cx.a, raw)).ok, true);
    const branches = normalized.get("properties").?.get("value").?.get("anyOf").?.array;
    try std.testing.expectEqualStrings("#/properties/base", branches[0].get("$ref").?.string);
    try std.testing.expectEqualStrings("null", branches[1].get("anyOf").?.array[1].get("type").?.string);
}

test "recipe 34 35 36 39 51 strict primitive and array constraints survive" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const cases = [_][]const u8{
        \\{"type":"integer","minimum":0.5,"maximum":10.5,"multipleOf":0.5}
        ,
        \\{"type":"integer","maximum":18446744073709551615}
        ,
        \\{"type":["string","null"],"enum":["on",null]}
        ,
        \\{"type":"string","const":"fixed"}
        ,
        \\{"type":"boolean","const":true}
        ,
        \\{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":3,"uniqueItems":true}
        ,
        \\{"type":"array","items":{"type":"array","items":{"type":"integer"},"minItems":0,"maxItems":0,"uniqueItems":false},"minItems":0}
        ,
    };
    for (cases) |raw| {
        const property = (try json.parse(cx.a, raw)).ok;
        const input = (try json.parse(cx.a, try std.fmt.allocPrint(cx.a, "{{\"type\":\"object\",\"properties\":{{\"value\":{s}}}}}", .{raw}))).ok;
        const before = try json.stringify(cx.a, input, .{});
        const normalized = try normalize(&cx, input, true);
        try std.testing.expectEqualStrings(try json.stringify(cx.a, property, .{}), try json.stringify(cx.a, normalized.get("properties").?.get("value").?, .{}));
        try std.testing.expectEqualStrings(before, try json.stringify(cx.a, input, .{}));
    }
    for ([_][]const u8{ "date-time", "date", "time", "uri", "duration", "custom-format" }) |format| {
        const input = (try json.parse(cx.a, try std.fmt.allocPrint(cx.a, "{{\"type\":\"object\",\"properties\":{{\"value\":{{\"type\":\"string\",\"format\":\"{s}\"}}}}}}", .{format}))).ok;
        const normalized = try normalize(&cx, input, true);
        try std.testing.expectEqualStrings(format, normalized.get("properties").?.get("value").?.get("format").?.string);
    }
}
