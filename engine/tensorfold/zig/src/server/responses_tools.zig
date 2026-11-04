//! Responses tool names at the API boundary and the function names seen by the model.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const Value = json.Value;
const Cx = errors.Cx;

pub const Name = struct { flat: []const u8, namespace: []const u8, local: []const u8 };

pub const Set = struct {
    offered: ?[]Value = null,
    namespaces: std.ArrayList(Name) = .empty,
    reserved: std.ArrayList([]const u8) = .empty,
    custom_patch: bool = false,

    pub fn init(cx: *Cx, raw: ?Value) errors.Refused!Set {
        var set: Set = .{};
        const value = raw orelse return set;
        if (value == .null) return set;
        if (value != .array) return cx.refuse("tools must be a list");
        for (value.array) |tool| if (tool.typeIs("function") or tool.typeIs("custom")) {
            try set.reserved.append(cx.a, tool.strField("name") orelse "");
        };
        var out: std.ArrayList(Value) = .empty;
        for (value.array) |tool| {
            if (tool.typeIs("function")) {
                const local = tool.strField("name") orelse "";
                if (local.len == 0) return cx.refuse("a function tool needs a name");
                try out.append(cx.a, try function(cx, tool));
            } else if (tool.typeIs("custom")) {
                if (!std.mem.eql(u8, tool.strField("name") orelse "", "apply_patch")) return cx.refuse("only the custom apply_patch tool is supported");
                set.custom_patch = true;
                const patch = (try json.parse(cx.a, patch_schema)).ok;
                try patch.object.put(cx.a, "description", .{ .string = patch_description });
                try out.append(cx.a, try function(cx, patch));
            } else if (tool.typeIs("namespace")) {
                const ns = tool.strField("name") orelse "";
                try validName(cx, ns);
                const inner = tool.get("tools") orelse return cx.refuse("a namespace needs tools");
                if (inner != .array) return cx.refuse("namespace tools must be a list");
                var seen: std.ArrayList([]const u8) = .empty;
                for (inner.array) |t| {
                    if (!t.typeIs("function")) return cx.refuse("a namespace may contain function tools only");
                    const local = t.strField("name") orelse "";
                    try validName(cx, local);
                    for (seen.items) |n| if (std.mem.eql(u8, n, local)) return cx.refuse("tool names within a namespace must be unique");
                    try seen.append(cx.a, local);
                    const flat = try set.unique(cx, ns, local);
                    try set.namespaces.append(cx.a, .{ .flat = flat, .namespace = ns, .local = local });
                    const f = try function(cx, t);
                    const obj = f.get("function").?.object;
                    try obj.put(cx.a, "name", .{ .string = flat });
                    if (tool.strField("description")) |desc| if (desc.len > 0) {
                        try obj.put(cx.a, "description", .{ .string = try std.mem.join(cx.a, "\n", &.{ desc, t.strField("description") orelse "" }) });
                    };
                    try out.append(cx.a, f);
                }
            } else return cx.refuse("tools must be functions, namespaces or custom apply_patch");
        }
        set.offered = out.items;
        return set;
    }

    fn taken(set: *const Set, candidate: []const u8) bool {
        for (set.reserved.items) |n| if (std.ascii.eqlIgnoreCase(n, candidate)) return true;
        for (set.namespaces.items) |n| if (std.ascii.eqlIgnoreCase(n.flat, candidate)) return true;
        return false;
    }

    fn unique(set: *Set, cx: *Cx, ns: []const u8, local: []const u8) errors.Refused![]const u8 {
        const base = try std.fmt.allocPrint(cx.a, "{s}::{s}", .{ ns, local });
        if (!set.taken(base)) return base;
        var i: usize = 1;
        while (true) : (i += 1) {
            const candidate = try std.fmt.allocPrint(cx.a, "{s}_{d}", .{ base, i });
            if (!set.taken(candidate)) return candidate;
        }
    }

    pub fn name(set: *Set, cx: *Cx, ns: ?[]const u8, local: []const u8) errors.Refused![]const u8 {
        const namespace = ns orelse return local;
        for (set.namespaces.items) |n| if (std.mem.eql(u8, namespace, n.namespace) and std.mem.eql(u8, local, n.local)) return n.flat;
        const flat = try set.unique(cx, namespace, local);
        try set.namespaces.append(cx.a, .{ .flat = flat, .namespace = namespace, .local = local });
        return flat;
    }

    pub fn resolve(set: *const Set, flat: []const u8) ?Name {
        for (set.namespaces.items) |n| if (std.mem.eql(u8, flat, n.flat)) return n;
        return null;
    }

    /// Stored messages retain local names; copy only namespaced calls before rendering them.
    pub fn mapMessages(set: *Set, cx: *Cx, messages: []Value) errors.Refused![]Value {
        for (messages) |m| if (m.get("tool_calls")) |calls| if (calls == .array) for (calls.array) |call| {
            const f = call.get("function") orelse continue;
            if (f.strField("namespace") == null) if (f.strField("name")) |n| try set.reserved.append(cx.a, n);
        };
        const out = try cx.a.dupe(Value, messages);
        for (out) |*m| {
            const calls = m.get("tool_calls") orelse continue;
            if (calls != .array) continue;
            var copied: ?[]Value = null;
            for (calls.array, 0..) |call, i| {
                const f = call.get("function") orelse continue;
                const ns = f.strField("namespace") orelse continue;
                const local = f.strField("name") orelse continue;
                if (copied == null) copied = try cx.a.dupe(Value, calls.array);
                const fc = try json.copyObject(cx.a, f.object);
                _ = fc.orderedRemove("namespace");
                try fc.put(cx.a, "name", .{ .string = try set.name(cx, ns, local) });
                const cc = try json.copyObject(cx.a, call.object);
                try cc.put(cx.a, "function", .{ .object = fc });
                copied.?[i] = .{ .object = cc };
            }
            if (copied) |c| {
                const mc = try json.copyObject(cx.a, m.object);
                try mc.put(cx.a, "tool_calls", .{ .array = c });
                m.* = .{ .object = mc };
            }
        }
        return out;
    }
};

fn validName(cx: *Cx, name: []const u8) errors.Refused!void {
    if (name.len == 0 or name.len > 128) return cx.refuse("namespace and function names must contain 1 to 128 characters");
    for (name) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-') return cx.refuse("namespace and function names must contain letters, digits, underscores or hyphens");
}

fn function(cx: *Cx, tool: Value) errors.Refused!Value {
    const f = try json.newObject(cx.a);
    for ([_][]const u8{ "name", "description", "parameters", "strict" }) |k| if (tool.field(k)) |v| try f.put(cx.a, k, v);
    const t = try json.newObject(cx.a);
    try t.put(cx.a, "type", .{ .string = "function" });
    try t.put(cx.a, "function", .{ .object = f });
    return .{ .object = t };
}

const patch_schema =
    \\{"name":"apply_patch","parameters":{"type":"object","additionalProperties":false,"properties":{"input":{"type":"string","description":"The entire contents of the apply_patch command (the *** Begin Patch ... *** End Patch envelope)."}},"required":["input"]}}
;

const patch_description = "Use the `apply_patch` tool to edit files.\nYour patch language is a stripped‑down, file‑oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high‑level envelope:\n\n*** Begin Patch\n[ one or more file sections ]\n*** End Patch\n\nWithin that envelope, you get a sequence of file operations.\nYou MUST include a header to specify the action you are taking.\nEach operation starts with one of three headers:\n\n*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).\n*** Delete File: <path> - remove an existing file. Nothing follows.\n*** Update File: <path> - patch an existing file in place (optionally with a rename).\n\nMay be immediately followed by *** Move to: <new path> if you want to rename the file.\nThen one or more “hunks”, each introduced by @@ (optionally followed by a hunk header).\nWithin a hunk each line starts with:\n\nFor instructions on [context_before] and [context_after]:\n- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change’s [context_after] lines in the second change’s [context_before] lines.\n- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:\n@@ class BaseClass\n[3 lines of pre-context]\n- [old_code]\n+ [new_code]\n[3 lines of post-context]\n\n- If a code block is repeated so many times in a class or function such that even a single `@@` statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:\n\n@@ class BaseClass\n@@ \t def method():\n[3 lines of pre-context]\n- [old_code]\n+ [new_code]\n[3 lines of post-context]\n\nThe full grammar definition is below:\nPatch := Begin { FileOp } End\nBegin := \"*** Begin Patch\" NEWLINE\nEnd := \"*** End Patch\" NEWLINE\nFileOp := AddFile | DeleteFile | UpdateFile\nAddFile := \"*** Add File: \" path NEWLINE { \"+\" line NEWLINE }\nDeleteFile := \"*** Delete File: \" path NEWLINE\nUpdateFile := \"*** Update File: \" path NEWLINE [ MoveTo ] { Hunk }\nMoveTo := \"*** Move to: \" newPath NEWLINE\nHunk := \"@@\" [ header ] NEWLINE { HunkLine } [ \"*** End of File\" NEWLINE ]\nHunkLine := (\" \" | \"-\" | \"+\") text NEWLINE\n\nA full patch can combine several operations:\n\n*** Begin Patch\n*** Add File: hello.txt\n+Hello world\n*** Update File: src/app.py\n*** Move to: src/main.py\n@@ def greet():\n-print(\"Hi\")\n+print(\"Hello, world!\")\n*** Delete File: obsolete.txt\n*** End Patch\n\nIt is important to remember:\n\n- You must include a header with your intended action (Add/Delete/Update)\n- You must prefix new lines with `+` even when creating a new file\n- File references can only be relative, NEVER ABSOLUTE.\n";

test "namespaces are collision safe, described and replayable without changing stored names" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const decl = (try json.parse(a,
        \\[{"type":"namespace","name":"files","description":"Files API","tools":[{"type":"function","name":"read","description":"Read a file","parameters":{"type":"object"}}]},{"type":"function","name":"files::read"},{"type":"namespace","name":"files","tools":[{"type":"function","name":"read"}]}]
    )).ok;
    var set = try Set.init(&cx, decl);
    try std.testing.expectEqualStrings("files::read_1", set.namespaces.items[0].flat);
    try std.testing.expectEqualStrings("files::read_2", set.namespaces.items[1].flat);
    try std.testing.expectEqualStrings("Files API\nRead a file", set.offered.?[0].get("function").?.get("description").?.string);
    try std.testing.expectEqualStrings("read", set.resolve("files::read_1").?.local);
    const history = (try json.parse(a,
        \\[{"role":"assistant","content":"","tool_calls":[{"type":"function","id":"c1","function":{"namespace":"files","name":"read","arguments":"{}"}},{"type":"function","id":"c2","function":{"namespace":"old","name":"gone","arguments":"{}"}}]}]
    )).ok;
    const mapped = try set.mapMessages(&cx, history.array);
    const calls = mapped[0].get("tool_calls").?.array;
    try std.testing.expectEqualStrings("files::read_1", calls[0].get("function").?.get("name").?.string);
    try std.testing.expect(!calls[0].get("function").?.has("namespace"));
    try std.testing.expectEqualStrings("read", history.array[0].get("tool_calls").?.array[0].get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("old::gone", calls[1].get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("old::gone", try set.name(&cx, "old", "gone"));
}

test "custom declarations and namespace validation" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const patch = try Set.init(&cx, (try json.parse(a, "[{\"type\":\"custom\",\"name\":\"apply_patch\"}]")).ok);
    try std.testing.expect(patch.custom_patch);
    const parameters = patch.offered.?[0].get("function").?.get("parameters").?;
    try std.testing.expectEqualStrings("string", parameters.get("properties").?.get("input").?.get("type").?.string);
    try std.testing.expectEqualStrings("input", parameters.get("required").?.array[0].string);
    const invalid = [_][]const u8{
        "[{\"type\":\"custom\",\"name\":\"other\"}]",
        "[{\"type\":\"namespace\",\"name\":\"n\",\"tools\":[{\"type\":\"custom\",\"name\":\"apply_patch\"}]}]",
        "[{\"type\":\"namespace\",\"name\":\"n\",\"tools\":[{\"type\":\"function\",\"name\":\"f\"},{\"type\":\"function\",\"name\":\"f\"}]}]",
        "[{\"type\":\"namespace\",\"name\":\"\",\"tools\":[]}]",
    };
    for (invalid) |raw| try std.testing.expectError(error.Refused, Set.init(&cx, (try json.parse(a, raw)).ok));
}

test "namespace aliases avoid the parser's case folded name collisions" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const decl = (try json.parse(a,
        \\[{"type":"namespace","name":"Files","tools":[{"type":"function","name":"read"}]},{"type":"namespace","name":"files","tools":[{"type":"function","name":"read"}]}]
    )).ok;
    const set = try Set.init(&cx, decl);
    try std.testing.expectEqualStrings("Files::read", set.namespaces.items[0].flat);
    try std.testing.expectEqualStrings("files::read_1", set.namespaces.items[1].flat);
    try std.testing.expectEqualStrings("files", set.resolve("files::read_1").?.namespace);
}
