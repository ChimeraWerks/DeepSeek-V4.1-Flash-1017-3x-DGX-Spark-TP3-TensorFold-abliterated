//! Reply text as it streams: think blocks, tool markup held back, Harmony channels, stop strings.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Markers = struct { open: []const u8, close: []const u8 };
/// Qwen's prompt opens the think block; Gemma 4's reply opens its thought channel.
pub const think_markers: Markers = .{ .open = "", .close = "</think>" };
/// The opener a reply writes itself when its template leaves the think block to it (Kolibri 1's does).
const think_open = "<think>";
pub const channel_markers: Markers = .{ .open = "<|channel>thought", .close = "<channel|>" };
/// (opener, closer) of a tool call's markup: Qwen's, Gemma 4's, DeepSeek-V4's DSML block, DeepSeek-V4.1's. DeepSeek-V4.1's
/// reference parser (encoding.py, parse_message_from_completion_text) starts the calls at "\n\n<｜DSML｜ calls": the two
/// newlines are the calls', not the text's, and the template writes them back in front of the block. The entry with them
/// keeps them out of the streamed text and holds a trailing newline back until the next token says whether calls follow;
/// the one without still reads a block the model opened without them.
pub const calls = [_][2][]const u8{
    .{ "<tool_call>", "</tool_call>" },
    .{ "<|tool_call>", "<tool_call|>" },
    .{ "<\u{ff5c}DSML\u{ff5c}tool_calls>", "</\u{ff5c}DSML\u{ff5c}tool_calls>" },
    .{ "\n\n<\u{ff5c}DSML\u{ff5c} calls>", "</\u{ff5c}DSML\u{ff5c} calls>" },
    .{ "<\u{ff5c}DSML\u{ff5c} calls>", "</\u{ff5c}DSML\u{ff5c} calls>" },
};

/// Byte ranges decoded from actual special IDs, not ordinary tokens spelling the same text.
pub const Span = struct { start: usize, end: usize };
pub const Markup = struct {
    text: []const u8 = "",
    spans: []const Span = &.{},
    token_aware: bool = false,

    pub fn allows(m: Markup, text: []const u8, at: usize, tag: []const u8) bool {
        if (!m.token_aware) return true;
        const offset = @intFromPtr(text.ptr) -| @intFromPtr(m.text.ptr);
        // A copy of the reply (a joined prefix, hidden call text) has no token map: fall back to spelling rather than lose its markers.
        if (@intFromPtr(text.ptr) < @intFromPtr(m.text.ptr) or offset + text.len > m.text.len) return true;
        const dsml = "\u{ff5c}DSML\u{ff5c}";
        var start = at;
        var len = tag.len;
        if (std.mem.indexOf(u8, tag, dsml)) |i| {
            start += i;
            len = dsml.len;
        } else {
            // Channel names are ordinary text after their special delimiter.
            for ([_][]const u8{ "<|channel>", "<|channel|>" }) |prefix| if (std.mem.startsWith(u8, tag, prefix)) {
                len = prefix.len;
                break;
            };
        }
        for (m.spans) |span| if (span.start <= offset + start and span.end >= offset + start + len) return true;
        return false;
    }

    pub fn find(m: Markup, text: []const u8, from: usize, tag: []const u8) ?usize {
        var at = from;
        while (findTag(text, at, tag)) |i| {
            if (m.allows(text, i, tag)) return i;
            at = i + 1;
        }
        return null;
    }

    pub fn partial(m: Markup, text: []const u8, tag: []const u8) usize {
        const held = partialTag(text, tag);
        if (held == 0 or !m.token_aware) return held;
        const prefix = tag[0..held];
        // A not-yet-written token can follow the '<' or paragraph break already decoded.
        const dsml = "\u{ff5c}DSML\u{ff5c}";
        if (std.mem.indexOf(u8, tag, dsml)) |i| {
            if (held <= i) return held;
            if (held < i + dsml.len) return held;
            return if (m.allows(text, text.len - held, tag[0 .. i + dsml.len])) held else 0;
        }
        // Special tokens arrive whole. A plain prefix must not be interpreted as control syntax.
        return if (m.allows(text, text.len - held, prefix)) held else 0;
    }
};

/// Decode ordinary runs together (including split UTF-8) and retain each special token's byte span.
pub const TokenText = struct {
    text: std.ArrayList(u8) = .empty,
    spans: std.ArrayList(Span) = .empty,
    plain: Incremental = .{},
    copied: usize = 0,

    pub fn markup(t: *const TokenText) Markup {
        return .{ .text = t.text.items, .spans = t.spans.items, .token_aware = true };
    }

    pub fn pending(t: *const TokenText) bool {
        return t.plain.pending();
    }

    pub fn extend(t: *TokenText, a: Allocator, source: anytype, tokens: []const u32, finished: bool) Allocator.Error![]const u8 {
        for (tokens) |id| {
            if (source.isOutputControl(id)) {
                const piece = try source.decode(a, &.{id});
                try t.flush(a, source);
                const start = t.text.items.len;
                try t.text.appendSlice(a, piece);
                try t.spans.append(a, .{ .start = start, .end = t.text.items.len });
            } else {
                const all = try t.plain.extend(a, source, decode, &.{id});
                try t.text.appendSlice(a, all[t.copied..]);
                t.copied = all.len;
            }
        }
        if (finished) try t.flush(a, source);
        return t.text.items;
    }

    fn decode(source: anytype, a: Allocator, tokens: []const u32) Allocator.Error![]const u8 {
        return source.decode(a, tokens);
    }

    fn flush(t: *TokenText, a: Allocator, source: anytype) Allocator.Error!void {
        if (t.plain.pending()) {
            const before = try source.decode(a, t.plain.tokens.items[t.plain.prefix..t.plain.read]);
            const after = try source.decode(a, t.plain.tokens.items[t.plain.prefix..]);
            try t.text.appendSlice(a, afterChars(after, charCount(before)));
        }
        t.plain = .{};
        t.copied = 0;
    }
};

/// ``text.find(tag, from)`` led by a vector scan for the tag's first byte: markup is rare in prose, so it is fast.
pub fn findTag(text: []const u8, from: usize, tag: []const u8) ?usize {
    if (tag.len == 0) return if (from <= text.len) from else null;
    var at = from;
    while (at < text.len) {
        const i = std.mem.indexOfScalarPos(u8, text, at, tag[0]) orelse return null;
        if (std.mem.startsWith(u8, text[i..], tag)) return i;
        at = i + 1;
    }
    return null;
}

fn find(text: []const u8, tag: []const u8) ?usize {
    return findTag(text, 0, tag);
}

pub fn isChannel(m: Markers) bool {
    return std.mem.eql(u8, m.close, channel_markers.close);
}

/// How many bytes at the end of ``text`` could begin ``tag``.
pub fn partialTag(text: []const u8, tag: []const u8) usize {
    if (tag.len == 0) return 0;
    var k = @min(tag.len - 1, text.len);
    while (k > 0) : (k -= 1) if (std.mem.endsWith(u8, text, tag[0..k])) return k;
    return 0;
}

fn trimNewlines(s: []const u8) []const u8 {
    return std.mem.trimStart(u8, s, "\n");
}

pub const Split = struct { reasoning: []const u8, answer: []const u8 };

/// (reasoning, answer) of a thinking reply; while it streams, a tail that could begin a marker is held back.
pub fn splitThinking(a: Allocator, full: []const u8, finished: bool, m: Markers) Allocator.Error!Split {
    return splitThinkingWith(a, full, finished, m, .{});
}

pub fn splitThinkingWith(a: Allocator, full: []const u8, finished: bool, m: Markers, markup: Markup) Allocator.Error!Split {
    var text = full;
    if (m.open.len > 0 and !(std.mem.startsWith(u8, text, m.open) and markup.allows(text, 0, m.open))) {
        if (markup.find(text, 0, m.open)) |start| if (start > 0) {
            var prefix = text[0..start];
            prefix = prefix[0 .. prefix.len - markup.partial(prefix, m.open)]; // a stray partial opener
            const inner = try splitThinkingWith(a, text[start..], finished, m, markup);
            return .{ .reasoning = inner.reasoning, .answer = try std.mem.concat(a, u8, &.{ prefix, inner.answer }) };
        };
        if (!finished) {
            var held = @max(markup.partial(text, m.open), markup.partial(text, m.close));
            for (calls) |c| held = @max(held, markup.partial(text, c[0]));
            return .{ .reasoning = "", .answer = text[0 .. text.len - held] };
        }
        return .{ .reasoning = "", .answer = text };
    }
    if (m.open.len > 0) {
        text = trimNewlines(text[m.open.len..]);
    } else if ((std.mem.startsWith(u8, text, think_open) and markup.allows(text, 0, think_open))) {
        text = trimNewlines(text[think_open.len..]);
    } else if (!finished and std.mem.startsWith(u8, think_open, text)) {
        return .{ .reasoning = "", .answer = "" }; // held while the reply may still be writing that opener
    }
    if (markup.find(text, 0, m.close)) |end| return .{ .reasoning = text[0..end], .answer = trimNewlines(text[end + m.close.len ..]) };
    var call: ?usize = null;
    for (calls) |c| if (markup.find(text, 0, c[0])) |at| {
        if (call == null or at < call.?) call = at;
    };
    if (call) |at| {
        // a call written before the block closes: held while the block may still close, the answer if the reply ends
        if (!finished) return .{ .reasoning = text[0..at], .answer = "" };
        for (calls) |c| if (std.mem.startsWith(u8, text[at..], c[0]) and markup.find(text[at..], 0, c[1]) != null)
            return .{ .reasoning = text[0..at], .answer = text[at..] };
        return .{ .reasoning = text, .answer = "" };
    }
    var held: usize = 0;
    if (!finished) {
        held = markup.partial(text, m.close);
        for (calls) |c| held = @max(held, markup.partial(text, c[0]));
    }
    return .{ .reasoning = text[0 .. text.len - held], .answer = "" };
}

/// Hides tool-call blocks and partial opening tags while streaming, so calls arrive as deltas instead.
pub fn hideToolCalls(a: Allocator, text: []const u8, finished: bool) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    return hideInto(a, &out, text, finished);
}

/// ``hideToolCalls`` into ``out`` (cleared first); text without a call block comes back as a slice of ``text``.
/// The tags of a DSML spelling, by its block's opener (null: not a DSML block). V4's parser requires a parameter's
/// ``string`` attribute; V4.1's reads one when it is there.
const Dsml = struct { open: []const u8, invoke_open: []const u8, invoke_close: []const u8, param_open: []const u8, param_close: []const u8, attr_required: bool };

fn dsmlTags(opener: []const u8) ?Dsml {
    const D = "\u{ff5c}DSML\u{ff5c}";
    if (std.mem.endsWith(u8, opener, "<" ++ D ++ " calls>")) return .{ .open = "<" ++ D ++ " calls>", .invoke_open = "<" ++ D ++ " invoke name=\"", .invoke_close = "</" ++ D ++ " invoke>", .param_open = "<" ++ D ++ " parameter name=\"", .param_close = "</" ++ D ++ " parameter>", .attr_required = false };
    if (std.mem.eql(u8, opener, "<" ++ D ++ "tool_calls>")) return .{ .open = opener, .invoke_open = "<" ++ D ++ "invoke name=\"", .invoke_close = "</" ++ D ++ "invoke>", .param_open = "<" ++ D ++ "parameter name=\"", .param_close = "</" ++ D ++ "parameter>", .attr_required = true };
    return null;
}

/// Whether every ``closer`` in an invoke's ``body`` lies inside a parameter value, as the parser reads values: from
/// the ``>`` after a parameter's name (and ``string`` attribute) to the first parameter closer after it.
fn closersInValues(m: Markup, body: []const u8, closer: []const u8, t: Dsml) bool {
    var c = m.find(body, 0, closer);
    var p: usize = 0;
    while (c) |at| {
        const ps = m.find(body, p, t.param_open) orelse return false;
        p = ps + 1;
        const ns = ps + t.param_open.len;
        const nq = std.mem.indexOfScalarPos(u8, body, ns, '"') orelse continue;
        if (!t.attr_required and nq == ns) continue;
        var v = nq + 1;
        if (std.mem.startsWith(u8, body[v..], " string=\"true\"")) {
            v += " string=\"true\"".len;
        } else if (std.mem.startsWith(u8, body[v..], " string=\"false\"")) {
            v += " string=\"false\"".len;
        } else if (t.attr_required) continue;
        if (v >= body.len or body[v] != '>') continue;
        v += 1;
        const pe = m.find(body, v, t.param_close) orelse continue;
        if (at < v) return false; // a closer before this value, outside any value
        var next: ?usize = at;
        while (next) |n| {
            if (n >= pe) break;
            next = m.find(body, n + closer.len, closer);
        }
        c = next;
        p = pe + t.param_close.len;
    }
    return true;
}

/// Whether ``rest`` is a proper start of ``tag``: the tag may still be being written.
fn begins(rest: []const u8, tag: []const u8) bool {
    return rest.len < tag.len and std.mem.startsWith(u8, tag, rest);
}

fn tagAt(m: Markup, text: []const u8, at: usize, tag: []const u8) bool {
    return std.mem.startsWith(u8, text[at..], tag) and m.allows(text, at, tag);
}

const Walk = union(enum) { closer: usize, more, broken };

/// Reads a DSML block from ``from`` (just past its opener) the way the parser reads it: invokes, each a name up to the
/// first quote followed by ``">``, each running to the first invoke closer after it, with only whitespace between them.
/// The block's closer is the first one met between invokes, so a closer or an opener inside a parameter value (a file
/// about DSML) neither ends nor splits the block. ``more``: the text ends inside the block; ``broken``: something else
/// appears, a name holds a tag, or an invoke holds the block's closer outside its values.
fn walkDsml(m: Markup, text: []const u8, from: usize, t: Dsml, closer: []const u8) Walk {
    var p = from;
    while (true) {
        while (p < text.len and isSpaceByte(text[p])) p += 1;
        if (p >= text.len) return .more;
        const rest = text[p..];
        if (tagAt(m, text, p, closer)) return .{ .closer = p };
        if (!tagAt(m, text, p, t.invoke_open)) return if (begins(rest, closer) or begins(rest, t.invoke_open)) .more else .broken;
        const ns = p + t.invoke_open.len;
        const q = std.mem.indexOfScalarPos(u8, text, ns, '"') orelse
            return if (std.mem.indexOfScalar(u8, text[ns..], '<') != null) .broken else .more;
        if (std.mem.indexOfScalar(u8, text[ns..q], '<') != null) return .broken;
        if (q + 2 > text.len) return .more;
        if (!std.mem.startsWith(u8, text[q..], "\">")) return .broken;
        const e = m.find(text, q + 2, t.invoke_close) orelse return .more;
        // the invoke may hold the block's closer only inside its parameter values (a file about DSML)
        if (!closersInValues(m, text[q + 2 .. e], closer, t)) return .broken;
        p = e + t.invoke_close.len;
    }
}

/// How many invokes in ``region`` the parser reads whole: a name up to its quote, then ``">``, then the first invoke
/// closer after it (``dsml41Calls``' reading, which skips any invoke that does not complete).
fn wholeInvokes(m: Markup, region: []const u8, t: Dsml) usize {
    var n: usize = 0;
    var pos: usize = 0;
    while (m.find(region, pos, t.invoke_open)) |s| {
        pos = s + 1;
        const ns = s + t.invoke_open.len;
        const q = std.mem.indexOfScalarPos(u8, region, ns, '"') orelse continue;
        if (q == ns or !std.mem.startsWith(u8, region[q..], "\">")) continue;
        const e = m.find(region, q + 2, t.invoke_close) orelse continue;
        pos = e + t.invoke_close.len;
        n += 1;
    }
    return n;
}

/// Where the call block opened before ``from`` closes: the index of its closer, or null while it may still close. A
/// DSML block's string values are not escaped, so a value can hold the block's own closer (a file about DSML). A block
/// that reads as the parser reads it (``walkDsml``) closes at the first closer between its invokes, not at one inside
/// an invoke, but only when that reading gives the parser more whole invokes than the first-closer rule gives over the
/// same text (that block and the blocks it would have been split into): otherwise, and in every other case (a block
/// that does not read that way, a reply that ended inside one, Qwen's and Gemma's markup), the first closer closes it.
/// While the reply streams, a DSML block still being written stays hidden.
pub fn blockEnd(m: Markup, text: []const u8, from: usize, opener: []const u8, closer: []const u8, finished: bool) ?usize {
    const first = m.find(text, from, closer);
    const t = dsmlTags(opener) orelse return first;
    switch (walkDsml(m, text, from, t, closer)) {
        .closer => |at| {
            if (first == null or at == first.?) return at;
            // the first-closer rule over the same text: the block to the first closer, then each block it opens after it
            var split = wholeInvokes(m, text[from..first.?], t);
            var pos = first.? + closer.len;
            while (m.find(text[0..at], pos, t.open)) |o| {
                const e = m.find(text, o + t.open.len, closer) orelse break;
                split += wholeInvokes(m, text[o + t.open.len .. e], t);
                pos = e + closer.len;
            }
            return if (wholeInvokes(m, text[from..at], t) > split) at else first;
        },
        .more => return if (finished) first else null,
        .broken => return first,
    }
}

pub fn hideInto(a: Allocator, out: *std.ArrayList(u8), text: []const u8, finished: bool) Allocator.Error![]const u8 {
    return hideIntoWith(a, out, text, finished, .{});
}

pub fn hideIntoWith(a: Allocator, out: *std.ArrayList(u8), text: []const u8, finished: bool, markup: Markup) Allocator.Error![]const u8 {
    out.clearRetainingCapacity();
    var pos: usize = 0;
    while (true) {
        var start: ?usize = null;
        var which: usize = 0;
        for (calls, 0..) |c, i| if (markup.find(text, pos, c[0])) |at| {
            if (start == null or at < start.?) {
                start = at;
                which = i;
            }
        };
        if (start == null) {
            const tail = text[pos..];
            var held: usize = 0;
            if (!finished) for (calls) |c| {
                held = @max(held, markup.partial(tail, c[0]));
            };
            if (pos == 0) return text[0 .. text.len - held];
            try out.appendSlice(a, tail[0 .. tail.len - held]);
            return out.items;
        }
        try out.appendSlice(a, text[pos..start.?]);
        const opener = calls[which][0];
        const end = blockEnd(markup, text, start.? + opener.len, opener, calls[which][1], finished) orelse return out.items;
        pos = end + calls[which][1].len;
    }
}

const harmony_final = "<|channel|>final<|message|>";
const harmony_analysis = "<|channel|>analysis<|message|>";
const harmony_ends = [_][]const u8{ "<|return|>", "<|end|>", "<|call|>", "<|start|>" };

fn cutAt(s: []const u8, marker: []const u8) []const u8 {
    return if (find(s, marker)) |i| s[0..i] else s;
}

pub const Harmony = struct { content: []const u8, reasoning: ?[]const u8 };

/// Harmony's final and analysis channels; other text passes unchanged.
pub fn parseHarmony(text: []const u8) Harmony {
    return parseHarmonyWith(text, .{});
}

pub fn parseHarmonyWith(text: []const u8, markup: Markup) Harmony {
    if (markup.find(text, 0, "<|channel|>") == null) return .{ .content = text, .reasoning = null };
    var reasoning: ?[]const u8 = null;
    if (harmonyFind(markup, text, harmony_analysis)) |i| {
        var r = text[i + harmony_analysis.len ..];
        r = harmonyCut(markup, r, harmony_final);
        for (harmony_ends) |e| r = harmonyCut(markup, r, e);
        reasoning = r;
    }
    const f = harmonyFind(markup, text, harmony_final) orelse return .{ .content = "", .reasoning = reasoning };
    var content = text[f + harmony_final.len ..];
    for (harmony_ends) |e| content = harmonyCut(markup, content, e);
    return .{ .content = content, .reasoning = reasoning };
}

/// The part of partly decoded output that should stream.
pub fn streamingVisible(text: []const u8) []const u8 {
    return streamingVisibleWith(text, .{});
}

pub fn streamingVisibleWith(text: []const u8, markup: Markup) []const u8 {
    if (markup.find(text, 0, "<|channel|>") == null) return text;
    return parseHarmonyWith(text, markup).content;
}

pub fn stripTrailing(tokens: []const u32, stops: []const u32) []const u32 {
    var end = tokens.len;
    while (end > 0 and std.mem.indexOfScalar(u32, stops, tokens[end - 1]) != null) end -= 1;
    return tokens[0..end];
}

/// A thinking reply's reasoning tokens: through its close, else all of them.
pub fn reasoningCount(tokens: []const u32, think_end: ?u32) usize {
    const end = think_end orelse return 0;
    return if (std.mem.indexOfScalar(u32, tokens, end)) |i| i + 1 else tokens.len;
}

/// Python's ``s[n:]`` with ``n`` counted in characters.
pub fn afterChars(s: []const u8, n: usize) []const u8 {
    var seen: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] & 0xc0 == 0x80) continue;
        if (seen == n) return s[i..];
        seen += 1;
    }
    return "";
}

pub fn charCount(s: []const u8) usize {
    var n: usize = 0;
    for (s) |b| n += @intFromBool(b & 0xc0 != 0x80);
    return n;
}

/// Python's ``str.strip()`` for the whitespace replies hold.
pub fn pyStrip(s: []const u8) []const u8 {
    var start: usize = 0;
    var end = s.len;
    while (start < end) {
        const n = spaceAt(s, start);
        if (n == 0) break;
        start += n;
    }
    while (end > start) {
        var back: usize = 1;
        while (back < 4 and end - back > start and s[end - back] & 0xc0 == 0x80) back += 1;
        if (spaceAt(s[0..end], end - back) != back) break;
        end -= back;
    }
    return s[start..end];
}

/// ``str.lstrip()``: the leading whitespace off.
pub fn pyLstrip(s: []const u8) []const u8 {
    return s[@intFromPtr(pyStrip(s).ptr) - @intFromPtr(s.ptr) ..];
}

/// ``str.rstrip()``: the trailing whitespace off.
pub fn pyRstrip(s: []const u8) []const u8 {
    const stripped = pyStrip(s);
    if (stripped.len == 0) return "";
    return s[0 .. @intFromPtr(stripped.ptr) - @intFromPtr(s.ptr) + stripped.len];
}

/// Python's ``\s`` on one byte (ASCII whitespace and the separators below space).
pub fn isSpaceByte(ch: u8) bool {
    return ch == ' ' or (ch >= 0x09 and ch <= 0x0d) or (ch >= 0x1c and ch <= 0x1f);
}

fn spaceAt(s: []const u8, i: usize) usize {
    const c = s[i];
    if (c == ' ' or (c >= 0x09 and c <= 0x0d) or (c >= 0x1c and c <= 0x1f)) return 1;
    if (c < 0x80) return 0;
    const n = std.unicode.utf8ByteSequenceLength(c) catch return 0;
    if (i + n > s.len) return 0;
    const cp = std.unicode.utf8Decode(s[i .. i + n]) catch return 0;
    return switch (cp) {
        0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => n,
        else => 0,
    };
}

/// Request stop strings: the visible text ends before the first, and a possible start is held while streaming.
pub const Stops = struct {
    strings: []const []const u8,

    pub fn visible(s: Stops, text: []const u8, partial: bool) []const u8 {
        var first: ?usize = null;
        for (s.strings) |stop| if (std.mem.indexOf(u8, text, stop)) |at| {
            if (first == null or at < first.?) first = at;
        };
        if (first) |at| return text[0..at];
        if (!partial) return text;
        var held: usize = 0;
        for (s.strings) |stop| {
            var size: usize = 1;
            while (size < stop.len and size <= text.len) : (size += 1) {
                if (std.mem.endsWith(u8, text, stop[0..size])) held = @max(held, size);
            }
        }
        return text[0 .. text.len - held];
    }

    /// The first stop the text holds, ties in request order (``matched_stop``).
    pub fn matched(s: Stops, text: []const u8) ?[]const u8 {
        var best: ?usize = null;
        var which: []const u8 = "";
        for (s.strings) |stop| if (std.mem.indexOf(u8, text, stop)) |at| {
            if (best == null or at < best.?) {
                best = at;
                which = stop;
            }
        };
        return if (best != null) which else null;
    }

    /// Tokens the engine's stop check decodes: the longest stop's length plus 8.
    pub fn tail(s: Stops) usize {
        var longest: usize = 0;
        for (s.strings) |stop| longest = @max(longest, charCount(stop));
        return longest + 8;
    }
};

/// Decodes only new tokens with the previous chunk as context, so characters split across tokens stay whole.
pub const Incremental = struct {
    tokens: std.ArrayList(u32) = .empty,
    text: std.ArrayList(u8) = .empty,
    prefix: usize = 0,
    read: usize = 0,

    pub fn extend(inc: *Incremental, a: Allocator, ctx: anytype, decode: anytype, more: []const u32) ![]const u8 {
        try inc.tokens.appendSlice(a, more);
        const before = try decode(ctx, a, inc.tokens.items[inc.prefix..inc.read]);
        const after = try decode(ctx, a, inc.tokens.items[inc.prefix..]);
        const nb = charCount(before);
        if (charCount(after) > nb and !std.mem.endsWith(u8, after, "\u{fffd}")) {
            try inc.text.appendSlice(a, afterChars(after, nb));
            inc.prefix = inc.read;
            inc.read = inc.tokens.items.len;
        }
        return inc.text.items;
    }

    /// A character is still split across tokens, so streaming waits for the rest.
    pub fn pending(inc: *const Incremental) bool {
        return inc.tokens.items.len > 0 and inc.read < inc.tokens.items.len;
    }
};

test "splits and holds" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = try splitThinking(a, "plan</think>\n\nHi", true, think_markers);
    try std.testing.expectEqualStrings("plan", s.reasoning);
    try std.testing.expectEqualStrings("Hi", s.answer);
    const held = try splitThinking(a, "plan</thi", false, think_markers);
    try std.testing.expectEqualStrings("plan", held.reasoning);
    try std.testing.expectEqualStrings("a b", try hideToolCalls(a, "a <tool_call>x</tool_call>b<tool", false));
    const stops: Stops = .{ .strings = &.{"END"} };
    try std.testing.expectEqualStrings("ab", stops.visible("abEN", true));
    try std.testing.expectEqualStrings("x", pyStrip(" \u{3000}x\u{a0}\n"));
    try std.testing.expectEqualStrings("lo", afterChars("h\u{e9}lo", 2));
}

test "a reply that writes its own think opener keeps the tag out of its reasoning" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = "<think>\nPlan.\n</think>\n\nDone.";
    var sent: []const u8 = "";
    for (1..reply.len + 1) |n| {
        const s = try splitThinking(a, reply[0..n], false, think_markers);
        try std.testing.expect(std.mem.indexOfScalar(u8, s.reasoning, '<') == null and std.mem.startsWith(u8, s.reasoning, sent));
        sent = s.reasoning;
    }
    for ([_][]const u8{ reply, "Plan.\n</think>\n\nDone." }) |text| {
        const s = try splitThinking(a, text, true, think_markers);
        try std.testing.expectEqualStrings("Plan.\n", s.reasoning);
        try std.testing.expectEqualStrings("Done.", s.answer);
    }
}

test "DeepSeek-V4.1's calls take their two newlines: text streams without them, and they wait for the next token" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const block = "<\u{ff5c}DSML\u{ff5c} calls>\n<\u{ff5c}DSML\u{ff5c} invoke name=\"t\">\n</\u{ff5c}DSML\u{ff5c} invoke>\n</\u{ff5c}DSML\u{ff5c} calls>";
    const reply = "I will check.\n\n" ++ block;
    // every prefix streams the text and nothing past it, and only ever grows
    var sent: []const u8 = "";
    for (1..reply.len + 1) |n| {
        const shown = try hideToolCalls(a, reply[0..n], false);
        try std.testing.expect(std.mem.startsWith(u8, "I will check.", shown) and std.mem.startsWith(u8, shown, sent));
        sent = shown;
    }
    try std.testing.expectEqualStrings("I will check.", try hideToolCalls(a, reply, true));
    // newlines that end a paragraph come out with the next token, or at the reply's end
    try std.testing.expectEqualStrings("a", try hideToolCalls(a, "a\n\n", false));
    try std.testing.expectEqualStrings("a\n\nb", try hideToolCalls(a, "a\n\nb", false));
    try std.testing.expectEqualStrings("a\n\n", try hideToolCalls(a, "a\n\n", true));
    // a block opened without them, and a third newline, which stays the text's
    try std.testing.expectEqualStrings("a", try hideToolCalls(a, "a" ++ block, true));
    try std.testing.expectEqualStrings("a\n", try hideToolCalls(a, "a\n\n\n" ++ block, true));
    // a call written inside the think block: the reasoning ends where the calls start, as the reference reads it
    const t = try splitThinking(a, "plan\n\n" ++ block, true, think_markers);
    try std.testing.expectEqualStrings("plan", t.reasoning);
}

fn harmonyFind(markup: Markup, text: []const u8, marker: []const u8) ?usize {
    return markup.find(text, 0, marker);
}

fn harmonyCut(markup: Markup, text: []const u8, marker: []const u8) []const u8 {
    return if (harmonyFind(markup, text, marker)) |i| text[0..i] else text;
}

test "token-aware thinking keeps plain markers at every byte boundary" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const literal = "literal </think> and <｜DSML｜ calls> stay in reasoning. ";
    const text = literal ++ "</think>\n\nAnswer </think> <｜DSML｜ calls> literal.";
    const spans = [_]Span{.{ .start = literal.len, .end = literal.len + "</think>".len }};
    const markup: Markup = .{ .text = text, .spans = &spans, .token_aware = true };
    var sent: []const u8 = "";
    for (0..text.len + 1) |end| {
        const split = try splitThinkingWith(a, text[0..end], false, think_markers, markup);
        try std.testing.expect(std.mem.startsWith(u8, split.reasoning, sent));
        try std.testing.expect(std.mem.startsWith(u8, literal, split.reasoning));
        sent = split.reasoning;
    }
    const final = try splitThinkingWith(a, text, true, think_markers, markup);
    try std.testing.expectEqualStrings(literal, final.reasoning);
    try std.testing.expectEqualStrings("Answer </think> <｜DSML｜ calls> literal.", final.answer);
    var hidden: std.ArrayList(u8) = .empty;
    try std.testing.expectEqualStrings(final.answer, try hideIntoWith(a, &hidden, final.answer, true, markup));
    const plain: Markup = .{ .text = text, .token_aware = true };
    try std.testing.expectEqualStrings(text, (try splitThinkingWith(a, text, true, think_markers, plain)).reasoning);
}

test "token decoder distinguishes a real ID from the same plain bytes" {
    const Fake = struct {
        fn isOutputControl(_: @This(), id: u32) bool {
            return id == 1000;
        }
        fn decode(_: @This(), a: Allocator, tokens: []const u32) Allocator.Error![]u8 {
            var out: std.ArrayList(u8) = .empty;
            for (tokens) |id| {
                if (id == 1000) try out.appendSlice(a, "</think>") else try out.append(a, @intCast(id));
            }
            return out.items;
        }
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var decoded: TokenText = .{};
    for ("literal </think>") |byte| _ = try decoded.extend(a, Fake{}, &.{byte}, false);
    _ = try decoded.extend(a, Fake{}, &.{ 1000, 'o', 'k' }, true);
    const split = try splitThinkingWith(a, decoded.text.items, true, think_markers, decoded.markup());
    try std.testing.expectEqualStrings("literal </think>", split.reasoning);
    try std.testing.expectEqualStrings("ok", split.answer);
    try std.testing.expectEqual(@as(usize, 1), decoded.spans.items.len);
}
