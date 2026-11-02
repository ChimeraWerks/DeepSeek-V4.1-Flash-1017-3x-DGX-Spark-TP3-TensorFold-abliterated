//! A queued request may wait for a cut another prompt is filling.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

pub const Waiting = struct { leader: ?*const Job = null, id: api.Id = 0, at: u32 = 0, since: i96 = 0, waited: bool = false };

pub const Wait = struct {
    ns: i96 = @as(i96, default_ms) * std.time.ns_per_ms,
    waits: u64 = 0,
    timeouts: u64 = 0,

    pub const default_ms: u32 = 60_000;

    pub fn fromEnv() Wait {
        return fromText(std.mem.span(std.c.getenv("TENSORFOLD_PREFIX_WAIT_MS") orelse return .{}));
    }

    pub fn fromText(text: []const u8) Wait {
        const t = std.mem.trim(u8, text, " \t");
        if (t.len == 0 or t[0] < '0' or t[0] > '9') return .{};
        const ms = std.fmt.parseInt(u32, t, 10) catch return .{};
        return .{ .ns = @as(i96, ms) * std.time.ns_per_ms };
    }
};

pub fn waiting(job: *const Job) bool {
    return job.prefix.leader != null;
}

/// The engine thread holds the host lock. Admitted fills never wait.
pub fn waits(h: *LaneHost, job: *Job, stepped: bool) bool {
    if (job.prefix.waited or h.prefix.ns <= 0 or !stepped) return false;
    const store = h.cache orelse return false;
    const r = job.request;
    if (!keyed(r)) return false;
    const have: u32 = if (store.findScoped(r.cache_namespace, r.cacheKey(), r.chunks)) |e| e.at else 0;
    const least = have +| @max(store.rules.min_gap, 1);
    var best: Waiting = .{};
    for (h.admitted.items) |l| {
        if (!l.started or l.prefilled != null or l.request.background or !keyed(l.request)) continue;
        for (l.marks) |mark| {
            // Planned marks are already chunk starts; other native backends keep at the grid floor.
            const at = if (store.rules.planned) mark else store.rules.floorGrid(mark);
            if (at <= l.reached or at < least or at <= best.at) continue;
            if (store.wouldServe(r.cache_namespace, r.cacheKey(), r.chunks, l.request.cache_namespace, l.request.cacheKey(), at))
                best = .{ .leader = l, .id = l.id, .at = at };
        }
    }
    if (best.leader == null) return false;
    best.since = std.Io.Clock.awake.now(h.io).toNanoseconds();
    best.waited = true;
    job.prefix = best;
    h.prefix.waits += 1;
    return true;
}

/// This lineage has no embedding, prompt-logprob or untaught request fields.
fn keyed(r: *const api.Request) bool {
    return r.images.len == 0 and r.cache_media.len == 0 and r.cacheKey().len > 0;
}

/// Release when the same fill reaches its cut, leaves admission, or times out.
pub fn release(h: *LaneHost) void {
    var now: ?i96 = null;
    for (h.queued.items) |job| {
        const w = job.prefix;
        const leader = w.leader orelse continue;
        const live = for (h.admitted.items) |l| {
            if (l == leader and l.id == w.id and l.started) break l;
        } else null;
        if (live) |l| if (l.reached < w.at and l.prefilled == null) {
            const t = now orelse std.Io.Clock.awake.now(h.io).toNanoseconds();
            now = t;
            if (t - w.since < h.prefix.ns) continue;
            h.prefix.timeouts += 1;
        };
        job.prefix.leader = null;
    }
}
