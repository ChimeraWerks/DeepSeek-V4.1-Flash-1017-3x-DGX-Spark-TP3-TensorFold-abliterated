//! The round loop on the fake target: drafted == one-token rounds, shared rounds == solo, greedy and sampled.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

const Case = struct {
    prompt: []const u32,
    max_new: u32 = 40,
    sampling: ?Sampling = null,
    drafts: bool = true,
    think_budget: u32 = 0,
};

fn model() !Config {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

/// Every case's emitted tokens, the cases admitted together and stepped until done.
fn run(cases: []const Case) ![][]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |c, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const GuardCase = struct {
    prompt: []const u32,
    max_new: u32 = 400,
    drafts: bool = true,
    think_budget: u32 = 0,
    loop_guard: bool = true,
    think_open: ?bool = null,
    cycle_after: usize = 70,
    pattern: []const u32 = &.{ 11, 12, 13 },
    answer_cycles: bool = false,
};

const GuardOut = struct {
    emitted: []u32,
    reason: sm.Reason,
    period: ?u32,
    finished: bool,
};

fn runGuard(cases: []const GuardCase, step_limit: usize) ![]GuardOut {
    var cfg = try model();
    defer cfg.deinit(gpa);
    const c = cases[0];
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = c.cycle_after, .probe_prompt = c.prompt.len, .pattern = c.pattern, .answer_cycles = c.answer_cycles };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |item, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "guard", .prompt = item.prompt, .max_new = item.max_new, .eos = &.{96}, .drafts = item.drafts, .proposer = p.proposer(), .think_budget = item.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .think_open = item.think_open, .loop_guard = item.loop_guard });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    var steps: usize = 0;
    while (engine.activeCount() > 0 and steps < step_limit) : (steps += 1) try engine.step();
    const out = try gpa.alloc(GuardOut, cases.len);
    for (out, streams) |*result, *s| result.* = .{ .emitted = try gpa.dupe(u32, s.emitted()), .reason = s.reason, .period = s.loop_period, .finished = s.finished };
    return out;
}

fn freeGuard(runs: []GuardOut) void {
    for (runs) |run_result| gpa.free(run_result.emitted);
    gpa.free(runs);
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };

test "drafted rounds commit the one-token decode, greedy and sampled" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s }});
        defer free(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }});
        defer free(plain);
        try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
        // and the fake target's own decode
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, &p1);
        for (drafted[0]) |t| {
            try std.testing.expectEqual(fake.next(history.items, s, history.items.len), t);
            try history.append(gpa, t);
        }
    }
}

test "shared rounds commit what each stream commits alone" {
    const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const together = try run(&.{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 } });
    defer free(together);
    const one = try run(&.{.{ .prompt = &p1 }});
    defer free(one);
    const two = try run(&.{.{ .prompt = &p2, .sampling = sampled, .max_new = 30 }});
    defer free(two);
    try std.testing.expectEqualSlices(u32, one[0], together[0]);
    try std.testing.expectEqualSlices(u32, two[0], together[1]);
}

test "the thinking budget's forced close is the same drafted, plain and shared" {
    const drafted = try run(&.{.{ .prompt = &p2, .think_budget = 9 }});
    defer free(drafted);
    const plain = try run(&.{.{ .prompt = &p2, .think_budget = 9, .drafts = false }});
    defer free(plain);
    const shared = try run(&.{ .{ .prompt = &p2, .think_budget = 9 }, .{ .prompt = &p1, .drafts = false } });
    defer free(shared);
    try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
    try std.testing.expectEqualSlices(u32, plain[0], shared[0]);
    try std.testing.expectEqual(@as(u32, 90), drafted[0][8]);
}

test "the loop guard matches drafted and plain, reports its period, and answers after close" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2 }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), drafted[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .stop);
    try std.testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, plain[0].emitted[329..]);

    const off = try runGuard(&.{.{ .prompt = &p2, .loop_guard = false, .think_open = true }}, 5000);
    defer freeGuard(off);
    try std.testing.expect(off[0].finished and off[0].reason == .length and off[0].emitted.len == 400);
}

test "a cycle in the answer does not refire or reclose" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), plain[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .length);
    var close_count: usize = 0;
    for (plain[0].emitted) |token| close_count += @intFromBool(token == 90);
    try std.testing.expectEqual(@as(usize, 1), close_count);
}

// 1.2.2xTF integration: actual Engine path, not a duplicate allocator model.
const be = @import("backend.zig");
const AdaptiveClock = struct {
    ms: f64=0,
    fn start(ptr: *anyopaque) void { const c: *@This()=@ptrCast(@alignCast(ptr)); c.ms=0; }
    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 { const c: *@This()=@ptrCast(@alignCast(ptr)); c.ms+=20; return c.ms; }
    fn clock(c: *@This()) be.Clock { return .{.ptr=c,.vtable=&.{.start=start,.elapsed_ms=elapsed}}; }
};
fn runAdaptiveCases(cases: []const Case, learning: bool, staggered: bool) ![][]u32 {
    return runAdaptiveTiming(cases, learning, staggered, .none);
}
fn runAdaptiveTiming(cases: []const Case, learning: bool, staggered: bool, capture: fake.Fake.CaptureTiming) ![][]u32 {
    var costs: [48]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*cost, width| cost.* = .{ .width=@intCast(width), .ms=5+@as(f64,@floatFromInt(width))*0.8 };
    var cfg=try Config.init(gpa,.{.adaptive_learning=learning,.exact_width=16,.gpu_tokens=true,.mtp=true,.speculate=true,.speculate_early=false,.drafts=4,.window_costs=&costs,.shared_costs=&costs,.mtp_step_ms=0.5,.hidden_rows=true,.batch_rows=48,.max_streams=12,.draft_streams=true},16,15);
    defer cfg.deinit(gpa);
    var target: fake.Fake=.{.gpa=gpa, .capture_timing=capture};
    defer target.deinit();
    var clock: AdaptiveClock=.{};
    var engine=Engine.init(gpa,&cfg,target.backend(),clock.clock());
    defer engine.deinit();
    const streams=try gpa.alloc(sm.Stream,cases.len);
    defer gpa.free(streams);
    const proposers=try gpa.alloc(SuffixLookup,cases.len);
    defer gpa.free(proposers);
    for (cases,streams,proposers) |case,*stream,*proposer| {
        proposer.*=try SuffixLookup.init(gpa,.{.min_match=4});
        stream.*=try sm.Stream.init(gpa,.{.id="adaptive-integration",.prompt=case.prompt,.max_new=case.max_new,.eos=&.{96},.sampling=case.sampling,.drafts=case.drafts,.proposer=proposer.proposer(),.think_budget=case.think_budget,.think_close=&.{90,91,92},.think_end=91});
    }
    defer for (streams,proposers) |*stream,*proposer| { stream.deinit(gpa); proposer.deinit(); };
    const first=if(staggered) @max(1,cases.len/2) else cases.len;
    for (streams[0..first]) |*stream| try engine.addStream(stream);
    var added=first;
    var steps: usize=0;
    while (engine.activeCount()>0 or added<cases.len) : (steps+=1) {
        if (added<cases.len and (steps>=3 or engine.activeCount()==0)) {
            for (streams[added..]) |*stream| try engine.addStream(stream);
            added=cases.len;
        }
        if (steps>1000) return error.AdaptiveIntegrationDidNotFinish;
        try engine.step();
    }
    if (learning) {
        try std.testing.expect(engine.learner!=null);
        try std.testing.expect(engine.learner.?.rounds>0);
        try std.testing.expect(engine.learner.?.acceptance_updates>0);
        // The fake target can reach EOS in its first two cold rounds. Such a
        // completion must not manufacture warm timing samples.
        if (capture == .verify) {
            try std.testing.expectEqual(@as(u64,0),engine.learner.?.cost_updates);
        } else if(engine.learner.?.rounds>2) {
            try std.testing.expect(engine.learner.?.cost_updates>0);
        } else try std.testing.expectEqual(@as(u64,0),engine.learner.?.cost_updates);
        if (capture != .none) {
            for (engine.learner.?.overhead) |cell| try std.testing.expectEqual(@as(u32, 0), cell.count);
            try std.testing.expect(target.timing_begins > 0);
        }
    } else try std.testing.expect(engine.learner==null);
    const out=try gpa.alloc([]u32,cases.len);
    for(out,streams) |*value,*stream| value.*=try gpa.dupe(u32,stream.emitted());
    return out;
}

test "active adaptive engine handles forced thinking closure mixed plain streams and early finish" {
    const cases=[_]Case{
        .{.prompt=&p2,.max_new=48,.think_budget=9},
        .{.prompt=&p1,.max_new=8,.drafts=false},
        .{.prompt=&p2,.max_new=1},
        .{.prompt=&p1,.max_new=52},
    };
    const old=try runAdaptiveCases(&cases,false,false);
    defer free(old);
    const new=try runAdaptiveCases(&cases,true,true);
    defer free(new);
    for(old,new) |a,b| try std.testing.expectEqualSlices(u32,a,b);
}

test "active adaptive engine matches original and serial for C1 through C12 greedy sampled and join leave" {
    for ([_]?Sampling{null,.{.seed=42,.temperature=0.7,.top_k=0,.top_p=0.95}}) |sampling| {
        var cases: [12]Case=undefined;
        for(1..13) |n| {
            for(cases[0..n],0..) |*case,i| case.*=.{.prompt=if(i%2==0) &p1 else &p2,.max_new=@intCast(32+i),.sampling=sampling};
            const reference=try runAdaptiveCases(cases[0..n],false,false);
            defer free(reference);
            const learned=try runAdaptiveCases(cases[0..n],true,true);
            defer free(learned);
            for(reference,learned,0..) |old,new,i| {
                try std.testing.expectEqualSlices(u32,old,new);
                const serial=try run(&.{.{.prompt=cases[i].prompt,.max_new=cases[i].max_new,.sampling=sampling,.drafts=false}});
                defer free(serial);
                try std.testing.expectEqualSlices(u32,serial[0],new);
            }
        }
    }
}

const grammar = @import("grammar.zig");

/// A forced sequence checks first-token masks, held and grammar-drafted verify rows, and stop handling.
fn constrainedCase(drafts: bool, mixed: bool, thinking: bool, sample: ?Sampling) !struct { guided: []u32, plain: ?[]u32, rounds: u64 } {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var seq: grammar.Sequence = .{ .want = &.{ 5, 6, 7, 8, 9, 10 }, .stop = 96, .gpa = gpa };
    defer seq.log.deinit(gpa);
    var constraint: grammar.Constraint = .init(seq.rules(), (fake.vocab + 31) / 32, if (thinking) 91 else null);
    var proposer = try SuffixLookup.init(gpa, .{ .min_match = 1000 });
    defer proposer.deinit();
    var guided = try sm.Stream.init(gpa, .{ .id = "guided", .prompt = &p1, .max_new = 40, .eos = &.{96}, .sampling = sample, .drafts = drafts, .proposer = proposer.proposer(), .think_budget = if (thinking) 1 else 0, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .grammar = &constraint });
    defer guided.deinit(gpa);
    var plain: ?sm.Stream = null;
    var plain_proposer: ?SuffixLookup = null;
    defer {
        if (plain) |*s| s.deinit(gpa);
        if (plain_proposer) |*p| p.deinit();
    }
    if (mixed) {
        plain_proposer = try SuffixLookup.init(gpa, .{ .min_match = 1000 });
        plain = try sm.Stream.init(gpa, .{ .id = "plain", .prompt = &p2, .max_new = 40, .eos = &.{96}, .sampling = sample, .proposer = plain_proposer.?.proposer() });
        try engine.addStream(&plain.?);
    }
    try engine.addStream(&guided);
    var steps: usize = 0;
    while (engine.activeCount() > 0) : (steps += 1) {
        if (steps > 100) return error.GrammarDidNotFinish;
        try engine.step();
    }
    try std.testing.expect(guided.finished and guided.reason == .stop);
    const out = try gpa.dupe(u32, guided.emitted());
    errdefer gpa.free(out);
    return .{ .guided = out, .plain = if (plain) |*s| try gpa.dupe(u32, s.emitted()) else null, .rounds = guided.rounds };
}

test "structured output: first, held, and grammar-drafted rows equal serial, including thinking" {
    for ([_]?Sampling{ null, .{ .seed = 37, .temperature = 0.7, .top_p = 0.95 } }) |sample| {
        for ([_]bool{ false, true }) |thinking| {
            const serial = try constrainedCase(false, false, thinking, sample);
            defer gpa.free(serial.guided);
            const drafted = try constrainedCase(true, false, thinking, sample);
            defer gpa.free(drafted.guided);
            try std.testing.expectEqualSlices(u32, serial.guided, drafted.guided);
            if (thinking) {
                try std.testing.expectEqualSlices(u32, &.{ 90, 91, 5, 6, 7, 8, 9, 10, 96 }, drafted.guided);
            } else try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7, 8, 9, 10, 96 }, drafted.guided);
            try std.testing.expect(drafted.rounds <= serial.rounds);
        }
    }
}

test "grammar drafting stays within the reply's remaining max_new" {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var seq: grammar.Sequence = .{ .want = &.{ 5, 6, 7, 8 }, .stop = 96, .gpa = gpa };
    defer seq.log.deinit(gpa);
    var constraint: grammar.Constraint = .init(seq.rules(), (fake.vocab + 31) / 32, null);
    try std.testing.expect(try constraint.follow(5));
    var s = try sm.Stream.init(gpa, .{ .id = "short", .prompt = &p1, .max_new = 2, .grammar = &constraint });
    defer s.deinit(gpa);
    try s.context.append(gpa, 5);
    s.pending = 5;
    const windows = @import("windows.zig");
    var short: windows.Plan = .{ .stream = &s, .position = s.cache_len, .kind = .none };
    try windows.constrain(&engine, &short);
    try std.testing.expectEqual(@as(usize, 0), short.tokens.len);
    s.max_new = 4;
    var longer: windows.Plan = .{ .stream = &s, .position = s.cache_len, .kind = .none };
    try windows.constrain(&engine, &longer);
    try std.testing.expectEqualSlices(u32, &.{ 6, 7 }, longer.tokens);
    try std.testing.expectEqual(@as(u32, 1), longer.grammar_from);
}

test "grammar masked shared rounds without extra drafts do not train plain cost or adaptive acceptance" {
    // This constraint leaves every candidate valid but still supplies a mask for every verify row.
    // A shared round cannot grow grammar drafts, so grammar_from stays zero throughout.
    const MaskOnly = struct {
        fn rules(self: *@This()) grammar.Rules {
            return .{ .ptr = self, .vtable = &.{ .accept = acceptFn, .rollback = rollbackFn, .terminated = terminatedFn, .fill = fillFn, .drafts = draftsFn, .free = freeFn } };
        }
        fn acceptFn(_: *anyopaque, _: u32) anyerror!bool { return true; }
        fn rollbackFn(_: *anyopaque, _: usize) anyerror!void {}
        fn terminatedFn(_: *anyopaque) bool { return false; }
        fn fillFn(_: *anyopaque, words: []u32) anyerror!void { @memset(words, std.math.maxInt(u32)); }
        fn draftsFn(_: *anyopaque, _: []u32) anyerror!usize { return 0; }
        fn freeFn(_: *anyopaque) void {}
    };
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*cost, width| cost.* = .{ .width = @intCast(width), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(width)) };
    var cfg = try Config.init(gpa, .{ .adaptive_learning = true, .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .shared_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 2, .draft_streams = true }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: AdaptiveClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var first_rules: MaskOnly = .{};
    var second_rules: MaskOnly = .{};
    var first_constraint: grammar.Constraint = .init(first_rules.rules(), (fake.vocab + 31) / 32, null);
    var second_constraint: grammar.Constraint = .init(second_rules.rules(), (fake.vocab + 31) / 32, null);
    var example_arena = std.heap.ArenaAllocator.init(gpa);
    defer example_arena.deinit();
    const example = try first_constraint.window(example_arena.allocator(), &.{ 5, 6 }, &.{ -1, 0 }, false, 2);
    try std.testing.expect(example.masks.len > 0 and example.extra.len == 0);
    var first = try sm.Stream.init(gpa, .{ .id = "masked-1", .prompt = &p1, .max_new = 24, .eos = &.{96}, .drafts = true, .grammar = &first_constraint });
    defer first.deinit(gpa);
    var second = try sm.Stream.init(gpa, .{ .id = "masked-2", .prompt = &p2, .max_new = 24, .eos = &.{96}, .drafts = true, .grammar = &second_constraint });
    defer second.deinit(gpa);
    try engine.addStream(&first);
    try engine.addStream(&second);
    var steps: usize = 0;
    while (engine.activeCount() > 0) : (steps += 1) {
        if (steps > 50) return error.MaskedRoundsDidNotFinish;
        try engine.step();
    }
    const learner = engine.learner orelse return error.MissingAdaptiveController;
    try std.testing.expect(target.rounds > 2);
    try std.testing.expectEqual(@as(u64, 0), learner.cost_updates);
    try std.testing.expectEqual(@as(u64, 0), learner.acceptance_updates);
    try std.testing.expect(first.drafted + second.drafted > 0);
}

test "structured output: a masked stream beside a plain stream leaves the plain tokens unchanged" {
    for ([_]?Sampling{ null, .{ .seed = 3, .temperature = 0.9, .top_p = 0.9 } }) |sample| {
        const mixed = try constrainedCase(true, true, false, sample);
        defer gpa.free(mixed.guided);
        defer gpa.free(mixed.plain.?);
        try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7, 8, 9, 10, 96 }, mixed.guided);
        const solo = try run(&.{.{ .prompt = &p2, .sampling = sample }});
        defer free(solo);
        try std.testing.expectEqualSlices(u32, solo[0], mixed.plain.?);
    }
}

test "graph telemetry excludes verify and draft capture timing but keeps acceptance and tokens" {
    const cases = [_]Case{
        .{ .prompt = &p2, .max_new = 80 },
        .{ .prompt = &p1, .max_new = 80, .sampling = .{ .seed = 17 } },
    };
    // C1 covers familyRound, C2 covers roundStreams; both eventually finish.
    for ([_]usize{ 1, 2 }) |count| {
        const clean = try runAdaptiveTiming(cases[0..count], true, false, .none);
        defer free(clean);
        for ([_]fake.Fake.CaptureTiming{ .verify, .draft }) |capture| {
            const dirty = try runAdaptiveTiming(cases[0..count], true, false, capture);
            defer free(dirty);
            for (clean, dirty) |want, got| try std.testing.expectEqualSlices(u32, want, got);
        }
    }
}

test "graph telemetry backend without hooks retains legacy clean timing" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var vt = target.backend().vtable.*;
    vt.begin_timing = null;
    vt.timing_clean = null;
    var backend = target.backend();
    backend.vtable = &vt;
    backend.beginTiming();
    try std.testing.expect(backend.timingClean());
    try std.testing.expectEqual(@as(usize, 0), target.timing_begins);
}

// The call gate's first token: the drawn token, not the backend's draw handle (#80).
fn lexBlank(_: *anyopaque, _: u32) bool {
    return false;
}
fn lexText(_: *anyopaque, _: std.mem.Allocator, _: u32) ![]const u8 {
    return "";
}
fn lexEncode(_: *anyopaque, _: std.mem.Allocator, _: []const u8) ![]const u32 {
    return &.{};
}

const gated_prompt = [_]u32{ 3, 1, 4, 1, 5, 7 }; // 7 = think_open last: the gate starts disarmed

fn runGated(prompt: []const u32, gated: bool) ![]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer p.deinit();
    var s = try sm.Stream.init(gpa, .{
        .id = "gated",
        .prompt = prompt,
        .max_new = 1,
        .eos = &.{96},
        .drafts = false,
        .proposer = p.proposer(),
        .think_close = &.{ 90, 91, 92 },
        .think_end = 8,
        .call = if (gated) .{
            .opener = 30,
            .think_open = 7,
            .think_end = 8,
            .lead = "",
            .names = &.{"x"},
            .tail = "",
            .lex = .{ .ctx = @ptrCast(&cfg), .blank = lexBlank, .text = lexText, .encode = lexEncode },
        } else null,
    });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    return gpa.dupe(u32, s.emitted());
}

test "a disarmed gate commits the drawn first token, not the backend's handle" {
    const expected = fake.next(&gated_prompt, null, gated_prompt.len);
    const gated = try runGated(&gated_prompt, true);
    defer gpa.free(gated);
    try std.testing.expectEqualSlices(u32, &.{expected}, gated);
}

test "an armed gate still forces the opener" {
    const armed = [_]u32{ 3, 1, 4, 1, 5 }; // no think opener in the prompt: armed
    const plain = try runGated(&armed, false);
    defer gpa.free(plain);
    try std.testing.expectEqualSlices(u32, &.{fake.next(&armed, null, armed.len)}, plain);
    const gated = try runGated(&armed, true);
    defer gpa.free(gated);
    try std.testing.expectEqualSlices(u32, &.{30}, gated);
}


test "penalties fake backend greedy sampled draft rollback and stream isolation" {
    const penalty = @import("penalties.zig");
    const settings = [_]penalty.Settings{ .{}, .{ .presence = 1.5, .frequency = 0.7 },
        .{ .presence = -0.5, .frequency = -0.3, .repetition = 1.3, .window = 5 },
        .{ .presence = 1, .include_prompt = true, .repetition = 2, .window = 100 } };
    for ([_]f64{0,0.8}) |temperature| for (settings) |p| {
        const sample: Sampling = .{ .seed = 123, .temperature = temperature, .top_k = 20, .top_p = 0.9, .min_p = 0.05, .penalties = p };
        const prompts = [_][]const u32{ &.{1,5,1,5,1,5}, &.{2,3,2,3,2,3}, &.{8,8,8,9} };
        var together: [3]Case = undefined;
        for (&together,prompts) |*c,prompt| c.* = .{ .prompt = prompt, .sampling = sample, .drafts = true, .max_new = 100 };
        const drafted = try run(&together);
        defer free(drafted);
        for (&together) |*c| c.drafts = false;
        const serial = try run(&together);
        defer free(serial);
        for (drafted,serial,together) |d,s,c| {
            try std.testing.expectEqualSlices(u32,s,d);
            const alone = try run(&.{c});
            defer free(alone);
            try std.testing.expectEqualSlices(u32,s,alone[0]);
        }
    };
}

test "reply logprobs drafts on and off retain each committed row and discard rejected rows" {
    const lp = @import("logprob.zig");
    var wanted: std.ArrayList(lp.Row) = .empty;
    defer wanted.deinit(gpa);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    for ([_]bool{ false, true }) |drafts| {
        var cfg = try model();
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        var proposer = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        defer proposer.deinit();
        var s = try sm.Stream.init(gpa, .{ .id = "scored", .prompt = &.{ 3, 7, 2, 5 }, .max_new = 60, .sampling = .{ .seed = 171 }, .drafts = drafts, .proposer = proposer.proposer(), .logprobs = 5, .think_budget = 20, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
        defer s.deinit(gpa);
        try engine.addStream(&s);
        while (engine.activeCount() > 0) try engine.step();
        try std.testing.expectEqual(s.emitted().len, s.score_rows.items.len);
        for (s.emitted(), s.score_rows.items) |t, row| try std.testing.expectEqual(t, row.chosen.id);
        if (!drafts) {
            try wanted.appendSlice(gpa, s.score_rows.items);
            try ids.appendSlice(gpa, s.emitted());
        } else {
            try std.testing.expect(s.accepted > 0);
            try std.testing.expect(s.drafted > s.accepted);
            try std.testing.expectEqualSlices(u32, ids.items, s.emitted());
            for (wanted.items, s.score_rows.items) |a, b| {
                try std.testing.expectEqual(a.chosen.logprob, b.chosen.logprob);
                try std.testing.expectEqual(a.count, b.count);
                for (a.top[0..a.count], b.top[0..b.count]) |x, y| {
                    try std.testing.expectEqual(x.id, y.id);
                    try std.testing.expectEqual(x.logprob, y.logprob);
                }
            }
        }
    }
}

test "reply logprobs off keeps tokens rounds clocks and score allocation unchanged" {
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var clocks: u64 = 0;
    var rounds: u64 = 0;
    for ([_]bool{ false, true }) |scored| {
        var cfg = try model();
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        var s = try sm.Stream.init(gpa, .{ .id = "off", .prompt = &.{ 3, 7, 2, 5 }, .max_new = 32, .sampling = .{ .seed = 171 }, .drafts = false, .logprobs = if (scored) 0 else null });
        defer s.deinit(gpa);
        try engine.addStream(&s);
        while (engine.activeCount() > 0) try engine.step();
        if (!scored) {
            try std.testing.expect(s.score_hook == null);
            try std.testing.expectEqual(@as(usize, 0), s.score_rows.capacity);
            try ids.appendSlice(gpa, s.emitted());
            clocks = clock.calls;
            rounds = target.rounds;
        } else {
            try std.testing.expectEqualSlices(u32, ids.items, s.emitted());
            try std.testing.expectEqual(ids.items.len, s.score_rows.items.len);
            try std.testing.expectEqual(clocks, clock.calls);
            try std.testing.expectEqual(rounds, target.rounds);
        }
    }
}

test "reply logprobs a rejected merge ends only its stream and the next scored stream runs" {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .reject_score = .{ .id = "bad", .at = 3 } };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var bad = try sm.Stream.init(gpa, .{ .id = "bad", .prompt = &.{ 3, 7, 2, 5 }, .max_new = 16, .sampling = .{ .seed = 171 }, .drafts = false, .logprobs = 2 });
    defer bad.deinit(gpa);
    var good = try sm.Stream.init(gpa, .{ .id = "good", .prompt = &.{ 4, 1, 6, 5 }, .max_new = 16, .sampling = .{ .seed = 172 }, .drafts = false, .logprobs = 2 });
    defer good.deinit(gpa);
    try engine.addStream(&bad);
    try engine.addStream(&good);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(bad.finished and bad.reason == .@"error");
    try std.testing.expect(std.mem.indexOf(u8, bad.problem, "could not score") != null);
    try std.testing.expectEqual(@as(usize, 3), bad.emitted().len); // the rejected token never landed
    try std.testing.expectEqual(bad.emitted().len, bad.score_rows.items.len);
    try std.testing.expect(good.finished and good.reason != .@"error");
    try std.testing.expectEqual(good.emitted().len, good.score_rows.items.len);
    const released = target.release_calls;
    try std.testing.expect(released >= 2); // the failed stream released the normal way, with the other
    // the next scored stream (a reused backend lane) scores every token
    var next = try sm.Stream.init(gpa, .{ .id = "next", .prompt = &.{ 3, 7, 2, 5 }, .max_new = 16, .sampling = .{ .seed = 171 }, .drafts = false, .logprobs = 2 });
    defer next.deinit(gpa);
    try engine.addStream(&next);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(next.finished and next.reason != .@"error");
    try std.testing.expectEqual(next.emitted().len, next.score_rows.items.len);
    try std.testing.expect(next.emitted().len > 3);
}
