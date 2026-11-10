#!/usr/bin/env python3
"""Report rank-zero TF_DS_SLICE_TRACE JSONL. All times are host microseconds.

Slice completion and its residual synchronize are NOT CUDA event/kernel time.
Parent scopes are inclusive; phase shares below are exclusive to avoid double billing.
"""
import argparse
from collections import Counter
import json
import sys


FIELDS = ('version', 'clock', 'slot', 'start', 'rows', 'replay', 'encoder_only',
          'layers', 'wall_us', 'slack_us', 'slice_sync', 'gpu_time_known',
          'between_chunks', 'events')


def validate(chunk):
    missing = set(FIELDS) - chunk.keys()
    if missing:
        raise ValueError('missing fields: ' + ', '.join(sorted(missing)))
    if chunk['version'] != 1:
        raise ValueError('unsupported trace version')
    events = chunk['events']
    wall = chunk['wall_us']
    if not isinstance(wall, int) or wall < 0:
        raise ValueError('invalid wall_us')
    children = [[] for _ in events]
    roots = []
    for i, e in enumerate(events):
        for k in ('name', 'kind', 'begin_us', 'end_us', 'parent', 'lo', 'hi'):
            if k not in e:
                raise ValueError(f'event {i}: missing {k}')
        a, b, parent = e['begin_us'], e['end_us'], e['parent']
        if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a <= b <= wall:
            raise ValueError(f'event {i}: invalid bounds')
        if parent is None:
            roots.append(i)
        elif not isinstance(parent, int) or not 0 <= parent < i:
            raise ValueError(f'event {i}: invalid parent')
        else:
            pe = events[parent]
            if not pe['begin_us'] <= a <= b <= pe['end_us']:
                raise ValueError(f'event {i}: outside parent')
            children[parent].append(i)
    for siblings in [roots] + children:
        previous = 0
        for i in siblings:
            if events[i]['begin_us'] < previous:
                raise ValueError(f'event {i}: overlapping siblings')
            previous = events[i]['end_us']
    exclusive = [e['end_us'] - e['begin_us'] - sum(
        events[j]['end_us'] - events[j]['begin_us'] for j in children[i])
        for i, e in enumerate(events)]
    if sum(exclusive) + chunk['slack_us'] != wall:
        raise ValueError('exclusive durations plus slack do not equal wall_us')
    gap = chunk['between_chunks']
    categories = ('decode_us', 'admission_us', 'cancels_us', 'service_us', 'idle_us', 'send_us', 'other_us')
    if sum(gap[k] for k in categories) != gap['wall_us']:
        raise ValueError('between-chunk accounting does not sum to wall_us')
    return exclusive, children


def intervals(events, kind):
    """Union intervals, including nested scopes of the same kind only once."""
    result = []
    for e in sorted((e for e in events if e['kind'] == kind), key=lambda e: e['begin_us']):
        a, b = e['begin_us'], e['end_us']
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def composition(events, start, end):
    out = Counter({'slack/uninstrumented': end-start})
    for e in events:
        us = max(0, min(end, e['end_us']) - max(start, e['begin_us']))
        parent = e['parent']
        out[events[parent]['name'] if parent is not None else 'slack/uninstrumented'] -= us
        out[e['name']] += us
    return Counter({name: us for name, us in out.items() if us})


def longest_without_park(chunk):
    ends = []
    at = 0
    services = [e for e in chunk['events'] if e['name'] == 'lane_host.sliceBetween']
    boundaries = [(e['begin_us'], e['end_us']) for e in services] if services else intervals(chunk['events'], 'park')
    for a, b in boundaries:
        ends.append((at, a))
        at = b
    ends.append((at, chunk['wall_us']))
    a, b = max(ends, key=lambda ab: ab[1] - ab[0])
    return a, b, composition(chunk['events'], a, b)


def event_path(events, index):
    path = []
    while index is not None:
        e = events[index]
        bounds = f'[{e["lo"]}:{e["hi"]}]' if e['hi'] > e['lo'] else ''
        path.append(e['name'] + bounds)
        index = e['parent']
    return ' > '.join(reversed(path))


def ancestor(events, index, name):
    index = events[index]['parent']
    while index is not None:
        if events[index]['name'] == name:
            return events[index]
        index = events[index]['parent']
    return None


def engram(chunk):
    """Where this chunk's Engram rows came from: a dropped read-ahead, reuse, joins and synchronous reads."""
    events = chunk['events']
    span = lambda e: e['end_us'] - e['begin_us']
    layer = lambda i: (ancestor(events, i, 'model.runSlice.layer') or {}).get('lo', '?')
    dropped = [e for e in events if e['name'] == 'model.resetEngramAhead.dropped_ready']
    own = any(e['lo'] == chunk['start'] for e in dropped)
    count = lambda name: sum(e['name'] == name for e in events)
    spawned = sum(e['name'] == 'model.readAhead.spawn' and ancestor(events, i, 'model.openChunk') is not None
                  for i, e in enumerate(events))
    joins = [f'L{layer(i)}={span(e)}' for i, e in enumerate(events)
             if e['name'] == 'model.runSlice.engram_consumption.thread_join']
    reads = [f'L{layer(i)}={span(e)}' for i, e in enumerate(events) if e['name'] == 'prompt.engramApply'
             and any(x['parent'] == i and x['name'] == 'prompt.engramApply.hashes' for x in events)]
    yielded = sum(span(e) for e in events if e['name'] == 'lanes.prefillStepFn.yield_joinAhead')
    return (f'dropped_ready={[(e["lo"], e["hi"]) for e in dropped]} own_rows_dropped={"yes" if own else "no"} '
            f'reused={count("model.openChunk.read_ahead_reused")} inflight={count("model.openChunk.read_ahead_inflight")} '
            f'spawned_at_open={spawned} yield_join={yielded} us; consumption joins us: {" ".join(joins) or "none"}; '
            f'synchronous reads us: {" ".join(reads) or "none"}')


def longest_layers(chunks, out, limit):
    rows = [(e['end_us'] - e['begin_us'], n, i) for n, c in enumerate(chunks)
            for i, e in enumerate(c['events']) if e['name'] == 'model.runSlice.layer']
    print('Longest layer host intervals (model.runSlice.layer; a slice-end drain is outside them):', file=out)
    for us, n, i in sorted(rows, reverse=True)[:limit]:
        events = chunks[n]['events']
        e = events[i]
        print(f'  chunk {n} layer {e["lo"]}: {us} us', file=out)
        for name, part in composition(events, e['begin_us'], e['end_us']).most_common(8):
            print(f'    {part:12d} us  {name}', file=out)


def report(chunks, out=sys.stdout, limit=20):
    totals, inclusive, waits = Counter(), Counter(), []
    wall = 0
    for n, chunk in enumerate(chunks):
        exclusive, children = validate(chunk)
        wall += chunk['wall_us']
        totals['slack/uninstrumented'] += chunk['slack_us']
        for i, (e, us) in enumerate(zip(chunk['events'], exclusive)):
            totals[e['name']] += us
            inclusive[e['name']] += e['end_us'] - e['begin_us']
            if (e['kind'] == 'wait' or e['name'].endswith(('frame_send', 'fill_chunk_frame'))) and not any(chunk['events'][j]['kind'] == 'wait' for j in children[i]):
                waits.append((e['end_us'] - e['begin_us'], n, event_path(chunk['events'], i)))
    print(f'{len(chunks)} chunks; {wall} us total chunk wall time', file=out)
    print('Exclusive phase totals (nested time counted once):', file=out)
    for name, us in totals.most_common():
        print(f'  {us:12d} us  {100 * us / wall if wall else 0:7.3f}%  {name}', file=out)
    print('Inclusive phase totals (overlapping parents; do not add these shares):', file=out)
    for name, us in inclusive.most_common():
        print(f'  {us:12d} us  {100 * us / wall if wall else 0:7.3f}%  {name}', file=out)
    print('Largest blocking/API-call intervals (call overhead included):', file=out)
    for us, n, name in sorted(waits, reverse=True)[:limit]:
        print(f'  chunk {n}: {us} us  {name}', file=out)
    for n, chunk in enumerate(chunks):
        a, b, parts = longest_without_park(chunk)
        print(f'Chunk {n} slot={chunk["slot"]} start={chunk["start"]} rows={chunk["rows"]}: longest without park {b-a} us [{a}, {b}]', file=out)
        for name, us in parts.most_common():
            print(f'  {us:12d} us  {name}', file=out)
        print('  Between chunks: ' + json.dumps(chunk['between_chunks'], sort_keys=True), file=out)
        print('  Engram: ' + engram(chunk), file=out)
        for park in (e for e in chunk['events'] if e['kind'] == 'park'):
            inside = [e for e in chunk['events'] if park['begin_us'] <= e['begin_us'] <= e['end_us'] <= park['end_us']]
            rounds = [e for e in inside if e['kind'] == 'decode']
            sends = sum(b-a for a,b in intervals(inside, 'send'))
            joins = sum(e['end_us']-e['begin_us'] for e in inside if e['name'].endswith('pre_park_join'))
            syncs = sum(e['end_us']-e['begin_us'] for e in inside if e['name'].endswith('pre_park_synchronize'))
            print(f'  Park: join={joins} us sync={syncs} us decode_rounds={len(rounds)} decode={sum(e["end_us"]-e["begin_us"] for e in rounds)} us send={sends} us', file=out)
    longest_layers(chunks, out, min(limit, 10))
    print('Slice host completion, NOT pure GPU time:', file=out)
    print('  chunk lo hi completion_us enqueue_us wait_or_API_us other_host_us residual_sync_us', file=out)
    one_layer = True
    for n, chunk in enumerate(chunks):
        for e in (e for e in chunk['events'] if e['kind'] == 'slice'):
            if chunk['encoder_only'] and e['lo'] >= chunk['layers']:
                print(f'  {n:5d} {e["lo"]:2d} {e["hi"]:2d} encoder replay cut; no layer launched', file=out)
                continue
            inside = [x for x in chunk['events'] if e['begin_us'] <= x['begin_us'] <= x['end_us'] <= e['end_us'] and x is not e]
            enqueue = sum(b-a for a,b in intervals(inside, 'enqueue'))
            wait_intervals = intervals(inside, 'wait')
            enqueue_intervals = intervals(inside, 'enqueue')
            overlap = sum(max(0, min(b,d)-max(a,c)) for a,b in wait_intervals for c,d in enqueue_intervals)
            wait = sum(b-a for a,b in wait_intervals) - overlap
            sync = sum(x['end_us']-x['begin_us'] for x in inside if x['name'] == 'model.fillSlice.trace_end_synchronize')
            total = e['end_us']-e['begin_us']
            print(f'  {n:5d} {e["lo"]:2d} {e["hi"]:2d} {total:13d} {enqueue:10d} {wait:14d} {total-wait-enqueue:13d} {sync:16d}', file=out)
            one_layer &= e['hi'] == e['lo'] + 1
    if one_layer:
        print('One-layer slices: lo is the layer index. GPU execution time is unidentifiable from these host clocks alone.', file=out)
    print('No-park intervals include pre-park joins and synchronizations when decode-service boundaries are present.', file=out)
    print('Trace-end synchronization changes overlap and park-share budgets. Other host time includes enqueueing and unseparated host work, not only kernel launches.', file=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace')
    parser.add_argument('--largest', type=int, default=20)
    args = parser.parse_args()
    chunks = []
    try:
        with open(args.trace, encoding='utf-8') as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    validate(chunk)
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f'line {line_number}: {error}') from error
                chunks.append(chunk)
        report(chunks, limit=args.largest)
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
