import copy
import importlib.util
import io
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('slice_trace_report', ROOT / 'slice_trace_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def event(name, kind, begin, end, parent=None, lo=0, hi=0):
    return dict(name=name, kind=kind, begin_us=begin, end_us=end, parent=parent, lo=lo, hi=hi)


def sample():
    return dict(version=1, clock='host_awake_us', slot=2, start=2048, rows=2048,
                replay=0, encoder_only=False, layers=2, wall_us=1000, slack_us=100,
                slice_sync=True, gpu_time_known=False,
                between_chunks=dict(wall_us=100, decode_rounds=1, decode_us=50,
                                    admission_us=10, cancels_us=5, service_us=10,
                                    idle_us=20, send_us=0, other_us=5),
                events=[event('model.fillSlice', 'slice', 0, 400, lo=0, hi=1),
                        event('engram_consumption.thread_join', 'wait', 20, 220, 0),
                        event('cuda.launch.launch', 'enqueue', 220, 250, 0),
                        event('model.fillSlice.trace_end_synchronize', 'wait', 250, 400, 0),
                        event('lanes.fillSliced.park', 'park', 400, 600),
                        event('lanes.fillSliced.pre_park_join', 'wait', 400, 430, 4),
                        event('lanes.fillSliced.pre_park_synchronize', 'wait', 430, 450, 4),
                        event('lane_host.sliceBetween', 'phase', 450, 600, 4),
                        event('core.step', 'decode', 450, 590, 7),
                        event('lane_host.send', 'send', 590, 600, 7),
                        event('model.fillSlice', 'slice', 600, 900, lo=1, hi=2),
                        event('cuda.launch.launch', 'enqueue', 600, 610, 10),
                        event('model.fillSlice.trace_end_synchronize', 'wait', 610, 900, 10)])


class TraceReportTests(unittest.TestCase):
    def test_recorded_fake_multi_chunk(self):
        chunks = [json.loads(line) for line in (ROOT / 'tests/slice_trace_recorded.jsonl').read_text().splitlines()]
        self.assertEqual(3, len(chunks))
        for c in chunks:
            exclusive, _ = report.validate(c)
            self.assertEqual(c['wall_us'], sum(exclusive) + c['slack_us'])
            self.assertEqual(5, c['layers'])
        output = io.StringIO()
        report.report(chunks, out=output)
        self.assertIn('3 chunks;', output.getvalue())
        self.assertIn('NOT pure GPU time', output.getvalue())

    def test_waits_parks_and_single_layers(self):
        c = sample()
        a, b, parts = report.longest_without_park(c)
        self.assertEqual((0, 450), (a, b))
        self.assertEqual(450, sum(parts.values()))
        self.assertEqual(200, parts['engram_consumption.thread_join'])
        self.assertEqual(30, parts['lanes.fillSliced.pre_park_join'])
        out = io.StringIO()
        report.report([c], out=out)
        text = out.getvalue()
        for expected in ('enqueue_us', 'One-layer slices', 'decode_rounds=1 decode=140 us send=10 us',
                         'join=30 us sync=20 us', '200 us  model.fillSlice[0:1] > engram_consumption.thread_join'):
            self.assertIn(expected, text)

    def test_nested_waits_not_double_counted(self):
        c = sample()
        c['events'].insert(2, event('nested_join', 'wait', 40, 100, 1))
        for e in c['events'][3:]:
            if e['parent'] is not None and e['parent'] >= 2:
                e['parent'] += 1
        exclusive, _ = report.validate(c)
        self.assertEqual(900, sum(exclusive))
        self.assertEqual(1000, sum(report.composition(c['events'], 0, 1000).values()))

    def test_invalid_bounds_parent_slack_gap_rejected(self):
        for key in ('bounds', 'parent', 'slack', 'gap'):
            c = copy.deepcopy(sample())
            if key == 'bounds': c['events'][1]['end_us'] = 401
            if key == 'parent': c['events'][1]['parent'] = 9
            if key == 'slack': c['slack_us'] = 0
            if key == 'gap': c['between_chunks']['decode_us'] = 999
            with self.subTest(key=key), self.assertRaises(ValueError):
                report.validate(c)

    def test_engram_read_ahead_and_longest_layer(self):
        c = sample()
        c['events'] = [event('lanes.resumeFill', 'phase', 0, 30),
                       event('model.resetEngramAhead', 'phase', 0, 30, 0),
                       event('model.resetEngramAhead.dropped_ready', 'phase', 30, 30, 1, lo=2048, hi=4096),
                       event('model.openChunk', 'phase', 30, 40),
                       event('model.readAhead.spawn', 'phase', 32, 35, 3),
                       event('model.fillSlice', 'slice', 40, 400, lo=1, hi=2),
                       event('model.runSlice.layer', 'phase', 40, 300, 5, lo=1, hi=2),
                       event('model.runSlice.engram_consumption.thread_join', 'wait', 50, 250, 6),
                       event('cuda.launch.launch', 'enqueue', 250, 260, 6),
                       event('model.fillSlice.trace_end_synchronize', 'wait', 300, 400, 5)]
        c['slack_us'] = 1000 - 30 - 10 - 360
        report.validate(c)
        text = report.engram(c)
        for expected in ('dropped_ready=[(2048, 4096)]', 'own_rows_dropped=yes', 'reused=0',
                         'spawned_at_open=1', 'consumption joins us: L1=200', 'synchronous reads us: none'):
            self.assertIn(expected, text)
        out = io.StringIO()
        report.report([c], out=out)
        text = out.getvalue()
        self.assertIn('chunk 0 layer 1: 260 us', text)
        self.assertIn('200 us  model.runSlice.engram_consumption.thread_join', text)

    def test_empty_trace(self):
        out = io.StringIO()
        report.report([], out=out)
        self.assertIn('0 chunks; 0 us', out.getvalue())


if __name__ == '__main__':
    unittest.main()
