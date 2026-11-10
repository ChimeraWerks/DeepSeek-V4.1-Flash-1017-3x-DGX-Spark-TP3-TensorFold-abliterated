import copy
import json
from pathlib import Path
import tempfile
import unittest
from compare_load_digests import compare, read


class Digests(unittest.TestCase):
    def setUp(self):
        self.rows = {name: {'name': name, 'bytes': 16, 'sha256': '0' * 64}
                     for name in ('layers.0.ffn.experts.trellis', 'layers.0.ffn.experts.ptr0')}

    def test_exact_and_relocated_pointers(self):
        other = copy.deepcopy(self.rows)
        other['layers.0.ffn.experts.ptr0']['sha256'] = '1' * 64
        self.assertEqual(compare(self.rows, other),
                         {'matched_tensors': 1, 'address_tables_excluded': 1})
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            compare(self.rows, other, True)

    def test_every_real_change_fails(self):
        for field, value in [('bytes', 17), ('sha256', '2' * 64)]:
            other = copy.deepcopy(self.rows)
            other['layers.0.ffn.experts.trellis'][field] = value
            with self.assertRaises(ValueError):
                compare(self.rows, other)
        other = copy.deepcopy(self.rows)
        other.pop('layers.0.ffn.experts.ptr0')
        with self.assertRaisesRegex(ValueError, 'names differ'):
            compare(self.rows, other)
        other = copy.deepcopy(self.rows)
        other['layers.0.ffn.experts.ptr0']['bytes'] = 8
        with self.assertRaisesRegex(ValueError, 'size differs'):
            compare(self.rows, other)

    def test_read_rejects_empty_duplicate_or_malformed(self):
        row = self.rows['layers.0.ffn.experts.trellis']
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'digests.jsonl'
            path.write_text(json.dumps(row) + '\n')
            self.assertEqual(read(path), {row['name']: row})
            for text in ['', '\n'.join([json.dumps(row)] * 2),
                         json.dumps(dict(row, sha256='bad')),
                         json.dumps(dict(row, bytes=-1))]:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read(path)


if __name__ == '__main__':
    unittest.main()
