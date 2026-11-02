"""Focused offline checks for the conversation-sized RoPE preflight."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "capacity_native_launch", Path(__file__).parents[1] / "src/chimera_sparks/assets/native_launch.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class RopePrefixTests(unittest.TestCase):
    def test_large_pool_accepts_conversation_prefix(self):
        launcher.validate_rope_rows({"rows": 1048625}, 1048576, 10000000)
        launcher.validate_rope_rows({"rows": 10000049}, 1048576, 10000000)
        with self.assertRaises(ValueError):
            launcher.validate_rope_rows({"rows": 1048624}, 1048576, 10000000)

    def test_ordinary_pool_and_clamped_window(self):
        launcher.validate_rope_rows({"rows": 1049}, 1000, 1000)
        launcher.validate_rope_rows({"rows": 1049}, 2000, 1000)
        with self.assertRaises(ValueError):
            launcher.validate_rope_rows({"rows": 1048}, 1000, 1000)

    def test_prefix_read_preserves_existing_file_and_tail(self):
        with tempfile.TemporaryDirectory() as root:
            table = Path(root) / "rope.f32"
            content = bytes(range(256)) * 100
            table.write_bytes(content)
            before = table.stat()
            with table.open("rb") as stream:
                prefix = stream.read((4 + 48 + 1) * 32 * 4)
            self.assertEqual(prefix, content[:len(prefix)])
            self.assertEqual(len(prefix), 53 * 32 * 4)
            self.assertEqual(table.read_bytes(), content)
            self.assertEqual(table.stat().st_mtime_ns, before.st_mtime_ns)
