"""Focused CPU contract: consume a fixture produced by namespace_test_root.zig.

Run with TF_NAMESPACE_FIXTURE pointing at that test's retained disposable root.
No native namespace is synthesized for the acceptance case; checklib is unchanged.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class NamespaceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("TF_NAMESPACE_FIXTURE"):
            raise unittest.SkipTest("set TF_NAMESPACE_FIXTURE to the disposable native spill fixture")
        cls.root = Path(os.environ["TF_NAMESPACE_FIXTURE"]).resolve(strict=True)
        source = Path(__file__).resolve().parents[1] / "tools/mia/checklib.py"
        tree = ast.parse(source.read_text())
        cls.fsops = next(ast.literal_eval(n.value) for n in tree.body
                         if isinstance(n, ast.Assign) and any(
                             isinstance(v, ast.Name) and v.id == "FSOPS" for v in n.targets))
        cls.namespace = "native-spill-" + "5a" * 32
        cls.target = next((cls.root / cls.namespace / "rank2/points").glob("*.man"))
        cls.relative = cls.target.relative_to(cls.root).as_posix()
        cls.marker = cls.root / ".chimera-spill-test"

    def mark(self, **changes):
        value = dict(schema="chimera.spill-test.v1", isolated=True,
                     rank=2, namespace=self.namespace)
        value.update(changes)
        self.marker.write_text(json.dumps(value))

    def invoke(self, path=None, rank=2, **kw):
        args = dict(op="truncate", root=str(self.root), rank=rank,
                    path=path or self.relative, to=0, **kw)
        return subprocess.run([sys.executable, "-I", "-c", self.fsops, json.dumps(args)],
                              capture_output=True, text=True, check=False)

    def refused(self, path=None, rank=2):
        before = self.target.read_bytes()
        result = self.invoke(path, rank)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.target.read_bytes(), before)

    def test_native_fixture_acceptance_and_guard_refusals(self):
        original = self.target.read_bytes()
        self.assertGreater(len(original), 32)
        # Refusals precede the destructive acceptance check; retain original fixture until last.
        self.marker.unlink(missing_ok=True)
        self.refused()
        for changes in (dict(namespace=self.namespace[:29]),
                        dict(namespace="native-spill-" + "6b" * 32),
                        dict(rank=0), dict(isolated=False)):
            with self.subTest(marker=changes):
                self.mark(**changes)
                self.refused()
        self.mark()
        self.refused(rank=0)
        self.refused(self.relative.replace("rank2", "rank0"))
        self.refused(self.relative.replace("/points/", "/points/../points/"))
        self.refused("/" + self.relative)
        # Actual link targets remain native-created; no hand-crafted data substitutes.
        symlink = self.target.with_name("namespace-symlink.man")
        symlink.symlink_to(self.target.name)
        try:
            self.refused(symlink.relative_to(self.root).as_posix())
        finally:
            symlink.unlink()
        hardlink = self.target.with_name("namespace-hardlink.man")
        hardlink.hardlink_to(self.target)
        try:
            self.refused(hardlink.relative_to(self.root).as_posix())
            self.refused()  # original path is also shared
        finally:
            hardlink.unlink()
        alias = "native-spill-" + "ab" * 32
        directory_link = self.root / alias
        directory_link.symlink_to(self.root / self.namespace, target_is_directory=True)
        try:
            # A valid full64 marker/path still cannot authorize following a directory symlink.
            self.mark(namespace=alias)
            self.refused(self.relative.replace(self.namespace, alias))
        finally:
            directory_link.unlink()
        self.mark()
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["size"], len(original))
        self.assertEqual(self.target.read_bytes(), b"")
        # The disposable qualification fixture deliberately remains truncated as evidence.


if __name__ == "__main__":
    unittest.main()
