import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_inline(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class PackageImportTest(unittest.TestCase):
    def test_aggregation_submodule_import_does_not_require_lm_eval(self):
        result = run_inline(
            """
import builtins
real_import = builtins.__import__
def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "lm_eval" or name.startswith("lm_eval."):
        raise ModuleNotFoundError("No module named 'lm_eval'")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = fake_import
from src.aggregation.average_postprocess import build_output
print(build_output.__name__)
"""
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("build_output", result.stdout)

    def test_experiment_submodule_import_does_not_require_transformers(self):
        result = run_inline(
            """
import builtins
real_import = builtins.__import__
def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "transformers" or name.startswith("transformers."):
        raise ModuleNotFoundError("No module named 'transformers'")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = fake_import
from src.experiment.think_n_bottom import resolve_selected_count
print(resolve_selected_count(repeats=4, bottom_fraction=0.5, selected_count=None))
"""
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("2", result.stdout)


if __name__ == "__main__":
    unittest.main()
