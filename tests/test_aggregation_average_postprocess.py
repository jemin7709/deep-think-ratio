import json
import tempfile
import unittest
from pathlib import Path

from src.aggregation.average_postprocess import build_output, find_sources


def write_postprocess(
    path: Path,
    *,
    run_dir: str,
    task: str = "aime24_custom",
    model: str = "openai/gpt-oss-120b",
    repeats: int = 4,
    k: int = 1,
    pass_at_1: float = 0.5,
    avg_at_n: float = 0.5,
    maj_at_n: float = 0.75,
) -> None:
    payload = {
        "run_dir": run_dir,
        "model": model,
        "task": task,
        "repeats": repeats,
        "k": k,
        "metrics": {
            "pass@1": pass_at_1,
            f"avg@{repeats}": avg_at_n,
            f"maj@{repeats}": maj_at_n,
            "num_docs": 30,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class AveragePostprocessTest(unittest.TestCase):
    def test_find_sources_supports_single_file_and_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            single_path = root / "postprocess_2026-03-21.json"
            write_postprocess(single_path, run_dir="/tmp/run-1")

            sources, output_dir = find_sources(single_path)
            self.assertEqual(sources, [single_path.resolve()])
            self.assertEqual(output_dir, single_path.resolve().parent)

            nested = root / "results" / "seed0"
            nested.mkdir(parents=True)
            nested_path = nested / "postprocess_2026-03-22.json"
            write_postprocess(nested_path, run_dir="/tmp/run-2")

            sources, output_dir = find_sources(root)
            self.assertEqual(output_dir, root.resolve())
            self.assertEqual(
                sources,
                [single_path.resolve(), nested_path.resolve()],
            )

    def test_build_output_averages_metrics_and_keeps_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path_a = root / "postprocess_a.json"
            path_b = root / "nested" / "postprocess_b.json"
            path_b.parent.mkdir()
            write_postprocess(
                path_a, run_dir="/tmp/run-a", pass_at_1=0.4, avg_at_n=0.4, maj_at_n=0.6
            )
            write_postprocess(
                path_b, run_dir="/tmp/run-b", pass_at_1=0.8, avg_at_n=0.8, maj_at_n=1.0
            )

            payloads = [
                json.loads(path_a.read_text(encoding="utf-8")),
                json.loads(path_b.read_text(encoding="utf-8")),
            ]
            output = build_output(root, [path_a, path_b], payloads)

            self.assertEqual(output["source_count"], 2)
            self.assertEqual(output["num_docs_per_source"], [30, 30])
            self.assertAlmostEqual(output["metrics_mean"]["pass@1"], 0.6)
            self.assertAlmostEqual(output["metrics_mean"]["avg@4"], 0.6)
            self.assertAlmostEqual(output["metrics_mean"]["maj@4"], 0.8)
            self.assertAlmostEqual(
                output["metrics_stddev"]["pass@1"], 0.282842712474619
            )
            self.assertNotIn("num_docs", output["metrics_stddev"])
            self.assertEqual(
                [entry["run_dir"] for entry in output["aggregation"]["source_metrics"]],
                ["/tmp/run-a", "/tmp/run-b"],
            )
            self.assertNotIn("source_stddev_mean", output)
            self.assertNotIn("stddev", output["aggregation"]["source_metrics"][0])

    def test_build_output_rejects_mismatched_metric_keys(self):
        payload_a = {
            "run_dir": "/tmp/run-a",
            "model": "model",
            "task": "task",
            "repeats": 4,
            "k": 1,
            "metrics": {"pass@1": 0.5, "num_docs": 30},
        }
        payload_b = {
            "run_dir": "/tmp/run-b",
            "model": "model",
            "task": "task",
            "repeats": 4,
            "k": 1,
            "metrics": {"avg@4": 0.5, "num_docs": 30},
        }

        with self.assertRaisesRegex(ValueError, "metric keys"):
            build_output(
                Path("/tmp"), [Path("/tmp/a"), Path("/tmp/b")], [payload_a, payload_b]
            )


if __name__ == "__main__":
    unittest.main()
