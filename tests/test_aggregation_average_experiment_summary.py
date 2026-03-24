import json
import tempfile
import unittest
from pathlib import Path

from src.aggregation.average_experiment_summary import build_output
from src.aggregation.average_experiment_summary import discover_summary_paths


def write_experiment_summary(
    path: Path,
    *,
    run_dir: str,
    output_dir: str,
    task: str = "aime24_custom",
    model: str = "openai/gpt-oss-120b",
    repeats: int = 48,
    selected_count: int = 24,
    prefix_len: int = 50,
    g: float = 0.5,
    rho: float = 0.85,
    think_maj: float = 0.8,
    cons_maj: float = 0.9,
    mean_avg: float = 0.75,
    total_full_tokens: float = 1000.0,
    total_think_tokens: float = 550.0,
    saved_tokens: float = 450.0,
    saved_pct: float = 0.45,
    delta_vs_cons: float = -0.1,
    delta_vs_mean: float = 0.05,
) -> None:
    payload = {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "task": task,
        "model": model,
        "repeats": repeats,
        "selected_count": selected_count,
        "prefix_len": prefix_len,
        "g": g,
        "rho": rho,
        "summary": {
            "metrics": {
                f"think_maj@{selected_count}": think_maj,
                f"cons_maj@{repeats}": cons_maj,
                f"mean_avg@{repeats}": mean_avg,
                "num_docs": 30,
            },
            "cost": {
                "total_full_tokens": total_full_tokens,
                "total_think_tokens": total_think_tokens,
                "mean_full_tokens_per_doc": total_full_tokens / 30,
                "mean_think_tokens_per_doc": total_think_tokens / 30,
                "saved_tokens": saved_tokens,
                "saved_pct": saved_pct,
            },
            "delta": {
                "vs_cons_maj": delta_vs_cons,
                "vs_mean_avg": delta_vs_mean,
            },
        },
        "docs": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class AverageExperimentSummaryTest(unittest.TestCase):
    def test_discover_summary_paths_filters_by_experiment_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wanted = root / "run-a" / "experiments" / "prefix50_top50" / "summary.json"
            other = root / "run-b" / "experiments" / "prefix30_top50" / "summary.json"
            skipped = (
                root
                / "experiments_aggregated"
                / "prefix50_top50"
                / "summary.json"
            )
            write_experiment_summary(
                wanted,
                run_dir="/tmp/run-a",
                output_dir="/tmp/run-a/experiments/prefix50_top50",
            )
            write_experiment_summary(
                other,
                run_dir="/tmp/run-b",
                output_dir="/tmp/run-b/experiments/prefix30_top50",
            )
            write_experiment_summary(
                skipped,
                run_dir="/tmp/run-c",
                output_dir="/tmp/run-c/experiments/prefix50_top50",
            )

            paths = discover_summary_paths(root, "prefix50_top50", "experiments_aggregated")

            self.assertEqual(paths, [wanted])

    def test_build_output_averages_metrics_cost_and_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path_a = root / "run-a" / "experiments" / "prefix50_top50" / "summary.json"
            path_b = root / "run-b" / "experiments" / "prefix50_top50" / "summary.json"
            aggregate_dir = root / "experiments_aggregated" / "prefix50_top50"
            write_experiment_summary(
                path_a,
                run_dir="/tmp/run-a",
                output_dir="/tmp/run-a/experiments/prefix50_top50",
                think_maj=0.8,
                cons_maj=0.9,
                mean_avg=0.7,
                total_full_tokens=900.0,
                total_think_tokens=450.0,
                saved_tokens=450.0,
                saved_pct=0.5,
                delta_vs_cons=-0.1,
                delta_vs_mean=0.1,
            )
            write_experiment_summary(
                path_b,
                run_dir="/tmp/run-b",
                output_dir="/tmp/run-b/experiments/prefix50_top50",
                think_maj=0.6,
                cons_maj=0.8,
                mean_avg=0.5,
                total_full_tokens=1200.0,
                total_think_tokens=720.0,
                saved_tokens=480.0,
                saved_pct=0.4,
                delta_vs_cons=-0.2,
                delta_vs_mean=0.1,
            )

            payloads = [
                json.loads(path_a.read_text(encoding="utf-8")),
                json.loads(path_b.read_text(encoding="utf-8")),
            ]
            output = build_output(
                experiment_name="prefix50_top50",
                input_root=root,
                aggregate_dir=aggregate_dir,
                sources=[path_a, path_b],
                payloads=payloads,
            )

            self.assertEqual(output["experiment"], "prefix50_top50")
            self.assertEqual(output["source_count"], 2)
            self.assertAlmostEqual(output["metrics_mean"]["think_maj@24"], 0.7)
            self.assertAlmostEqual(output["metrics_mean"]["cons_maj@48"], 0.85)
            self.assertAlmostEqual(output["cost_mean"]["saved_pct"], 0.45)
            self.assertAlmostEqual(output["delta_mean"]["vs_cons_maj"], -0.15)
            self.assertAlmostEqual(
                output["metrics_stddev"]["think_maj@24"],
                0.14142135623730956,
            )
            self.assertAlmostEqual(
                output["cost_stddev"]["total_full_tokens"],
                212.13203435596427,
            )
            self.assertAlmostEqual(output["delta_stddev"]["vs_mean_avg"], 0.0)
            self.assertNotIn("num_docs", output["metrics_stddev"])
            self.assertEqual(
                [entry["run_dir"] for entry in output["aggregation"]["source_summaries"]],
                ["/tmp/run-a", "/tmp/run-b"],
            )

    def test_build_output_rejects_mismatched_metric_keys(self):
        payload_a = {
            "run_dir": "/tmp/run-a",
            "output_dir": "/tmp/run-a/experiments/prefix50_top50",
            "task": "task",
            "model": "model",
            "repeats": 48,
            "selected_count": 24,
            "prefix_len": 50,
            "g": 0.5,
            "rho": 0.85,
            "summary": {
                "metrics": {"think_maj@24": 0.5, "num_docs": 30},
                "cost": {"saved_pct": 0.5},
                "delta": {"vs_cons_maj": -0.1},
            },
        }
        payload_b = {
            "run_dir": "/tmp/run-b",
            "output_dir": "/tmp/run-b/experiments/prefix50_top50",
            "task": "task",
            "model": "model",
            "repeats": 48,
            "selected_count": 24,
            "prefix_len": 50,
            "g": 0.5,
            "rho": 0.85,
            "summary": {
                "metrics": {"cons_maj@48": 0.5, "num_docs": 30},
                "cost": {"saved_pct": 0.5},
                "delta": {"vs_cons_maj": -0.1},
            },
        }

        with self.assertRaisesRegex(ValueError, "metric keys"):
            build_output(
                experiment_name="prefix50_top50",
                input_root=Path("/tmp"),
                aggregate_dir=Path("/tmp/experiments_aggregated/prefix50_top50"),
                sources=[Path("/tmp/a"), Path("/tmp/b")],
                payloads=[payload_a, payload_b],
            )


if __name__ == "__main__":
    unittest.main()
