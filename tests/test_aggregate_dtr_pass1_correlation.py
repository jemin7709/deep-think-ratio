import json
import tempfile
import unittest
from pathlib import Path

from src.aggregation.aggregate_dtr_pass1_correlation import (
    aggregate_bins,
    build_plot_bins,
    discover_summary_paths,
    load_source_summaries,
    write_aggregated_json,
)


def write_summary(path: Path, *, run_dir: str, model: str = "openai/gpt-oss-120b", bins: list[dict] | None = None) -> None:
    payload = {
        "run_dir": run_dir,
        "task": "aime24_custom",
        "model": model,
        "dtr_path": f"{run_dir}/dtr_results_from_jsd.json",
        "results_path": f"{run_dir}/results_2026-03-22.json",
        "samples_path": f"{run_dir}/samples_aime24_custom_2026-03-22.jsonl",
        "num_sequences": 4,
        "num_bins": 2,
        "pearson_r_binned": 1.0,
        "bins": bins
        or [
            {
                "bin_index": 1,
                "count": 2,
                "rank_start": 1,
                "rank_end": 2,
                "dtr_min": 0.1,
                "dtr_max": 0.3,
                "mean_dtr": 0.2,
                "pass_at_1": 0.5,
            },
            {
                "bin_index": 2,
                "count": 2,
                "rank_start": 3,
                "rank_end": 4,
                "dtr_min": 0.6,
                "dtr_max": 0.8,
                "mean_dtr": 0.7,
                "pass_at_1": 1.0,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class AggregateDtrPass1CorrelationTest(unittest.TestCase):
    def test_discover_summary_paths_recurses_and_skips_aggregate_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_summary = root / "aime24_custom" / "model" / "0" / "stamp" / "dtr_pass1_correlation.json"
            run_summary.parent.mkdir(parents=True)
            write_summary(run_summary, run_dir="/tmp/run-0")

            skipped_summary = root / "dtr_pass1_correlation_aggregated" / "dtr_pass1_correlation.json"
            skipped_summary.parent.mkdir()
            write_summary(skipped_summary, run_dir="/tmp/aggregate")

            paths = discover_summary_paths(root, "dtr_pass1_correlation_aggregated")
            self.assertEqual(paths, [run_summary])

    def test_aggregate_bins_averages_bin_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path_a = root / "run_a.json"
            path_b = root / "run_b.json"
            write_summary(path_a, run_dir="/tmp/run-a")
            write_summary(
                path_b,
                run_dir="/tmp/run-b",
                bins=[
                    {
                        "bin_index": 1,
                        "count": 4,
                        "rank_start": 1,
                        "rank_end": 4,
                        "dtr_min": 0.2,
                        "dtr_max": 0.4,
                        "mean_dtr": 0.3,
                        "pass_at_1": 0.25,
                    },
                    {
                        "bin_index": 2,
                        "count": 4,
                        "rank_start": 5,
                        "rank_end": 8,
                        "dtr_min": 0.7,
                        "dtr_max": 0.9,
                        "mean_dtr": 0.8,
                        "pass_at_1": 0.75,
                    },
                ],
            )

            summaries = load_source_summaries([path_a, path_b])
            aggregated = aggregate_bins(summaries)
            plot_bins = build_plot_bins(aggregated)

            self.assertEqual(aggregated[0].source_count, 2)
            self.assertAlmostEqual(aggregated[0].mean_source_bin_size, 3.0)
            self.assertAlmostEqual(aggregated[0].mean_dtr, 0.25)
            self.assertAlmostEqual(aggregated[0].pass_at_1, 0.375)
            self.assertEqual(plot_bins[0].count, 3)

    def test_load_source_summaries_rejects_mixed_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path_a = root / "run_a.json"
            path_b = root / "run_b.json"
            write_summary(path_a, run_dir="/tmp/run-a", model="model-a")
            write_summary(path_b, run_dir="/tmp/run-b", model="model-b")

            with self.assertRaisesRegex(ValueError, "task and model"):
                load_source_summaries([path_a, path_b])

    def test_write_aggregated_json_emits_expected_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            aggregate_dir = root / "dtr_pass1_correlation_aggregated"
            aggregate_dir.mkdir()
            summary_path = root / "run.json"
            write_summary(summary_path, run_dir="/tmp/run")
            summaries = load_source_summaries([summary_path])
            aggregated = aggregate_bins(summaries)

            output_path = write_aggregated_json(
                input_root=root,
                aggregate_dir=aggregate_dir,
                summaries=summaries,
                aggregated_bins=aggregated,
                plot_path=aggregate_dir / "dtr_pass1_correlation.png",
                plot_summary_path=aggregate_dir / "plot_dtr_pass1_correlation_summary.json",
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task"], "aime24_custom")
            self.assertEqual(payload["source_count"], 1)
            self.assertEqual(payload["run_dirs"], ["/tmp/run"])
            self.assertEqual(payload["plot_path"], str(aggregate_dir / "dtr_pass1_correlation.png"))


if __name__ == "__main__":
    unittest.main()
