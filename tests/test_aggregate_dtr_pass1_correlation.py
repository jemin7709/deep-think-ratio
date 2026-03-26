import json
import tempfile
import unittest
from pathlib import Path

from src.aggregation.aggregate_dtr_pass1_correlation import (
    aggregate_bins,
    aggregated_json_name,
    build_plot_bins,
    discover_summary_paths,
    load_source_summaries,
    plot_filename,
    plot_summary_json_name,
    write_aggregated_json,
)


def write_summary(
    path: Path,
    *,
    run_dir: str,
    model: str = "openai/gpt-oss-120b",
    dtr_scope: str = "full",
    prefix_len: int | None = None,
    start_token: int | None = None,
    end_token: int | None = None,
    bins: list[dict] | None = None,
) -> None:
    payload = {
        "run_dir": run_dir,
        "task": "aime24_custom",
        "model": model,
        "dtr_scope": dtr_scope,
        "prefix_len": prefix_len,
        "start_token": start_token,
        "end_token": end_token,
        "dtr_path": f"{run_dir}/dtr/dtr_g0.5_rho0.85.json",
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
            run_summary = (
                root
                / "aime24_custom"
                / "model"
                / "0"
                / "stamp"
                / "dtr_pass1_correlation"
                / "dtr_pass1_correlation_bins2.json"
            )
            run_summary.parent.mkdir(parents=True)
            write_summary(run_summary, run_dir="/tmp/run-0")

            skipped_summary = (
                root
                / "dtr_pass1_correlation_aggregated"
                / "dtr_pass1_correlation_bins2.json"
            )
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

            summaries = load_source_summaries([path_a, path_b], prefix_len=None)
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
                load_source_summaries([path_a, path_b], prefix_len=None)

    def test_load_source_summaries_filters_by_prefix_len(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            full_path = root / "full.json"
            prefix_path = root / "prefix.json"
            write_summary(full_path, run_dir="/tmp/run-full")
            write_summary(
                prefix_path,
                run_dir="/tmp/run-prefix",
                dtr_scope="prefix",
                prefix_len=2,
            )

            summaries = load_source_summaries([full_path, prefix_path], prefix_len=2)

            self.assertEqual([summary.run_dir for summary in summaries], [Path("/tmp/run-prefix")])
            self.assertEqual(summaries[0].dtr_scope, "prefix")
            self.assertEqual(summaries[0].prefix_len, 2)

    def test_load_source_summaries_filters_by_token_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            full_path = root / "full.json"
            window_path = root / "window.json"
            write_summary(full_path, run_dir="/tmp/run-full")
            write_summary(
                window_path,
                run_dir="/tmp/run-window",
                dtr_scope="window",
                start_token=50,
                end_token=None,
            )

            summaries = load_source_summaries(
                [full_path, window_path],
                start_token=50,
                end_token=None,
            )

            self.assertEqual([summary.run_dir for summary in summaries], [Path("/tmp/run-window")])
            self.assertEqual(summaries[0].dtr_scope, "window")
            self.assertEqual(summaries[0].start_token, 50)
            self.assertIsNone(summaries[0].end_token)

    def test_load_source_summaries_filters_by_num_bins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bins2_path = root / "bins2.json"
            bins3_path = root / "bins3.json"
            write_summary(bins2_path, run_dir="/tmp/run-bins2")
            write_summary(bins3_path, run_dir="/tmp/run-bins3")
            payload = json.loads(bins3_path.read_text(encoding="utf-8"))
            payload["num_bins"] = 3
            payload["bins"].append(
                {
                    "bin_index": 3,
                    "count": 2,
                    "rank_start": 5,
                    "rank_end": 6,
                    "dtr_min": 0.9,
                    "dtr_max": 1.0,
                    "mean_dtr": 0.95,
                    "pass_at_1": 1.0,
                }
            )
            bins3_path.write_text(json.dumps(payload), encoding="utf-8")

            summaries = load_source_summaries(
                [bins2_path, bins3_path],
                num_bins=2,
            )

            self.assertEqual([summary.run_dir for summary in summaries], [Path("/tmp/run-bins2")])

    def test_write_aggregated_json_emits_expected_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            aggregate_dir = root / "dtr_pass1_correlation_aggregated"
            aggregate_dir.mkdir()
            summary_path = root / "run.json"
            write_summary(summary_path, run_dir="/tmp/run")
            summaries = load_source_summaries([summary_path], prefix_len=None)
            aggregated = aggregate_bins(summaries)

            output_path = write_aggregated_json(
                input_root=root,
                aggregate_dir=aggregate_dir,
                summaries=summaries,
                aggregated_bins=aggregated,
                plot_path=aggregate_dir / plot_filename(2),
                plot_summary_path=aggregate_dir / plot_summary_json_name(2),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task"], "aime24_custom")
            self.assertEqual(payload["dtr_scope"], "full")
            self.assertIsNone(payload["prefix_len"])
            self.assertIsNone(payload["start_token"])
            self.assertIsNone(payload["end_token"])
            self.assertEqual(payload["source_count"], 1)
            self.assertEqual(payload["run_dirs"], ["/tmp/run"])
            self.assertEqual(output_path.name, aggregated_json_name(2))
            self.assertEqual(
                payload["plot_path"], str(aggregate_dir / plot_filename(2))
            )

    def test_write_aggregated_json_uses_prefix_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            aggregate_dir = root / "dtr_pass1_correlation_aggregated"
            aggregate_dir.mkdir()
            summary_path = root / "run_prefix.json"
            write_summary(
                summary_path,
                run_dir="/tmp/run-prefix",
                dtr_scope="prefix",
                prefix_len=3,
            )
            summaries = load_source_summaries([summary_path], prefix_len=3)
            aggregated = aggregate_bins(summaries)

            output_path = write_aggregated_json(
                input_root=root,
                aggregate_dir=aggregate_dir,
                summaries=summaries,
                aggregated_bins=aggregated,
                plot_path=aggregate_dir / plot_filename(2, prefix_len=3),
                plot_summary_path=aggregate_dir / plot_summary_json_name(2, prefix_len=3),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dtr_scope"], "prefix")
            self.assertEqual(payload["prefix_len"], 3)
            self.assertEqual(output_path.name, aggregated_json_name(2, prefix_len=3))
            self.assertEqual(
                payload["plot_path"],
                str(aggregate_dir / plot_filename(2, prefix_len=3)),
            )


if __name__ == "__main__":
    unittest.main()
