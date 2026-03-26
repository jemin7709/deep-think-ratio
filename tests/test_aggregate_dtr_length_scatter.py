import json
import tempfile
import unittest
from pathlib import Path

from src.aggregation.aggregate_dtr_length_scatter import (
    DEFAULT_AGGREGATE_DIR_NAME,
    OUTPUT_JSON_FILENAME_AGGREGATED,
    OUTPUT_PLOT_FILENAME,
    aggregate_points,
    build_title,
    discover_summary_paths,
    load_source_summaries,
    write_aggregated_json,
)


def write_summary(
    path: Path,
    *,
    run_dir: str,
    model: str = "openai/gpt-oss-120b",
    points: list[dict] | None = None,
) -> None:
    payload = {
        "run_dir": run_dir,
        "task": "aime24_custom",
        "model": model,
        "dtr_path": f"{run_dir}/dtr/dtr_g0.5_rho0.85.json",
        "results_path": f"{run_dir}/results_2026-03-22.json",
        "num_sequences": 2,
        "pearson_r": 1.0,
        "dtr_min": 0.1,
        "dtr_max": 0.3,
        "length_min": 10,
        "length_max": 20,
        "length_mean": 15.0,
        "points": points
        or [
            {
                "doc_id": 0,
                "repeat_index": 0,
                "dtr": 0.1,
                "response_length": 10,
                "is_correct": True,
            },
            {
                "doc_id": 0,
                "repeat_index": 1,
                "dtr": 0.3,
                "response_length": 20,
                "is_correct": False,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class AggregateDtrLengthScatterTest(unittest.TestCase):
    def test_discover_summary_paths_recurses_and_skips_aggregate_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_summary = (
                root
                / "aime24_custom"
                / "model"
                / "0"
                / "stamp"
                / "dtr_length_scatter"
                / "dtr_length_scatter.json"
            )
            run_summary.parent.mkdir(parents=True)
            write_summary(run_summary, run_dir="/tmp/run-0")

            skipped_summary = (
                root / DEFAULT_AGGREGATE_DIR_NAME / "dtr_length_scatter.json"
            )
            skipped_summary.parent.mkdir()
            write_summary(skipped_summary, run_dir="/tmp/aggregate")

            paths = discover_summary_paths(root, DEFAULT_AGGREGATE_DIR_NAME)

            self.assertEqual(paths, [run_summary])

    def test_aggregate_points_concatenates_all_source_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path_a = root / "run_a.json"
            path_b = root / "run_b.json"
            write_summary(path_a, run_dir="/tmp/run-a")
            write_summary(
                path_b,
                run_dir="/tmp/run-b",
                points=[
                    {
                        "doc_id": 1,
                        "repeat_index": 0,
                        "dtr": 0.2,
                        "response_length": 12,
                        "is_correct": True,
                    },
                    {
                        "doc_id": 1,
                        "repeat_index": 1,
                        "dtr": 0.4,
                        "response_length": 24,
                        "is_correct": False,
                    },
                ],
            )

            summaries = load_source_summaries([path_a, path_b])
            points = aggregate_points(summaries)

            self.assertEqual(len(points), 4)
            self.assertEqual(points[0].dtr, 0.1)
            self.assertEqual(points[-1].run_dir, "/tmp/run-b")
            self.assertEqual(
                [getattr(point, "is_correct") for point in points],
                [True, True, False, False],
            )

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
            aggregate_dir = root / DEFAULT_AGGREGATE_DIR_NAME
            aggregate_dir.mkdir()
            summary_path = root / "run.json"
            write_summary(summary_path, run_dir="/tmp/run")
            summaries = load_source_summaries([summary_path])
            points = aggregate_points(summaries)

            output_path = write_aggregated_json(
                input_root=root,
                aggregate_dir=aggregate_dir,
                summaries=summaries,
                points=points,
                plot_path=aggregate_dir / OUTPUT_PLOT_FILENAME,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task"], "aime24_custom")
            self.assertEqual(payload["source_count"], 1)
            self.assertEqual(payload["run_dirs"], ["/tmp/run"])
            self.assertEqual(payload["num_sequences"], 2)
            self.assertEqual(output_path.name, OUTPUT_JSON_FILENAME_AGGREGATED)
            self.assertEqual(
                payload["plot_path"], str(aggregate_dir / OUTPUT_PLOT_FILENAME)
            )
            self.assertIn("is_correct", payload["points"][0])
            self.assertEqual(
                [point["is_correct"] for point in payload["points"]], [True, False]
            )

    def test_build_title_mentions_runs_and_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.json"
            write_summary(path, run_dir="/tmp/run")
            summaries = load_source_summaries([path])

            self.assertEqual(
                build_title(summaries, 14400, None),
                "aime24_custom | openai/gpt-oss-120b | DTR vs Response Length (1 runs, 14400 points)",
            )


if __name__ == "__main__":
    unittest.main()
