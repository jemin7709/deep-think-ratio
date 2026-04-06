import json
import tempfile
import unittest
from pathlib import Path

from src.aggregation.aggregate_dtr_problem_difficulty import (
    OUTPUT_JSON_FILENAME_AGGREGATED,
    OUTPUT_PLOT_FILENAME,
    OUTPUT_BUCKET_PLOT_FILENAME,
    DEFAULT_AGGREGATE_DIR_NAME,
    build_title,
    discover_summary_paths,
    load_source_summaries,
    run_level_summary_payload,
    write_aggregated_json,
)


def write_summary(
    path: Path,
    *,
    run_dir: str,
    model: str = "openai/gpt-oss-120b",
    spearman_r: float,
    num_problems: int,
    bucket_summaries: list[dict],
    problems: list[dict] | None = None,
) -> None:
    payload = {
        "run_dir": run_dir,
        "task": "aime24_custom",
        "model": model,
        "dtr_path": f"{run_dir}/dtr/dtr_g0.5_rho0.85.json",
        "results_path": f"{run_dir}/results_2026-03-22.json",
        "samples_path": f"{run_dir}/samples_aime24_custom_2026-03-22.jsonl",
        "spearman_r": spearman_r,
        "num_problems": num_problems,
        "mean_accuracy": 0.6 if num_problems else 0.0,
        "mean_dtr": 0.4 if num_problems else 0.0,
        "mean_response_length": 24.0 if num_problems else 0.0,
        "bucket_summaries": bucket_summaries,
        "problems": problems
        or [
            {
                "doc_id": 0,
                "accuracy": 1.0,
                "difficulty_score": 0.0,
                "mean_dtr": 0.5,
                "mean_response_length": 24.0,
                "difficulty_bucket": "easy",
                "correct_count": 4,
                "repeat_count": 4,
            },
            {
                "doc_id": 1,
                "accuracy": 0.0,
                "difficulty_score": 1.0,
                "mean_dtr": 0.3,
                "mean_response_length": 24.0,
                "difficulty_bucket": "hard",
                "correct_count": 0,
                "repeat_count": 4,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class AggregateDtrProblemDifficultyTest(unittest.TestCase):
    def test_discover_summary_paths_recurses_and_skips_aggregate_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_summary = (
                root
                / "aime24_custom"
                / "model"
                / "0"
                / "stamp"
                / "dtr_problem_difficulty"
                / "dtr_problem_difficulty.json"
            )
            run_summary.parent.mkdir(parents=True)
            write_summary(
                run_summary,
                run_dir="/tmp/run-a",
                spearman_r=0.1,
                num_problems=2,
                bucket_summaries=[],
            )

            skipped = (
                root / DEFAULT_AGGREGATE_DIR_NAME / "dtr_problem_difficulty.json"
            )
            skipped.parent.mkdir(parents=True)
            write_summary(
                skipped,
                run_dir="/tmp/run-b",
                spearman_r=0.2,
                num_problems=2,
                bucket_summaries=[],
            )

            paths = discover_summary_paths(root, DEFAULT_AGGREGATE_DIR_NAME)
            self.assertEqual(paths, [run_summary])

    def test_load_source_summaries_rejects_mixed_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path_a = root / "a.json"
            path_b = root / "b.json"
            write_summary(
                path_a,
                run_dir="/tmp/run-a",
                model="model-a",
                spearman_r=0.1,
                num_problems=3,
                bucket_summaries=[],
            )
            write_summary(
                path_b,
                run_dir="/tmp/run-b",
                model="model-b",
                spearman_r=0.2,
                num_problems=3,
                bucket_summaries=[],
            )

            with self.assertRaisesRegex(ValueError, "task and model"):
                load_source_summaries([path_a, path_b])

    def test_load_source_summary_backfills_missing_bucket_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.json"
            write_summary(
                path,
                run_dir="/tmp/run",
                spearman_r=0.2,
                num_problems=2,
                bucket_summaries=[],
                problems=[
                    {
                        "doc_id": 0,
                        "accuracy": 1.0,
                        "difficulty_score": 0.0,
                        "mean_dtr": 0.5,
                        "mean_response_length": 24.0,
                        "difficulty_bucket": "easy",
                        "correct_count": 4,
                        "repeat_count": 4,
                    },
                    {
                        "doc_id": 1,
                        "accuracy": 0.0,
                        "difficulty_score": 1.0,
                        "mean_dtr": 0.3,
                        "mean_response_length": 24.0,
                        "difficulty_bucket": "hard",
                        "correct_count": 0,
                        "repeat_count": 4,
                    },
                ],
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["bucket_summaries"]
            path.write_text(json.dumps(payload), encoding="utf-8")

            summaries = load_source_summaries([path])

            self.assertEqual(
                [entry.bucket for entry in summaries[0].bucket_summaries],
                ["easy", "medium", "hard"],
            )
            self.assertEqual(summaries[0].bucket_summaries[0].num_problems, 1)
            self.assertEqual(summaries[0].bucket_summaries[1].num_problems, 0)

    def test_load_source_summary_backfills_missing_scalar_fields_from_problems(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.json"
            write_summary(
                path,
                run_dir="/tmp/run",
                spearman_r=0.2,
                num_problems=2,
                bucket_summaries=[],
                problems=[
                    {
                        "doc_id": 0,
                        "accuracy": 1.0,
                        "difficulty_score": 0.0,
                        "mean_dtr": 0.5,
                        "mean_response_length": 20.0,
                        "difficulty_bucket": "easy",
                        "correct_count": 4,
                        "repeat_count": 4,
                    },
                    {
                        "doc_id": 1,
                        "accuracy": 0.0,
                        "difficulty_score": 1.0,
                        "mean_dtr": 0.3,
                        "mean_response_length": 40.0,
                        "difficulty_bucket": "hard",
                        "correct_count": 0,
                        "repeat_count": 4,
                    },
                ],
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key in (
                "bucket_summaries",
                "spearman_r",
                "num_problems",
                "mean_accuracy",
                "mean_dtr",
                "mean_response_length",
            ):
                del payload[key]
            path.write_text(json.dumps(payload), encoding="utf-8")

            summaries = load_source_summaries([path])

            self.assertEqual(summaries[0].num_problems, 2)
            self.assertAlmostEqual(summaries[0].mean_accuracy, 0.5)
            self.assertAlmostEqual(summaries[0].mean_dtr, 0.4)
            self.assertAlmostEqual(summaries[0].mean_response_length, 30.0)
            self.assertAlmostEqual(summaries[0].spearman_r, -1.0)

    def test_aggregate_bucket_summaries_calculates_seed_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path_a = root / "a.json"
            path_b = root / "b.json"

            write_summary(
                path_a,
                run_dir="/tmp/run-a",
                spearman_r=0.1,
                num_problems=4,
                bucket_summaries=[
                    {
                        "bucket": "easy",
                        "num_problems": 1,
                        "mean_accuracy": 0.8,
                        "mean_dtr": 0.2,
                        "mean_response_length": 24.0,
                    },
                    {
                        "bucket": "medium",
                        "num_problems": 2,
                        "mean_accuracy": 0.5,
                        "mean_dtr": 0.4,
                        "mean_response_length": 28.0,
                    },
                    {
                        "bucket": "hard",
                        "num_problems": 1,
                        "mean_accuracy": 0.0,
                        "mean_dtr": 0.7,
                        "mean_response_length": 32.0,
                    },
                ],
            )
            write_summary(
                path_b,
                run_dir="/tmp/run-b",
                spearman_r=0.3,
                num_problems=2,
                bucket_summaries=[
                    {
                        "bucket": "easy",
                        "num_problems": 0,
                        "mean_accuracy": 0.0,
                        "mean_dtr": 0.0,
                        "mean_response_length": 0.0,
                    },
                    {
                        "bucket": "medium",
                        "num_problems": 1,
                        "mean_accuracy": 0.7,
                        "mean_dtr": 0.6,
                        "mean_response_length": 24.0,
                    },
                    {
                        "bucket": "hard",
                        "num_problems": 1,
                        "mean_accuracy": 0.1,
                        "mean_dtr": 0.9,
                        "mean_response_length": 40.0,
                    },
                ],
            )

            summaries = load_source_summaries([path_a, path_b])
            run_summary = run_level_summary_payload(summaries)

            self.assertEqual(run_summary.source_count, 2)
            self.assertEqual(
                list(run_summary.bucket_summaries.keys()),
                ["easy", "medium", "hard"],
            )
            self.assertEqual(run_summary.overall_num_problems["mean"], 3.0)
            self.assertAlmostEqual(
                run_summary.overall_num_problems["std"],
                1.4142135623730951,
            )
            self.assertEqual(
                run_summary.bucket_summaries["easy"].num_problems_mean,
                0.5,
            )
            self.assertAlmostEqual(
                run_summary.bucket_summaries["easy"].mean_accuracy_mean,
                0.4,
            )
            self.assertEqual(len(run_summary.scatter_points), 4)
            self.assertEqual(
                [point.difficulty_bucket for point in run_summary.scatter_points],
                ["easy", "hard", "easy", "hard"],
            )

    def test_build_title_mentions_runs_and_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.json"
            write_summary(
                path,
                run_dir="/tmp/run",
                spearman_r=0.2,
                num_problems=10,
                bucket_summaries=[],
            )
            summaries = load_source_summaries([path])
            title = build_title(summaries, 7, None)

            self.assertEqual(
                title,
                "aime24_custom | openai/gpt-oss-120b | Problem Difficulty (1 runs, 7 problems)",
            )

    def test_write_aggregated_json_includes_bucket_summaries_and_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            aggregate_dir = root / DEFAULT_AGGREGATE_DIR_NAME
            aggregate_dir.mkdir()
            path = root / "run.json"
            write_summary(
                path,
                run_dir="/tmp/run",
                spearman_r=0.2,
                num_problems=4,
                bucket_summaries=[
                    {
                        "bucket": "easy",
                        "num_problems": 4,
                        "mean_accuracy": 0.7,
                        "mean_dtr": 0.2,
                        "mean_response_length": 20.0,
                    },
                    {
                        "bucket": "medium",
                        "num_problems": 0,
                        "mean_accuracy": 0.0,
                        "mean_dtr": 0.0,
                        "mean_response_length": 0.0,
                    },
                    {
                        "bucket": "hard",
                        "num_problems": 0,
                        "mean_accuracy": 0.0,
                        "mean_dtr": 0.0,
                        "mean_response_length": 0.0,
                    },
                ],
            )
            summaries = load_source_summaries([path])
            output_path = write_aggregated_json(
                input_root=root,
                aggregate_dir=aggregate_dir,
                summaries=summaries,
                run_summary=run_level_summary_payload(summaries),
                output_scatter_path=aggregate_dir / OUTPUT_PLOT_FILENAME,
                output_bucket_plot_path=aggregate_dir / OUTPUT_BUCKET_PLOT_FILENAME,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["source_count"], 1)
            self.assertEqual(payload["source_paths"], [str(path)])
            self.assertEqual(payload["run_dirs"], ["/tmp/run"])
            self.assertEqual(payload["spearman_r_mean"], 0.2)
            self.assertEqual(output_path.name, OUTPUT_JSON_FILENAME_AGGREGATED)
            self.assertEqual(
                payload["plot_path"],
                str(aggregate_dir / OUTPUT_PLOT_FILENAME),
            )
            self.assertEqual(
                payload["bucket_plot_path"],
                str(aggregate_dir / OUTPUT_BUCKET_PLOT_FILENAME),
            )
            self.assertEqual(len(payload["bucket_summaries"]), 3)
            self.assertEqual(payload["bucket_summaries"][0]["bucket"], "easy")


if __name__ == "__main__":
    unittest.main()
