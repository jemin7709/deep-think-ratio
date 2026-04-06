import json
import tempfile
import unittest
from pathlib import Path

from src.aggregation.dtr_problem_difficulty import (
    OUTPUT_BUCKET_PLOT_FILENAME,
    OUTPUT_JSON_FILENAME,
    OUTPUT_PLOT_FILENAME,
    BucketSummary,
    ProblemDifficultyRow,
    build_bucket_summary,
    build_bucket_summaries,
    build_title,
    default_output_dir,
    difficulty_bucket,
    load_dtr_records_by_key,
    load_problem_rows,
    resolve_paths,
    resolve_samples_path,
    spearman_r,
    write_summary_json,
)
from src.dtr.jsd_utils import dtr_results_path
from src.plot.dtr_problem_difficulty import (
    difficulty_bands,
    difficulty_note_text,
    difficulty_scatter_x_label,
    plot_bucket_summary_to_png,
    plot_scatter_to_png,
)
from tasks.aime24.utils import build_gpt_oss_reasoning_tags


def gpt_oss_completion(reasoning: str, content: str) -> str:
    return (
        "<|start|>assistant<|channel|>analysis<|message|>"
        f"{reasoning}"
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        f"{content}"
        "<|end|>"
    )


class DtrProblemDifficultyTest(unittest.TestCase):
    def test_load_dtr_records_by_key_requires_num_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dtr_path = Path(tmpdir) / "dtr.json"
            dtr_path.write_text(
                json.dumps([{"doc_id": 0, "repeat_index": 0, "dtr": 0.2}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing num_tokens"):
                load_dtr_records_by_key(dtr_path)

    def test_load_dtr_records_by_key_parses_doc_repeat_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dtr_path = Path(tmpdir) / "dtr.json"
            dtr_path.write_text(
                json.dumps(
                    [
                        {
                            "doc_id": 3,
                            "repeat_index": 1,
                            "dtr": 0.7,
                            "num_tokens": 12,
                        },
                        {
                            "doc_id": 2,
                            "repeat_index": 0,
                            "dtr": 0.2,
                            "num_tokens": 5,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            records = load_dtr_records_by_key(dtr_path)
            self.assertEqual(
                sorted(records),
                [(2, 0), (3, 1)],
            )
            self.assertEqual(records[(2, 0)].dtr, 0.2)
            self.assertEqual(records[(2, 0)].response_length, 5)

    def test_difficulty_bucket_thresholds(self):
        self.assertEqual(difficulty_bucket(0.75), "easy")
        self.assertEqual(difficulty_bucket(0.76), "easy")
        self.assertEqual(difficulty_bucket(0.5), "medium")
        self.assertEqual(difficulty_bucket(0.25), "hard")
        self.assertEqual(difficulty_bucket(0.24), "hard")

    def test_load_problem_rows_scores_each_repeat_and_aggregates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            samples_path = (
                run_dir / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
            )
            samples_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "doc_id": row_doc_id,
                            "target": target,
                            "resps": [[
                                gpt_oss_completion("r1", first),
                                gpt_oss_completion("r2", second),
                            ]],
                        }
                    )
                    + "\n"
                    for row_doc_id, target, first, second in [
                        (0, "42", "\\boxed{42}", "\\boxed{0}"),
                        (1, "7", "\\boxed{7}", "\\boxed{6}"),
                    ]
                ),
                encoding="utf-8",
            )
            dtr_by_key = {
                (0, 0): (0.1, 10),
                (0, 1): (0.3, 20),
                (1, 0): (0.6, 30),
                (1, 1): (0.8, 40),
            }

            sequence_rows = load_problem_rows(
                dtr_by_key,
                samples_path,
                reasoning_tags=build_gpt_oss_reasoning_tags(),
            )

            self.assertEqual(
                [
                    (row.doc_id, row.repeat_index, row.dtr, row.correct)
                    for row in sequence_rows
                ],
                [
                    (0, 0, 0.1, True),
                    (0, 1, 0.3, False),
                    (1, 0, 0.6, True),
                    (1, 1, 0.8, False),
                ],
            )

            problem_rows = [
                ProblemDifficultyRow(
                    doc_id=0,
                    accuracy=0.5,
                    difficulty_score=0.5,
                    mean_dtr=0.2,
                    mean_response_length=15.0,
                    difficulty_bucket="medium",
                    correct_count=1,
                    repeat_count=2,
                ),
                ProblemDifficultyRow(
                    doc_id=1,
                    accuracy=0.5,
                    difficulty_score=0.5,
                    mean_dtr=0.7,
                    mean_response_length=35.0,
                    difficulty_bucket="medium",
                    correct_count=1,
                    repeat_count=2,
                ),
            ]

            summary = build_bucket_summary("medium", problem_rows)
            self.assertEqual(summary.bucket, "medium")
            self.assertEqual(summary.num_problems, 2)
            self.assertAlmostEqual(summary.mean_accuracy, 0.5)
            self.assertAlmostEqual(summary.mean_dtr, 0.45)
            self.assertAlmostEqual(summary.mean_response_length, 25.0)

    def test_build_bucket_summaries_covers_all(self):
        rows = [
            ProblemDifficultyRow(
                doc_id=0,
                accuracy=1.0,
                difficulty_score=0.0,
                mean_dtr=0.5,
                mean_response_length=12.0,
                difficulty_bucket="easy",
                correct_count=4,
                repeat_count=4,
            ),
            ProblemDifficultyRow(
                doc_id=1,
                accuracy=0.5,
                difficulty_score=0.5,
                mean_dtr=0.6,
                mean_response_length=10.0,
                difficulty_bucket="medium",
                correct_count=2,
                repeat_count=4,
            ),
            ProblemDifficultyRow(
                doc_id=2,
                accuracy=0.0,
                difficulty_score=1.0,
                mean_dtr=0.2,
                mean_response_length=8.0,
                difficulty_bucket="hard",
                correct_count=0,
                repeat_count=4,
            ),
        ]
        bucket_summaries = build_bucket_summaries(rows)
        self.assertEqual(
            [entry.bucket for entry in bucket_summaries],
            ["easy", "medium", "hard"],
        )
        self.assertEqual([entry.num_problems for entry in bucket_summaries], [1, 1, 1])

    def test_plot_helpers_describe_difficulty_score_and_bands(self):
        self.assertEqual(difficulty_note_text(0.5), "Spearman ρ = 0.500")
        self.assertEqual(
            difficulty_scatter_x_label(),
            "Difficulty Score (= 1 - accuracy, 0=easiest, 1=hardest)",
        )
        self.assertEqual(
            [(band.bucket, band.start, band.end) for band in difficulty_bands()],
            [
                ("easy", 0.0, 0.25),
                ("medium", 0.25, 0.75),
                ("hard", 0.75, 1.0),
            ],
        )

    def test_resolve_samples_path_matches_results_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            results_path = run_dir / "results_2026-03-22T00-00-00.json"
            expected = (
                run_dir
                / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
            )
            (run_dir / "samples_aime24_custom_2026-03-23T00-00-00.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )
            expected.write_text("{}\n", encoding="utf-8")

            self.assertEqual(
                resolve_samples_path(
                    run_dir=run_dir,
                    task_name="aime24_custom",
                    results_path=results_path,
                    samples_path=None,
                ),
                expected,
            )

    def test_resolve_paths_uses_default_output_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            results_path = run_dir / "results_2026-03-22T00-00-00.json"
            samples_path = (
                run_dir / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
            )
            results_path.write_text(
                json.dumps({"results": {"aime24_custom": {}}}),
                encoding="utf-8",
            )
            samples_path.write_text("{}\n", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "run_dir": run_dir,
                    "dtr_path": None,
                    "results_path": None,
                    "samples_path": None,
                    "output_plot": None,
                    "output_bucket_plot": None,
                    "output_json": None,
                },
            )()

            dtr_path, _results_path, _samples_path, output_plot, output_bucket_plot, output_json = (
                resolve_paths(args)
            )

            self.assertEqual(dtr_path, dtr_results_path(run_dir))
            self.assertEqual(
                output_plot, default_output_dir(run_dir) / OUTPUT_PLOT_FILENAME
            )
            self.assertEqual(
                output_bucket_plot,
                default_output_dir(run_dir) / OUTPUT_BUCKET_PLOT_FILENAME,
            )
            self.assertEqual(output_json, default_output_dir(run_dir) / OUTPUT_JSON_FILENAME)

    def test_spearman_r_returns_zero_when_no_variance(self):
        self.assertEqual(spearman_r([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]), 0.0)

    def test_write_summary_json_emits_required_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            results_path = run_dir / "results_2026-03-22T00-00-00.json"
            samples_path = run_dir / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
            dtr_path = dtr_results_path(run_dir)
            output_path = default_output_dir(run_dir) / OUTPUT_JSON_FILENAME

            results_path.write_text(
                json.dumps({"config": {"model_args": {"pretrained": "openai/gpt-oss-120b"}}}),
                encoding="utf-8",
            )
            samples_path.write_text("{}\n", encoding="utf-8")
            dtr_path.parent.mkdir(parents=True, exist_ok=True)
            dtr_path.write_text(
                json.dumps(
                    [
                        {
                            "doc_id": 0,
                            "repeat_index": 0,
                            "dtr": 0.1,
                            "num_tokens": 10,
                        },
                        {
                            "doc_id": 0,
                            "repeat_index": 1,
                            "dtr": 0.2,
                            "num_tokens": 20,
                        },
                        {
                            "doc_id": 1,
                            "repeat_index": 0,
                            "dtr": 0.8,
                            "num_tokens": 40,
                        },
                        {
                            "doc_id": 1,
                            "repeat_index": 1,
                            "dtr": 0.9,
                            "num_tokens": 80,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            bucket_summaries = [
                BucketSummary(
                    bucket="easy",
                    num_problems=1,
                    mean_accuracy=1.0,
                    mean_dtr=0.15,
                    mean_response_length=15.0,
                ),
                BucketSummary(
                    bucket="medium",
                    num_problems=0,
                    mean_accuracy=0.0,
                    mean_dtr=0.0,
                    mean_response_length=0.0,
                ),
                BucketSummary(
                    bucket="hard",
                    num_problems=1,
                    mean_accuracy=0.0,
                    mean_dtr=0.85,
                    mean_response_length=60.0,
                ),
            ]
            problems = [
                ProblemDifficultyRow(
                    doc_id=0,
                    accuracy=1.0,
                    difficulty_score=0.0,
                    mean_dtr=0.15,
                    mean_response_length=15.0,
                    difficulty_bucket="easy",
                    correct_count=2,
                    repeat_count=2,
                ),
                ProblemDifficultyRow(
                    doc_id=1,
                    accuracy=0.0,
                    difficulty_score=1.0,
                    mean_dtr=0.85,
                    mean_response_length=60.0,
                    difficulty_bucket="hard",
                    correct_count=0,
                    repeat_count=2,
                ),
            ]

            write_summary_json(
                run_dir=run_dir,
                task_name="aime24_custom",
                model_name="openai/gpt-oss-120b",
                dtr_path=dtr_path,
                results_path=results_path,
                samples_path=samples_path,
                output_path=output_path,
                problems=problems,
                bucket_summaries=bucket_summaries,
                spearman=0.5,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task"], "aime24_custom")
            self.assertEqual(payload["model"], "openai/gpt-oss-120b")
            self.assertEqual(payload["num_problems"], 2)
            self.assertAlmostEqual(payload["spearman_r"], 0.5)
            self.assertEqual(payload["bucket_summaries"][0]["bucket"], "easy")
            self.assertEqual(len(payload["problems"]), 2)
            self.assertEqual(
                payload["problems"][0],
                {
                    "doc_id": 0,
                    "accuracy": 1.0,
                    "difficulty_score": 0.0,
                    "mean_dtr": 0.15,
                    "mean_response_length": 15.0,
                    "difficulty_bucket": "easy",
                    "correct_count": 2,
                    "repeat_count": 2,
                },
            )

    def test_title_and_plot_helpers_generate_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_scatter = Path(tmpdir) / "scatter.png"
            output_bucket = Path(tmpdir) / "bucket.png"
            run_dir = Path("/tmp/results/aime24/model/0/seed")
            points = [
                ProblemDifficultyRow(
                    doc_id=0,
                    accuracy=1.0,
                    difficulty_score=0.0,
                    mean_dtr=0.5,
                    mean_response_length=30.0,
                    difficulty_bucket="easy",
                    correct_count=2,
                    repeat_count=2,
                )
            ]
            buckets = [
                BucketSummary(
                    bucket="easy",
                    num_problems=1,
                    mean_accuracy=1.0,
                    mean_dtr=0.5,
                    mean_response_length=30.0,
                ),
                BucketSummary(
                    bucket="medium",
                    num_problems=0,
                    mean_accuracy=0.0,
                    mean_dtr=0.0,
                    mean_response_length=0.0,
                ),
                BucketSummary(
                    bucket="hard",
                    num_problems=0,
                    mean_accuracy=0.0,
                    mean_dtr=0.0,
                    mean_response_length=0.0,
                ),
            ]

            plot_scatter_to_png(
                points=points,
                spearman=1.0,
                output_path=output_scatter,
                title=build_title(run_dir, "aime24_custom", "openai/gpt-oss-120b", None),
            )
            plot_bucket_summary_to_png(
                bucket_summaries=buckets,
                output_path=output_bucket,
                title=build_title(run_dir, "aime24_custom", "openai/gpt-oss-120b", None),
            )

            self.assertTrue(output_scatter.is_file())
            self.assertTrue(output_bucket.is_file())


if __name__ == "__main__":
    unittest.main()
