import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.aggregation.dtr_length_scatter import (
    DEFAULT_LENGTH_MODE,
    DEFAULT_OUTPUT_DIR_NAME,
    REASONING_LENGTH_MODE,
    SequenceLengthPoint,
    build_title,
    default_output_dir,
    load_response_lengths_by_key,
    load_correctness_by_key,
    load_points,
    pearson_r,
    resolve_length_reasoning_tags,
    resolve_model_name,
    resolve_paths,
    resolve_samples_path,
    write_summary_json,
)
from src.dtr.jsd_utils import dtr_results_path
from tasks.aime24.utils import build_gpt_oss_reasoning_tags


def gpt_oss_completion(reasoning: str, content: str) -> str:
    return (
        "<|start|>assistant<|channel|>analysis<|message|>"
        f"{reasoning}"
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        f"{content}"
        "<|end|>"
    )


class PlotDtrLengthScatterTest(unittest.TestCase):
    def test_load_points_reads_dtr_and_num_tokens(self):
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
                            "is_correct": False,
                        },
                        {
                            "doc_id": 2,
                            "repeat_index": 0,
                            "dtr": 0.2,
                            "num_tokens": 5,
                            "is_correct": True,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            points = load_points(dtr_path)

            self.assertEqual(
                [
                    (
                        point.doc_id,
                        point.repeat_index,
                        point.dtr,
                        point.response_length,
                        point.is_correct,
                    )
                    for point in points
                ],
                [(2, 0, 0.2, 5, True), (3, 1, 0.7, 12, False)],
            )

    def test_load_points_requires_num_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dtr_path = Path(tmpdir) / "dtr.json"
            dtr_path.write_text(
                json.dumps([{"doc_id": 0, "repeat_index": 0, "dtr": 0.1}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing num_tokens"):
                load_points(dtr_path)

    def test_load_correctness_by_key_scores_each_repeat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples_aime24_custom_2026-03-22.jsonl"
            rows = [
                {
                    "doc_id": 0,
                    "target": "42",
                    "resps": [
                        [
                            gpt_oss_completion("r1", "\\boxed{42}"),
                            gpt_oss_completion("r2", "\\boxed{0}"),
                        ]
                    ],
                }
            ]
            samples_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            correctness_by_key = load_correctness_by_key(
                samples_path,
                reasoning_tags=build_gpt_oss_reasoning_tags(),
            )

            self.assertEqual(correctness_by_key, {(0, 0): True, (0, 1): False})

    def test_write_summary_json_matches_expected_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            results_path = run_dir / "results_2026-03-22T00-00-00.json"
            samples_path = run_dir / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
            dtr_path = dtr_results_path(run_dir)
            output_path = default_output_dir(run_dir) / "dtr_length_scatter.full.json"
            results_path.write_text(
                json.dumps(
                    {
                        "config": {
                            "model_args": {"pretrained": "openai/gpt-oss-120b"},
                        },
                        "results": {"aime24_custom": {}},
                    }
                ),
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
                            "is_correct": True,
                        },
                        {
                            "doc_id": 0,
                            "repeat_index": 1,
                            "dtr": 0.3,
                            "num_tokens": 20,
                            "is_correct": False,
                        },
                        {
                            "doc_id": 1,
                            "repeat_index": 0,
                            "dtr": 0.6,
                            "num_tokens": 40,
                            "is_correct": True,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            points = [
                SequenceLengthPoint(
                    doc_id=0,
                    repeat_index=0,
                    dtr=0.1,
                    response_length=10,
                    is_correct=True,
                ),
                SequenceLengthPoint(
                    doc_id=0,
                    repeat_index=1,
                    dtr=0.3,
                    response_length=20,
                    is_correct=False,
                ),
                SequenceLengthPoint(
                    doc_id=1,
                    repeat_index=0,
                    dtr=0.6,
                    response_length=40,
                    is_correct=True,
                ),
            ]
            pearson = pearson_r(
                [point.dtr for point in points],
                [float(point.response_length) for point in points],
            )
            write_summary_json(
                run_dir=run_dir,
                task_name="aime24_custom",
                model_name="openai/gpt-oss-120b",
                dtr_path=dtr_path,
                results_path=results_path,
                samples_path=samples_path,
                output_path=output_path,
                points=points,
                pearson=pearson,
                length_mode=DEFAULT_LENGTH_MODE,
            )

            summary = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(
                default_output_dir(run_dir), run_dir / DEFAULT_OUTPUT_DIR_NAME
            )
            self.assertEqual(summary["task"], "aime24_custom")
            self.assertEqual(summary["model"], "openai/gpt-oss-120b")
            self.assertEqual(summary["samples_path"], str(samples_path))
            self.assertEqual(summary["length_mode"], DEFAULT_LENGTH_MODE)
            self.assertEqual(summary["num_sequences"], 3)
            self.assertAlmostEqual(summary["pearson_r"], pearson)
            self.assertEqual(summary["dtr_min"], 0.1)
            self.assertEqual(summary["dtr_max"], 0.6)
            self.assertEqual(summary["length_min"], 10)
            self.assertEqual(summary["length_max"], 40)
            self.assertAlmostEqual(summary["length_mean"], 70 / 3)
            self.assertEqual(summary["correct_count"], 2)
            self.assertEqual(summary["incorrect_count"], 1)
            self.assertAlmostEqual(summary["correct_length_mean"], 25)
            self.assertAlmostEqual(summary["incorrect_length_mean"], 20)
            self.assertEqual(len(summary["points"]), 3)
            self.assertEqual(
                [entry["is_correct"] for entry in summary["points"]],
                [True, False, True],
            )

    def test_write_summary_json_defaults_empty_accuracy_groups_to_null(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            results_path = run_dir / "results_2026-03-22T00-00-00.json"
            dtr_path = dtr_results_path(run_dir)
            output_path = default_output_dir(run_dir) / "dtr_length_scatter.full.json"
            results_path.write_text(json.dumps({"results": {"aime24_custom": {}}}), encoding="utf-8")
            dtr_path.parent.mkdir(parents=True, exist_ok=True)
            dtr_path.write_text(
                json.dumps(
                    [
                        {
                            "doc_id": 0,
                            "repeat_index": 0,
                            "dtr": 0.1,
                            "num_tokens": 10,
                            "is_correct": None,
                        },
                        {
                            "doc_id": 0,
                            "repeat_index": 1,
                            "dtr": 0.3,
                            "num_tokens": 20,
                            "is_correct": None,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            points = [
                SequenceLengthPoint(
                    doc_id=0,
                    repeat_index=0,
                    dtr=0.1,
                    response_length=10,
                    is_correct=None,
                ),
                SequenceLengthPoint(
                    doc_id=0,
                    repeat_index=1,
                    dtr=0.3,
                    response_length=20,
                    is_correct=None,
                ),
            ]
            pearson = pearson_r(
                [point.dtr for point in points],
                [float(point.response_length) for point in points],
            )

            write_summary_json(
                run_dir=run_dir,
                task_name="aime24_custom",
                model_name="openai/gpt-oss-120b",
                dtr_path=dtr_path,
                results_path=results_path,
                samples_path=None,
                output_path=output_path,
                points=points,
                pearson=pearson,
                length_mode=DEFAULT_LENGTH_MODE,
            )

            summary = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["correct_count"], 0)
            self.assertEqual(summary["incorrect_count"], 0)
            self.assertIsNone(summary["correct_length_mean"])
            self.assertIsNone(summary["incorrect_length_mean"])

    def test_resolve_paths_uses_default_output_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            results_path = run_dir / "results_2026-03-22T00-00-00.json"
            samples_path = run_dir / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
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
                    "output_plot": None,
                    "output_json": None,
                    "length_mode": DEFAULT_LENGTH_MODE,
                },
            )()

            dtr_path, _results_path, output_plot, output_json = resolve_paths(args)

            self.assertEqual(dtr_path, dtr_results_path(run_dir))
            self.assertEqual(
                output_plot, default_output_dir(run_dir) / "dtr_length_scatter.full.png"
            )
            self.assertEqual(
                output_json, default_output_dir(run_dir) / "dtr_length_scatter.full.json"
            )

    @patch("src.aggregation.dtr_length_scatter.AutoTokenizer.from_pretrained")
    def test_load_response_lengths_by_key_counts_reasoning_until_end_token(
        self, tokenizer_mock: MagicMock
    ):
        class FakeTokenizer:
            def encode(
                self, text: str, add_special_tokens: bool = False
            ) -> list[int]:
                del add_special_tokens
                return list(range(len(text)))

        tokenizer_mock.return_value = FakeTokenizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples_aime24_custom_2026-03-22.jsonl"
            samples_path.write_text(
                json.dumps(
                    {
                        "doc_id": 0,
                        "target": "42",
                        "resps": [[gpt_oss_completion("abc", "\\boxed{42}")]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            lengths = load_response_lengths_by_key(
                samples_path,
                length_mode=REASONING_LENGTH_MODE,
                model_name="openai/gpt-oss-120b",
                reasoning_tags=build_gpt_oss_reasoning_tags(),
            )

            self.assertEqual(lengths, {(0, 0): len("abc<|end|>")})

    def test_load_response_lengths_by_key_rejects_missing_harmony_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples_aime24_custom_2026-03-22.jsonl"
            samples_path.write_text(
                json.dumps(
                    {
                        "doc_id": 0,
                        "target": "42",
                        "resps": [[
                            "<|start|>assistant<|channel|>analysis<|message|>abc"
                        ]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "harmony"):
                load_response_lengths_by_key(
                    samples_path,
                    length_mode=REASONING_LENGTH_MODE,
                    model_name="openai/gpt-oss-120b",
                    reasoning_tags=build_gpt_oss_reasoning_tags(),
                )

    def test_resolve_length_reasoning_tags_falls_back_for_gpt_oss(self):
        aggregated = {
            "config": {
                "model_args": {"pretrained": "openai/gpt-oss-120b"},
            }
        }

        self.assertEqual(
            resolve_length_reasoning_tags(aggregated, length_mode=REASONING_LENGTH_MODE),
            build_gpt_oss_reasoning_tags(),
        )

    def test_resolve_samples_path_matches_results_suffix_before_latest_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            results_path = run_dir / "results_2026-03-22T00-00-00.json"
            newer_samples = run_dir / "samples_aime24_custom_2026-03-23T00-00-00.jsonl"
            matched_samples = run_dir / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"

            results_path.write_text("{}", encoding="utf-8")
            newer_samples.write_text("{}\n", encoding="utf-8")
            matched_samples.write_text("{}\n", encoding="utf-8")

            resolved = resolve_samples_path(
                run_dir=run_dir,
                task_name="aime24_custom",
                results_path=results_path,
                samples_path=None,
            )

            self.assertEqual(resolved, matched_samples)

    def test_resolve_samples_path_requires_explicit_path_when_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            results_path = run_dir / "custom_results.json"
            (run_dir / "samples_aime24_custom_old.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            (run_dir / "samples_aime24_custom_new.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(FileNotFoundError, "pass --samples-path"):
                resolve_samples_path(
                    run_dir=run_dir,
                    task_name="aime24_custom",
                    results_path=results_path,
                    samples_path=None,
                )

    def test_resolve_model_name_prefers_pretrained_config(self):
        run_dir = Path("/tmp/results/aime24_custom/model/0/2026-03-25T00-00-00")
        aggregated = {"config": {"model_args": {"pretrained": "openai/gpt-oss-120b"}}}

        self.assertEqual(resolve_model_name(aggregated, run_dir), "openai/gpt-oss-120b")
        self.assertEqual(
            build_title(run_dir, "aime24_custom", "model", None),
            f"{run_dir.name} | aime24_custom | model | DTR vs Response Length",
        )


if __name__ == "__main__":
    unittest.main()
