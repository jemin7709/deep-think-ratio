import json
import tempfile
import unittest
from pathlib import Path

from src.aggregation.dtr_pass1_correlation import (
    DEFAULT_OUTPUT_DIR_NAME,
    DEFAULT_SUMMARY_FILENAME,
    build_title,
    default_output_dir,
    load_dtr_by_key,
    load_sequence_results,
    make_bins,
    pearson_r,
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


class PlotDtrPass1CorrelationTest(unittest.TestCase):
    def test_load_sequence_results_scores_raw_responses_per_repeat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples_aime24_custom_2026-03-22.jsonl"
            rows = [
                {
                    "doc_id": 0,
                    "target": "42",
                    "resps": [[
                        gpt_oss_completion("r1", "\\boxed{42}"),
                        gpt_oss_completion("r2", "\\boxed{0}"),
                    ]],
                },
                {
                    "doc_id": 1,
                    "target": "7",
                    "resps": [[
                        gpt_oss_completion("r3", "\\boxed{7}"),
                        gpt_oss_completion("r4", "\\boxed{7}"),
                    ]],
                },
            ]
            samples_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            dtr_by_key = {
                (0, 0): 0.1,
                (0, 1): 0.3,
                (1, 0): 0.6,
                (1, 1): 0.8,
            }

            sequence_rows = load_sequence_results(
                dtr_by_key,
                samples_path,
                reasoning_tags=build_gpt_oss_reasoning_tags(),
            )

            self.assertEqual(
                [(row.doc_id, row.repeat_index, row.dtr, row.pass_at_1) for row in sequence_rows],
                [(0, 0, 0.1, 1.0), (0, 1, 0.3, 0.0), (1, 0, 0.6, 1.0), (1, 1, 0.8, 1.0)],
            )

    def test_make_bins_and_summary_json_match_expected_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            results_path = run_dir / "results_2026-03-22T00-00-00.json"
            samples_path = run_dir / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
            dtr_path = dtr_results_path(run_dir)
            output_path = default_output_dir(run_dir) / DEFAULT_SUMMARY_FILENAME

            results_path.write_text(
                json.dumps({"config": {"model_args": {"pretrained": "openai/gpt-oss-120b"}}}),
                encoding="utf-8",
            )
            samples_path.write_text("", encoding="utf-8")
            dtr_path.parent.mkdir(parents=True, exist_ok=True)
            dtr_path.write_text(
                json.dumps(
                    [
                        {"doc_id": 0, "repeat_index": 0, "dtr": 0.1},
                        {"doc_id": 0, "repeat_index": 1, "dtr": 0.3},
                        {"doc_id": 1, "repeat_index": 0, "dtr": 0.6},
                        {"doc_id": 1, "repeat_index": 1, "dtr": 0.8},
                    ]
                ),
                encoding="utf-8",
            )

            rows = [
                *load_sequence_results(
                    load_dtr_by_key(dtr_path),
                    self._write_samples(samples_path),
                    reasoning_tags=build_gpt_oss_reasoning_tags(),
                )
            ]
            bins = make_bins(rows, num_bins=2)
            write_summary_json(
                run_dir=run_dir,
                task_name="aime24_custom",
                model_name="openai/gpt-oss-120b",
                dtr_path=dtr_path,
                results_path=results_path,
                samples_path=samples_path,
                output_path=output_path,
                rows=rows,
                bins=bins,
                binned_pearson=pearson_r(
                    [entry.mean_dtr for entry in bins],
                    [entry.pass_at_1 for entry in bins],
                ),
            )

            summary = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(default_output_dir(run_dir), run_dir / DEFAULT_OUTPUT_DIR_NAME)
            self.assertEqual(build_title(run_dir, "aime24_custom", "model", None), f"{run_dir.name} | aime24_custom | model | DTR vs Pass@1")
            self.assertEqual(summary["task"], "aime24_custom")
            self.assertEqual(summary["model"], "openai/gpt-oss-120b")
            self.assertEqual(summary["num_sequences"], 4)
            self.assertEqual(summary["num_bins"], 2)
            self.assertAlmostEqual(summary["bins"][0]["mean_dtr"], 0.2)
            self.assertAlmostEqual(summary["bins"][0]["pass_at_1"], 0.5)
            self.assertAlmostEqual(summary["bins"][1]["mean_dtr"], 0.7)
            self.assertAlmostEqual(summary["bins"][1]["pass_at_1"], 1.0)
            self.assertAlmostEqual(summary["pearson_r_binned"], 1.0)

    def test_load_sequence_results_rejects_unmatched_dtr_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples.jsonl"
            samples_path.write_text(
                json.dumps({"doc_id": 0, "target": "42", "resps": [["42"]]}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "without matching sample rows"):
                load_sequence_results(
                    {(0, 0): 0.1, (9, 0): 0.2},
                    samples_path,
                    reasoning_tags=None,
                )

    def _write_samples(self, samples_path: Path) -> Path:
        rows = [
            {
                "doc_id": 0,
                "target": "42",
                "resps": [[
                    gpt_oss_completion("r1", "\\boxed{42}"),
                    gpt_oss_completion("r2", "\\boxed{0}"),
                ]],
            },
            {
                "doc_id": 1,
                "target": "7",
                "resps": [[
                    gpt_oss_completion("r3", "\\boxed{7}"),
                    gpt_oss_completion("r4", "\\boxed{7}"),
                ]],
            },
        ]
        samples_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return samples_path


if __name__ == "__main__":
    unittest.main()
