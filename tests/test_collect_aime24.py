import json
import tempfile
import unittest
from pathlib import Path

from tasks.aime24.metrics import (
    build_postprocess_payload,
    format_summary,
    infer_repeats,
    infer_task_name,
    summarize_run,
    write_postprocess_artifacts,
)
from tasks.aime24.utils import build_gpt_oss_reasoning_tags


def sample_record(doc_id, target, completions, filtered_resps=None):
    return {
        "doc_id": doc_id,
        "target": str(target),
        "resps": [completions],
        "filtered_resps": filtered_resps or [["wrong"]],
        "filter": "none",
        "arguments": {},
    }


def gpt_oss_completion(reasoning: str, content: str) -> str:
    return (
        "<|start|>assistant<|channel|>analysis<|message|>"
        f"{reasoning}"
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        f"{content}"
        "<|end|>"
    )


class CollectAime24Test(unittest.TestCase):
    def test_pass_at_1_matches_average_accuracy_using_all_repeats(self):
        samples = [
            sample_record(0, "42", ["42", "0", "1", "2"]),
            sample_record(1, "7", ["9", "7", "7", "8"]),
        ]
        summary = summarize_run(samples, k=1, expected_n=4)
        self.assertAlmostEqual(summary["pass"], 0.375)
        self.assertAlmostEqual(summary["avg"], 0.375)

    def test_majority_vote_uses_extracted_target(self):
        samples = [
            sample_record(
                0,
                "42",
                [
                    "I think the answer is \\boxed{42}.",
                    "After checking again, $42$.",
                    "x = 42",
                    "41",
                ],
            )
        ]
        summary = summarize_run(samples, k=1, expected_n=4)
        self.assertEqual(summary["maj"], 1.0)

    def test_infer_task_name_and_repeats_from_aggregated_results(self):
        aggregated = {
            "results": {"aime24_custom": {}},
            "configs": {"aime24_custom": {"repeats": 10}},
        }

        self.assertEqual(infer_task_name(aggregated), "aime24_custom")
        self.assertEqual(infer_repeats(aggregated, "aime24_custom"), 10)

    def test_infer_repeats_from_task_config_when_aggregated_missing(self):
        aggregated = {"results": {"aime24_custom": {}}}
        self.assertEqual(infer_repeats(aggregated, "aime24_custom"), 48)

    def test_write_postprocess_artifacts_uses_raw_resps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            aggregated = {
                "config": {
                    "model": "dummy",
                    "model_args": {"pretrained": "openai/gpt-oss-120b"},
                    "metadata": {
                        "reasoning_tags": build_gpt_oss_reasoning_tags(),
                    },
                },
                "results": {"aime24_custom": {}},
                "configs": {"aime24_custom": {"repeats": 4}},
            }
            (run_dir / "results_2026-03-21T00-00-00.json").write_text(
                json.dumps(aggregated),
                encoding="utf-8",
            )

            sample = sample_record(
                0,
                "42",
                [
                    gpt_oss_completion("reasoning a", "\\boxed{42}"),
                    gpt_oss_completion("reasoning b", "\\boxed{0}"),
                    gpt_oss_completion("reasoning c", "x = 42"),
                    gpt_oss_completion("reasoning d", "41"),
                ],
                filtered_resps=[["0"]],
            )
            (run_dir / "samples_aime24_custom_2026-03-21T00-00-00.jsonl").write_text(
                json.dumps(sample) + "\n",
                encoding="utf-8",
            )

            postprocess_path, summary_path = write_postprocess_artifacts(run_dir=run_dir)

            payload = json.loads(postprocess_path.read_text(encoding="utf-8"))
            rendered = summary_path.read_text(encoding="utf-8")

            self.assertEqual(postprocess_path.name, "postprocess_2026-03-21T00-00-00.json")
            self.assertEqual(summary_path.name, "summary_2026-03-21T00-00-00.txt")
            self.assertEqual(payload["model"], "openai/gpt-oss-120b")
            self.assertEqual(payload["metrics"]["avg@4"], 0.5)
            self.assertEqual(payload["metrics"]["maj@4"], 1.0)
            self.assertNotIn("first@1", payload["metrics"])
            self.assertIn("avg@4: 0.500000", rendered)
            self.assertIn("maj@4: 1.000000", rendered)
            self.assertNotIn("first@1", rendered)

    def test_summarize_run_rejects_repeat_mismatch(self):
        samples = [sample_record(0, "42", ["42", "0", "1"])]
        with self.assertRaisesRegex(ValueError, "expected 4 completions"):
            summarize_run(samples, k=1, expected_n=4)

    def test_build_postprocess_payload_and_summary_render(self):
        aggregated = {"config": {"model_args": {"pretrained": "dummy/model"}}}
        summary = {
            "num_docs": 1.0,
            "pass": 0.5,
            "avg": 0.5,
            "maj": 1.0,
        }
        payload = build_postprocess_payload(
            Path("/tmp/run"),
            aggregated,
            summary,
            "aime24_custom",
            1,
            10,
        )
        rendered = format_summary(
            Path("/tmp/run"),
            aggregated,
            summary,
            "aime24_custom",
            1,
            10,
        )

        self.assertEqual(payload["metrics"]["avg@10"], 0.5)
        self.assertNotIn("first@1", payload["metrics"])
        self.assertEqual(payload["model"], "dummy/model")
        self.assertIn("task: aime24_custom", rendered)
        self.assertIn("avg@10: 0.500000", rendered)


if __name__ == "__main__":
    unittest.main()
