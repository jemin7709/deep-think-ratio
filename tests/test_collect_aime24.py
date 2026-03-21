import json
import tempfile
import unittest
from pathlib import Path

from tasks.aime24.metrics import (
    canonical_answers,
    format_summary,
    majority_answer,
    pass_at_k,
    summarize_run,
)
from tasks.aime24.utils import is_equiv


def sample_record(doc_id, target, completions):
    return {
        "doc_id": doc_id,
        "target": str(target),
        "resps": [completions],
        "filter": "none",
    }


class CollectAime24Test(unittest.TestCase):
    def test_pass_at_k_handles_edges(self):
        self.assertEqual(pass_at_k(25, 0, 1), 0.0)
        self.assertEqual(pass_at_k(25, 25, 1), 1.0)
        self.assertEqual(pass_at_k(25, 1, 25), 1.0)

    def test_pass_at_1_matches_average_accuracy(self):
        samples = [
            sample_record(0, "42", ["42", "0", "1", "2"]),
            sample_record(1, "7", ["9", "7", "7", "8"]),
        ]
        summary = summarize_run(samples, k=1, expected_n=4)
        self.assertAlmostEqual(summary["pass"], 0.375)

    def test_majority_vote_uses_canonical_answer(self):
        completions = [
            "I think the answer is \\boxed{42}.",
            "After checking again, $42$.",
            "x = 42",
            "41",
        ]
        answers = canonical_answers(completions)
        self.assertTrue(is_equiv(majority_answer(answers), "42"))

    def test_collect_run_from_synthetic_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            model_dir = run_dir / "dummy"
            model_dir.mkdir()

            aggregated = {"config": {"model": "dummy"}}
            (model_dir / "results_2026-03-21T00-00-00.json").write_text(
                json.dumps(aggregated),
                encoding="utf-8",
            )

            sample = sample_record(0, "42", ["42", "0", "42", "41"])
            with (model_dir / "samples_aime24_sc_25_2026-03-21T00-00-00.jsonl").open(
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(sample) + "\n")

            from tasks.aime24.metrics import load_aggregated, load_samples

            loaded_aggregated = load_aggregated(run_dir)
            loaded_samples = load_samples(run_dir, "aime24_sc_25")

            self.assertEqual(loaded_aggregated["config"]["model"], "dummy")
            self.assertEqual(len(loaded_samples), 1)
            rendered = format_summary(
                model_dir,
                loaded_aggregated,
                {"num_docs": 1.0, "pass": 0.5, "avg": 0.5, "maj": 1.0, "first": 1.0},
                1,
                25,
            )
            self.assertIn("avg@25: 0.500000", rendered)
            self.assertIn("maj@25: 1.000000", rendered)


if __name__ == "__main__":
    unittest.main()
