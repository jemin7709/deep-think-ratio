import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from src.dtr.jsd_utils import compute_dtr_from_jsd_matrix
from src.dtr.jsd_utils import infer_task_name
from src.dtr.jsd_utils import latest_matching_file
from src.dtr.jsd_utils import latest_samples_file
from src.dtr.jsd_utils import load_aggregated_results
from src.dtr.jsd_utils import load_samples
from src.dtr.jsd_utils import resolve_model_path


def sample_row(*, doc_id: int, prompt: str, completions: list[str]) -> dict:
    return {
        "doc_id": doc_id,
        "target": "42",
        "resps": [completions],
        "filter": "none",
        "arguments": {
            "gen_args_0": {
                "arg_0": prompt,
            }
        },
    }


class DtrIoTest(unittest.TestCase):
    def test_latest_results_and_samples_are_selected_from_run_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            older_results = run_dir / "results_2026-03-21T00-00-00.json"
            newer_results = run_dir / "results_2026-03-22T00-00-00.json"
            older_results.write_text(
                json.dumps({"config": {"model_args": {"pretrained": "older/model"}}}),
                encoding="utf-8",
            )
            newer_results.write_text(
                json.dumps(
                    {
                        "config": {"model_args": {"pretrained": "newer/model"}},
                        "results": {"aime24_custom": {}},
                    }
                ),
                encoding="utf-8",
            )
            os.utime(older_results, (1, 1))
            os.utime(newer_results, (2, 2))

            older_samples = run_dir / "samples_aime24_custom_2026-03-21T00-00-00.jsonl"
            newer_samples = run_dir / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
            older_samples.write_text("{}\n", encoding="utf-8")
            newer_samples.write_text("{}\n", encoding="utf-8")
            os.utime(older_samples, (1, 1))
            os.utime(newer_samples, (2, 2))

            self.assertEqual(
                latest_matching_file(run_dir, "results_*.json"),
                newer_results,
            )
            self.assertEqual(
                load_aggregated_results(run_dir),
                json.loads(newer_results.read_text()),
            )
            self.assertEqual(
                infer_task_name(load_aggregated_results(run_dir)),
                "aime24_custom",
            )
            self.assertEqual(
                latest_samples_file(run_dir, "aime24_custom"),
                newer_samples,
            )
            self.assertEqual(
                resolve_model_path(load_aggregated_results(run_dir)),
                "newer/model",
            )

    def test_load_samples_expands_repeat_indices_from_current_lm_eval_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
            row = sample_row(
                doc_id=7,
                prompt="prompt text",
                completions=["resp-0", "resp-1", "resp-2"],
            )
            samples_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            samples = load_samples(samples_path)

            self.assertEqual(
                [
                    (sample.doc_id, sample.repeat_index, sample.response_text)
                    for sample in samples
                ],
                [(7, 0, "resp-0"), (7, 1, "resp-1"), (7, 2, "resp-2")],
            )
            self.assertTrue(all(sample.prompt_text == "prompt text" for sample in samples))

    def test_load_samples_filters_requested_repeat_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
            row = sample_row(
                doc_id=3,
                prompt="prompt text",
                completions=["resp-0", "resp-1", "resp-2"],
            )
            samples_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            samples = load_samples(samples_path, repeat_indices=[1])

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].repeat_index, 1)
            self.assertEqual(samples[0].response_text, "resp-1")


class DtrComputationTest(unittest.TestCase):
    def test_compute_dtr_from_jsd_matrix_marks_deep_tokens(self):
        jsd_matrix = torch.tensor(
            [
                [0.9, 0.6, 0.2],
                [0.4, 0.3, 0.2],
                [0.9, 0.7, 0.6],
            ]
        )

        result = compute_dtr_from_jsd_matrix(jsd_matrix, g=0.5, rho=0.75)

        self.assertAlmostEqual(result.dtr, 2 / 3)
        self.assertEqual(result.settling_depth.tolist(), [3, 1, 4])
        self.assertEqual(result.deep_mask.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
