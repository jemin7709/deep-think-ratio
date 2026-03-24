import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from src.dtr.jsd import build_jsd_payload
from src.dtr.jsd import save_jsd_payload
from src.dtr.jsd_utils import compute_dtr_from_jsd_matrix
from src.dtr.jsd_utils import dtr_results_path
from src.dtr.jsd_utils import infer_task_name
from src.dtr.jsd_utils import jsd_output_dir
from src.dtr.jsd_utils import latest_matching_file
from src.dtr.jsd_utils import latest_samples_file
from src.dtr.jsd_utils import load_aggregated_results
from src.dtr.jsd_utils import load_samples
from src.dtr.jsd_utils import resolve_model_path
from src.dtr.jsd_utils import tokenize_prompt_and_response


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
            self.assertEqual(
                dtr_results_path(run_dir, g=0.5, rho=0.85),
                run_dir / "dtr" / "dtr_g0.5_rho0.85.json",
            )
            self.assertEqual(
                jsd_output_dir(
                    run_dir,
                    hidden_state_mode="normed_normed",
                    token_block_size=128,
                ),
                run_dir / "jsd_matrices" / "normed_normed_tb128",
            )

    def test_load_samples_expands_repeat_indices_from_current_lm_eval_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = (
                Path(tmpdir) / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
            )
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
            self.assertTrue(
                all(sample.prompt_text == "prompt text" for sample in samples)
            )

    def test_load_samples_filters_requested_repeat_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = (
                Path(tmpdir) / "samples_aime24_custom_2026-03-22T00-00-00.jsonl"
            )
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
    def test_tokenize_prompt_and_response_keeps_response_tokens_across_boundary_merge(
        self,
    ):
        class FakeTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                mapping = {
                    "a": [1],
                    "b": [3],
                    "ab": [2],
                }
                return mapping[text]

        prompt_token_ids, response_token_ids = tokenize_prompt_and_response(
            FakeTokenizer(),
            "a",
            "b",
        )

        self.assertEqual(prompt_token_ids.tolist(), [1])
        self.assertEqual(response_token_ids.tolist(), [3])

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

    def test_compute_dtr_from_jsd_matrix_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least one token row"):
            compute_dtr_from_jsd_matrix(torch.ones((0, 3)), g=0.5, rho=0.85)
        with self.assertRaisesRegex(ValueError, "at least one layer column"):
            compute_dtr_from_jsd_matrix(torch.ones((2, 0)), g=0.5, rho=0.85)
        with self.assertRaisesRegex(ValueError, r"g must be in the interval \[0, 1\]"):
            compute_dtr_from_jsd_matrix(torch.ones((2, 3)), g=-0.1, rho=0.85)
        with self.assertRaisesRegex(ValueError, r"g must be in the interval \[0, 1\]"):
            compute_dtr_from_jsd_matrix(torch.ones((2, 3)), g=1.1, rho=0.85)
        with self.assertRaisesRegex(
            ValueError, r"rho must be in the interval \(0, 1\]"
        ):
            compute_dtr_from_jsd_matrix(torch.ones((2, 3)), g=0.5, rho=0.0)
        with self.assertRaisesRegex(
            ValueError, r"rho must be in the interval \(0, 1\]"
        ):
            compute_dtr_from_jsd_matrix(torch.ones((2, 3)), g=0.5, rho=1.2)

    def test_dtr_module_rejects_empty_matrix_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            matrix_dir = jsd_output_dir(
                run_dir,
                hidden_state_mode="normed_normed",
                token_block_size=128,
            )
            matrix_dir.mkdir(parents=True)

            result = subprocess.run(
                ["uv", "run", "python", "-m", "src.dtr.dtr", str(run_dir)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(dtr_results_path(run_dir).exists())

    def test_save_jsd_payload_rejects_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "doc3_rep4.pt"
            first_payload = build_jsd_payload(
                doc_id=3,
                repeat_index=4,
                model_path="model/a",
                task_name="aime24_custom",
                samples_path=Path("/tmp/samples_a.jsonl"),
                hidden_state_mode="normed_normed",
                token_block_size=128,
                response_token_ids=torch.tensor([11, 12], dtype=torch.long),
                jsd_matrix=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            )
            second_payload = build_jsd_payload(
                doc_id=3,
                repeat_index=4,
                model_path="model/b",
                task_name="aime24_custom",
                samples_path=Path("/tmp/samples_a.jsonl"),
                hidden_state_mode="normed_normed",
                token_block_size=128,
                response_token_ids=torch.tensor([11, 12], dtype=torch.long),
                jsd_matrix=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            )

            save_jsd_payload(output_path, first_payload)

            with self.assertRaisesRegex(FileExistsError, "mismatched metadata"):
                save_jsd_payload(output_path, second_payload)

    def test_save_jsd_payload_allows_same_metadata_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "doc3_rep4.pt"
            payload = build_jsd_payload(
                doc_id=3,
                repeat_index=4,
                model_path="model/a",
                task_name="aime24_custom",
                samples_path=Path("/tmp/samples_a.jsonl"),
                hidden_state_mode="normed_normed",
                token_block_size=128,
                response_token_ids=torch.tensor([11, 12], dtype=torch.long),
                jsd_matrix=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            )

            save_jsd_payload(output_path, payload)
            save_jsd_payload(output_path, payload)

            restored = torch.load(output_path, map_location="cpu", weights_only=False)
            self.assertEqual(restored["model_path"], "model/a")
            self.assertEqual(restored["hidden_state_mode"], "normed_normed")


if __name__ == "__main__":
    unittest.main()
