import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.dtr.jsd_utils import jsd_output_dir
from src.experiment.think_n_bottom import build_output_dir
from src.experiment.think_n_bottom import resolve_selected_count
from src.experiment.think_n_bottom import run_experiment


def make_results_payload(
    *, repeats: int, reasoning_tags: list[list[str]] | None = None
) -> dict:
    config: dict[str, object] = {"model_args": {"pretrained": "openai/gpt-oss-120b"}}
    if reasoning_tags is not None:
        config["reasoning_tags"] = reasoning_tags
    return {
        "config": config,
        "results": {"aime24_custom": {}},
        "configs": {"aime24_custom": {"repeats": repeats}},
    }


def make_sample_row(*, doc_id: int, target: str, completions: list[str]) -> dict:
    return {
        "doc_id": doc_id,
        "target": target,
        "resps": [completions],
        "filter": "none",
        "arguments": {},
    }


def deep_jsd(num_tokens: int) -> torch.Tensor:
    return torch.full((num_tokens, 3), 0.6)


def shallow_jsd(num_tokens: int) -> torch.Tensor:
    return torch.full((num_tokens, 3), 0.1)


def write_run_fixture(
    *,
    run_dir: Path,
    repeats: int,
    sample_rows: list[dict],
    matrices: dict[tuple[int, int], torch.Tensor],
    num_tokens: dict[tuple[int, int], int],
    reasoning_tags: list[list[str]] | None = None,
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "results_2026-03-23T00-00-00.json").write_text(
        json.dumps(
            make_results_payload(repeats=repeats, reasoning_tags=reasoning_tags)
        ),
        encoding="utf-8",
    )
    (run_dir / "samples_aime24_custom_2026-03-23T00-00-00.jsonl").write_text(
        "\n".join(json.dumps(row) for row in sample_rows) + "\n",
        encoding="utf-8",
    )
    matrix_dir = jsd_output_dir(
        run_dir,
        hidden_state_mode="normed_normed",
        token_block_size=128,
    )
    matrix_dir.mkdir(parents=True)
    for key, jsd_matrix in matrices.items():
        doc_id, repeat_index = key
        torch.save(
            {
                "doc_id": doc_id,
                "repeat_index": repeat_index,
                "num_tokens": num_tokens[key],
                "jsd_matrix": jsd_matrix,
            },
            matrix_dir / f"doc{doc_id}_rep{repeat_index}.pt",
        )


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()


class ThinkNBottomExperimentTest(unittest.TestCase):
    def test_resolve_selected_count_uses_fraction_or_explicit_value(self):
        self.assertEqual(
            resolve_selected_count(
                repeats=48, bottom_fraction=0.5, selected_count=None
            ),
            24,
        )
        self.assertEqual(
            resolve_selected_count(repeats=48, bottom_fraction=0.5, selected_count=7),
            7,
        )
        with self.assertRaisesRegex(
            ValueError, "selected_count must be between 1 and repeats"
        ):
            resolve_selected_count(repeats=48, bottom_fraction=0.5, selected_count=0)
        with self.assertRaisesRegex(
            ValueError, "bottom_fraction must be in the interval"
        ):
            resolve_selected_count(repeats=48, bottom_fraction=1.5, selected_count=None)

    def test_build_output_dir_keeps_run_identity_under_experiment_tree(self):
        run_dir = Path("/tmp/results/aime24_custom/gpt-oss-120b/0/20260323T000000Z")
        output_dir = build_output_dir(
            run_dir=run_dir,
            prefix_len=50,
            repeats=48,
            selected_count=24,
            g=0.5,
            rho=0.85,
        )
        self.assertEqual(
            output_dir,
            run_dir / "experiments" / "prefix50_bottom24of48_g0.5_rho0.85",
        )

    @patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    def test_run_experiment_uses_lower_prefix_dtr_and_saves_rep_metrics(
        self, _tokenizer_mock
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = (
                Path(tmpdir)
                / "results"
                / "aime24_custom"
                / "gpt-oss-120b"
                / "0"
                / "20260323T000000Z"
            )
            sample_rows = [
                make_sample_row(
                    doc_id=0,
                    target="42",
                    completions=[
                        "42 x x x x",
                        "42 y y y y",
                        "0 ok unique",
                        "0 still unique enough",
                    ],
                )
            ]
            matrices = {
                (0, 0): deep_jsd(5),
                (0, 1): deep_jsd(5),
                (0, 2): shallow_jsd(5),
                (0, 3): shallow_jsd(5),
            }
            num_tokens = {(0, index): 5 for index in range(4)}
            write_run_fixture(
                run_dir=run_dir,
                repeats=4,
                sample_rows=sample_rows,
                matrices=matrices,
                num_tokens=num_tokens,
            )

            summary_json, summary_txt = run_experiment(
                run_dir=run_dir,
                prefix_len=2,
                selected_count=2,
            )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            rendered = summary_txt.read_text(encoding="utf-8")

            self.assertEqual(payload["summary"]["metrics"]["bottom_maj@2"], 0.0)
            self.assertEqual(
                payload["cost_definition"]["bottom_tokens"],
                "sum_selected(min(prefix_len, full_num_tokens) + full_num_tokens)",
            )
            self.assertEqual(payload["summary"]["metrics"]["mean_avg@4"], 0.5)
            self.assertEqual(payload["summary"]["delta"]["vs_mean_avg"], -0.5)
            self.assertEqual(payload["docs"][0]["selected_repeat_indices"], [2, 3])
            self.assertEqual(payload["docs"][0]["ranked_repeats"][0]["repeat_index"], 2)
            self.assertEqual(payload["docs"][0]["ranked_repeats"][0]["prefix_dtr"], 0.0)
            self.assertEqual(
                payload["docs"][0]["selection_stats"]["selected_mean_num_tokens"],
                5.0,
            )
            self.assertFalse(
                payload["docs"][0]["selection_stats"]["selected_majority_correct"]
            )
            self.assertEqual(payload["docs"][0]["metrics"]["selected_word_rep_2"], 0.0)
            self.assertGreater(payload["docs"][0]["metrics"]["full_word_rep_2"], 0.0)
            self.assertIn("bottom_maj@2: 0.000000", rendered)
            self.assertIn(
                "mean_selected_tokens_per_selected_repeat: 5.000000",
                rendered,
            )
            self.assertIn(
                "cost_formula_bottom_tokens: "
                "sum_selected(min(prefix_len, full_num_tokens) + full_num_tokens)",
                rendered,
            )
            self.assertIn("selected_token_rep_2", rendered)
            self.assertIn("full_word_rep_4", rendered)
            self.assertTrue(summary_json.is_file())
            self.assertTrue(summary_txt.is_file())
            self.assertEqual(
                summary_json.parent.name,
                "prefix2_bottom2of4_g0.5_rho0.85",
            )

    @patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    def test_run_experiment_breaks_dtr_ties_by_repeat_index(self, _tokenizer_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = (
                Path(tmpdir)
                / "results"
                / "aime24_custom"
                / "gpt-oss-120b"
                / "0"
                / "20260323T000000Z"
            )
            sample_rows = [
                make_sample_row(
                    doc_id=0,
                    target="42",
                    completions=["42", "0", "0", "0"],
                )
            ]
            matrices = {(0, index): shallow_jsd(4) for index in range(4)}
            num_tokens = {(0, index): 4 for index in range(4)}
            write_run_fixture(
                run_dir=run_dir,
                repeats=4,
                sample_rows=sample_rows,
                matrices=matrices,
                num_tokens=num_tokens,
            )

            summary_json, _summary_txt = run_experiment(
                run_dir=run_dir,
                prefix_len=2,
                selected_count=2,
            )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["docs"][0]["selected_repeat_indices"], [0, 1])

    @patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    def test_run_experiment_sets_rep_n_zero_for_short_sequences(self, _tokenizer_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = (
                Path(tmpdir)
                / "results"
                / "aime24_custom"
                / "gpt-oss-120b"
                / "0"
                / "20260323T000000Z"
            )
            sample_rows = [
                make_sample_row(
                    doc_id=0,
                    target="42",
                    completions=["42", "0", "0", "0"],
                )
            ]
            matrices = {(0, index): shallow_jsd(1) for index in range(4)}
            num_tokens = {(0, index): 1 for index in range(4)}
            write_run_fixture(
                run_dir=run_dir,
                repeats=4,
                sample_rows=sample_rows,
                matrices=matrices,
                num_tokens=num_tokens,
            )

            summary_json, _summary_txt = run_experiment(
                run_dir=run_dir,
                prefix_len=1,
                selected_count=2,
            )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["docs"][0]["selection_stats"]["selected_mean_num_tokens"],
                1.0,
            )
            self.assertTrue(
                payload["docs"][0]["selection_stats"]["selected_majority_correct"]
            )
            self.assertEqual(payload["docs"][0]["metrics"]["selected_token_rep_2"], 0.0)
            self.assertEqual(payload["docs"][0]["metrics"]["selected_word_rep_4"], 0.0)

    @patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    def test_run_experiment_computes_repetition_on_raw_response(self, _tokenizer_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = (
                Path(tmpdir)
                / "results"
                / "aime24_custom"
                / "gpt-oss-120b"
                / "0"
                / "20260323T000000Z"
            )
            repeated = (
                "<|channel|>analysis<|message|> loop loop loop "
                "<|end|><|start|>assistant<|channel|>final<|message|>42 once"
            )
            sample_rows = [
                make_sample_row(
                    doc_id=0,
                    target="42",
                    completions=[repeated, repeated, "0 once", "0 twice"],
                )
            ]
            matrices = {
                (0, 0): deep_jsd(4),
                (0, 1): deep_jsd(4),
                (0, 2): shallow_jsd(4),
                (0, 3): shallow_jsd(4),
            }
            num_tokens = {(0, index): 4 for index in range(4)}
            write_run_fixture(
                run_dir=run_dir,
                repeats=4,
                sample_rows=sample_rows,
                matrices=matrices,
                num_tokens=num_tokens,
                reasoning_tags=[
                    [
                        "<|channel|>analysis<|message|>",
                        "<|end|><|start|>assistant<|channel|>final<|message|>",
                    ]
                ],
            )

            summary_json, summary_txt = run_experiment(
                run_dir=run_dir,
                prefix_len=2,
                selected_count=2,
            )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            rendered = summary_txt.read_text(encoding="utf-8")
            self.assertEqual(payload["docs"][0]["selected_repeat_indices"], [2, 3])
            self.assertGreater(payload["docs"][0]["metrics"]["full_word_rep_2"], 0.0)
            match = re.search(
                r"^full_word_rep_2:\s*([0-9]+(?:\.[0-9]+)?)$",
                rendered,
                re.MULTILINE,
            )
            self.assertIsNotNone(match)
            self.assertGreater(float(match.group(1)), 0.0)

    @patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    def test_run_experiment_rejects_non_positive_prefix_len(self, _tokenizer_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = (
                Path(tmpdir)
                / "results"
                / "aime24_custom"
                / "gpt-oss-120b"
                / "0"
                / "20260323T000000Z"
            )
            write_run_fixture(
                run_dir=run_dir,
                repeats=1,
                sample_rows=[
                    make_sample_row(doc_id=0, target="42", completions=["42"])
                ],
                matrices={(0, 0): shallow_jsd(1)},
                num_tokens={(0, 0): 1},
            )

            with self.assertRaisesRegex(ValueError, "prefix_len must be >= 1"):
                run_experiment(run_dir=run_dir, prefix_len=0, selected_count=1)


if __name__ == "__main__":
    unittest.main()
