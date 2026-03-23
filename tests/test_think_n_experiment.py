import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.experiment.think_n import (
    build_output_dir,
    resolve_selected_count,
    run_experiment,
)


def make_results_payload(*, repeats: int) -> dict:
    return {
        "config": {"model_args": {"pretrained": "openai/gpt-oss-120b"}},
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


def mixed_prefix_jsd(*, prefix_tokens: int, total_tokens: int) -> torch.Tensor:
    return torch.cat(
        [deep_jsd(prefix_tokens), shallow_jsd(total_tokens - prefix_tokens)],
        dim=0,
    )


def write_run_fixture(
    *,
    run_dir: Path,
    repeats: int,
    sample_rows: list[dict],
    matrices: dict[tuple[int, int], torch.Tensor],
    num_tokens: dict[tuple[int, int], int],
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "results_2026-03-23T00-00-00.json").write_text(
        json.dumps(make_results_payload(repeats=repeats)),
        encoding="utf-8",
    )
    (run_dir / "samples_aime24_custom_2026-03-23T00-00-00.jsonl").write_text(
        "\n".join(json.dumps(row) for row in sample_rows) + "\n",
        encoding="utf-8",
    )
    matrix_dir = run_dir / "jsd_matrices"
    matrix_dir.mkdir()
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


class ThinkNExperimentTest(unittest.TestCase):
    def test_resolve_selected_count_uses_fraction_or_explicit_value(self):
        self.assertEqual(
            resolve_selected_count(repeats=48, top_fraction=0.5, selected_count=None),
            24,
        )
        self.assertEqual(
            resolve_selected_count(repeats=48, top_fraction=0.5, selected_count=7),
            7,
        )

    def test_build_output_dir_keeps_run_identity_under_experiment_tree(self):
        run_dir = Path("/tmp/results/aime24_custom/gpt-oss-120b/0/20260323T000000Z")
        output_dir = build_output_dir(
            run_dir=run_dir,
            prefix_len=50,
            repeats=48,
            selected_count=24,
        )
        self.assertEqual(
            output_dir,
            run_dir
            / "experiments"
            / "prefix50_top50",
        )

    def test_run_experiment_uses_prefix_dtr_and_saves_summary_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "aime24_custom" / "gpt-oss-120b" / "0" / "20260323T000000Z"
            sample_rows = [
                make_sample_row(
                    doc_id=0,
                    target="42",
                    completions=["42", "42", "0", "0"],
                )
            ]
            matrices = {
                (0, 0): mixed_prefix_jsd(prefix_tokens=2, total_tokens=4),
                (0, 1): deep_jsd(4),
                (0, 2): shallow_jsd(4),
                (0, 3): shallow_jsd(4),
            }
            num_tokens = {
                (0, 0): 4,
                (0, 1): 4,
                (0, 2): 4,
                (0, 3): 4,
            }
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

            self.assertEqual(payload["summary"]["metrics"]["think_maj@2"], 1.0)
            self.assertEqual(payload["summary"]["metrics"]["mean_avg@4"], 0.5)
            self.assertEqual(payload["summary"]["delta"]["vs_mean_avg"], 0.5)
            self.assertEqual(payload["docs"][0]["selected_repeat_indices"], [0, 1])
            self.assertEqual(payload["docs"][0]["ranked_repeats"][0]["repeat_index"], 0)
            self.assertEqual(payload["docs"][0]["ranked_repeats"][0]["prefix_dtr"], 1.0)
            self.assertIn("think_maj@2: 1.000000", rendered)
            self.assertTrue(summary_json.is_file())
            self.assertTrue(summary_txt.is_file())

    def test_run_experiment_cost_counts_short_prefixes_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "aime24_custom" / "gpt-oss-120b" / "0" / "20260323T000000Z"
            sample_rows = [
                make_sample_row(
                    doc_id=0,
                    target="42",
                    completions=["42", "42", "0", "0"],
                )
            ]
            matrices = {
                (0, 0): deep_jsd(2),
                (0, 1): deep_jsd(6),
                (0, 2): shallow_jsd(3),
                (0, 3): shallow_jsd(7),
            }
            num_tokens = {
                (0, 0): 2,
                (0, 1): 6,
                (0, 2): 3,
                (0, 3): 7,
            }
            write_run_fixture(
                run_dir=run_dir,
                repeats=4,
                sample_rows=sample_rows,
                matrices=matrices,
                num_tokens=num_tokens,
            )

            summary_json, _summary_txt = run_experiment(
                run_dir=run_dir,
                prefix_len=5,
                selected_count=2,
            )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            cost = payload["summary"]["cost"]
            self.assertEqual(cost["total_full_tokens"], 18)
            self.assertEqual(cost["total_think_tokens"], 16)
            self.assertEqual(cost["saved_tokens"], 2)

    def test_run_experiment_breaks_dtr_ties_by_repeat_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "aime24_custom" / "gpt-oss-120b" / "0" / "20260323T000000Z"
            sample_rows = [
                make_sample_row(
                    doc_id=0,
                    target="42",
                    completions=["42", "0", "0", "0"],
                )
            ]
            matrices = {
                (0, 0): deep_jsd(4),
                (0, 1): deep_jsd(4),
                (0, 2): deep_jsd(4),
                (0, 3): deep_jsd(4),
            }
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

    def test_run_experiment_rejects_missing_matrix_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "aime24_custom" / "gpt-oss-120b" / "0" / "20260323T000000Z"
            sample_rows = [
                make_sample_row(
                    doc_id=0,
                    target="42",
                    completions=["42", "0", "0", "0"],
                )
            ]
            matrices = {
                (0, 0): deep_jsd(4),
                (0, 1): deep_jsd(4),
                (0, 2): deep_jsd(4),
            }
            num_tokens = {(0, index): 4 for index in range(3)}
            write_run_fixture(
                run_dir=run_dir,
                repeats=4,
                sample_rows=sample_rows,
                matrices=matrices,
                num_tokens=num_tokens,
            )

            with self.assertRaisesRegex(ValueError, "missing prefix DTR"):
                run_experiment(run_dir=run_dir, prefix_len=2, selected_count=2)


if __name__ == "__main__":
    unittest.main()
