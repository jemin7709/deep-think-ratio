import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from src.deep_think_tokens_project.dtr_pass1_correlation import resolve_paths
from src.deep_think_tokens_project.dtr_pass1_correlation import load_dtr_by_key
from src.deep_think_tokens_project.dtr_pass1_correlation import load_prefix_dtr_by_key
from src.deep_think_tokens_project.dtr_pass1_correlation import load_sequence_results
from src.deep_think_tokens_project.dtr_pass1_correlation import make_bins
from src.deep_think_tokens_project.dtr_pass1_correlation import pearson_r
from src.deep_think_tokens_project.dtr_pass1_correlation import write_summary_json
from src.deep_think_tokens_project.io import copied_run_dir
from src.deep_think_tokens_project.io import copy_raw_run
from src.deep_think_tokens_project.io import discover_run_dirs
from src.deep_think_tokens_project.io import dtr_results_path
from src.deep_think_tokens_project.io import jsd_output_dir
from src.deep_think_tokens_project.jsd import replay_tokens
from src.deep_think_tokens_project.think_n import run_experiment as run_top_experiment
from src.deep_think_tokens_project.think_n_bottom import (
    run_experiment as run_bottom_experiment,
)
from src.deep_think_tokens_project.utils import compute_dtr_from_divergence_matrix


class FakeTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        token_ids: list[int] = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) + 1
            token_ids.append(self._vocab[token])
        return token_ids


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


def results_payload(*, repeats: int) -> dict:
    return {
        "config": {"model_args": {"pretrained": "openai/gpt-oss-120b"}},
        "results": {"aime24_custom": {}},
        "configs": {"aime24_custom": {"repeats": repeats}},
    }


def deep_divergence(num_tokens: int) -> torch.Tensor:
    return torch.full((3, num_tokens), 0.6)


def shallow_divergence(num_tokens: int) -> torch.Tensor:
    return torch.full((3, num_tokens), 0.1)


def medium_divergence() -> torch.Tensor:
    return torch.tensor(
        [
            [0.9, 0.4, 0.9, 0.4],
            [0.8, 0.4, 0.8, 0.4],
            [0.4, 0.4, 0.4, 0.4],
        ],
        dtype=torch.float32,
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
    (run_dir / "results_2026-03-25T00-00-00.json").write_text(
        json.dumps(results_payload(repeats=repeats)),
        encoding="utf-8",
    )
    (run_dir / "samples_aime24_custom_2026-03-25T00-00-00.jsonl").write_text(
        "\n".join(json.dumps(row) for row in sample_rows) + "\n",
        encoding="utf-8",
    )
    matrix_dir = jsd_output_dir(run_dir)
    matrix_dir.mkdir(parents=True)
    for key, divergence_matrix in matrices.items():
        doc_id, repeat_index = key
        torch.save(
            {
                "doc_id": doc_id,
                "repeat_index": repeat_index,
                "num_tokens": num_tokens[key],
                "divergence_matrix": divergence_matrix,
                "response_token_ids": torch.tensor(
                    [10] * num_tokens[key],
                    dtype=torch.long,
                ),
                "model_path": "openai/gpt-oss-120b",
                "task_name": "aime24_custom",
                "samples_path": str(
                    run_dir / "samples_aime24_custom_2026-03-25T00-00-00.jsonl"
                ),
            },
            matrix_dir / f"doc{doc_id}_rep{repeat_index}.pt",
        )


def write_dtr_rows(run_dir: Path) -> Path:
    dtr_path = dtr_results_path(run_dir, g=0.5, p=0.9)
    dtr_path.parent.mkdir(parents=True, exist_ok=True)
    dtr_path.write_text(
        json.dumps(
            [
                {
                    "doc_id": 0,
                    "repeat_index": 0,
                    "dtr": 1.0,
                    "num_tokens": 4,
                    "num_deep_tokens": 4,
                },
                {
                    "doc_id": 0,
                    "repeat_index": 1,
                    "dtr": 0.5,
                    "num_tokens": 4,
                    "num_deep_tokens": 2,
                },
                {
                    "doc_id": 0,
                    "repeat_index": 2,
                    "dtr": 0.0,
                    "num_tokens": 4,
                    "num_deep_tokens": 0,
                },
                {
                    "doc_id": 0,
                    "repeat_index": 3,
                    "dtr": 0.0,
                    "num_tokens": 4,
                    "num_deep_tokens": 0,
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return dtr_path


class PrepareResultsTest(unittest.TestCase):
    def test_copy_raw_run_only_copies_results_and_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "results"
            source_run = source_root / "aime24_custom" / "model" / "0" / "timestamp"
            source_run.mkdir(parents=True)
            (source_run / "results_foo.json").write_text("{}", encoding="utf-8")
            (source_run / "samples_aime24_custom_foo.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (source_run / "dtr").mkdir()
            (source_run / "dtr" / "old.json").write_text("[]", encoding="utf-8")
            (source_run / "jsd_matrices").mkdir()
            (source_run / "jsd_matrices" / "old.pt").write_bytes(b"pt")
            ignored_run = source_root / "aime24_custom" / "model" / "1" / "timestamp"
            ignored_run.mkdir(parents=True)
            (ignored_run / "results_foo.json").write_text("{}", encoding="utf-8")

            target_root = Path(tmpdir) / "results_deep_think_tokens"
            self.assertEqual(discover_run_dirs(source_root), [source_run])
            self.assertEqual(
                copied_run_dir(
                    source_run,
                    source_root=source_root,
                    target_root=target_root,
                ),
                target_root / "aime24_custom" / "model" / "0" / "timestamp",
            )
            copied = copy_raw_run(
                source_run,
                source_root=source_root,
                target_root=target_root,
            )

            self.assertEqual(
                copied,
                target_root / "aime24_custom" / "model" / "0" / "timestamp",
            )
            self.assertTrue((copied / "results_foo.json").is_file())
            self.assertTrue((copied / "samples_aime24_custom_foo.jsonl").is_file())
            self.assertFalse((copied / "dtr").exists())
            self.assertFalse((copied / "jsd_matrices").exists())


class DtrSemanticsTest(unittest.TestCase):
    def test_replay_tokens_keeps_last_prompt_and_response_prefix(self):
        replay = replay_tokens(
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4, 5, 6], dtype=torch.long),
        )
        self.assertEqual(replay.tolist(), [3, 4, 5])

    def test_compute_dtr_from_divergence_matrix_matches_upstream_semantics(self):
        divergence_matrix = torch.tensor(
            [
                [0.9, 0.4, 0.9],
                [0.6, 0.3, 0.7],
                [0.2, 0.2, 0.6],
            ],
            dtype=torch.float32,
        )

        result = compute_dtr_from_divergence_matrix(
            divergence_matrix,
            g=0.5,
            p=0.75,
        )

        self.assertAlmostEqual(result.dtr, 2 / 3)
        self.assertEqual(result.first_deep_layer.tolist(), [2, 0, 3])
        self.assertEqual(result.deep_mask.tolist(), [True, False, True])

    def test_dtr_results_path_uses_p_not_rho(self):
        run_dir = Path("/tmp/run")
        self.assertEqual(
            dtr_results_path(run_dir, g=0.5, p=0.9),
            run_dir / "dtr" / "dtr_g0.5_p0.9.json",
        )


class ThinkNExperimentTest(unittest.TestCase):
    @patch(
        "src.experiment.repetition_metrics.AutoTokenizer.from_pretrained",
        return_value=FakeTokenizer(),
    )
    def test_top_and_bottom_experiments_use_p_slug_and_prefix_dtr(
        self, _tokenizer_mock: object
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = (
                Path(tmpdir)
                / "results_deep_think_tokens"
                / "aime24_custom"
                / "gpt-oss-120b"
                / "0"
                / "20260325T000000Z"
            )
            sample_rows = [
                sample_row(
                    doc_id=0,
                    prompt="prompt",
                    completions=["42", "42", "0", "0"],
                )
            ]
            matrices = {
                (0, 0): deep_divergence(4),
                (0, 1): medium_divergence(),
                (0, 2): shallow_divergence(4),
                (0, 3): shallow_divergence(4),
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

            top_summary_json, _top_summary_txt = run_top_experiment(
                run_dir=run_dir,
                prefix_len=2,
                selected_count=2,
            )
            bottom_summary_json, _bottom_summary_txt = run_bottom_experiment(
                run_dir=run_dir,
                prefix_len=2,
                selected_count=2,
            )

            top_payload = json.loads(top_summary_json.read_text(encoding="utf-8"))
            bottom_payload = json.loads(bottom_summary_json.read_text(encoding="utf-8"))

            self.assertEqual(top_payload["summary"]["metrics"]["think_maj@2"], 1.0)
            self.assertEqual(top_payload["docs"][0]["selected_repeat_indices"], [0, 1])
            self.assertEqual(top_payload["docs"][0]["ranked_repeats"][0]["prefix_dtr"], 1.0)
            self.assertEqual(top_payload["p"], 0.9)
            self.assertEqual(top_payload["rho"], 0.9)
            self.assertEqual(top_summary_json.parent.name, "prefix2_top2of4_g0.5_p0.9")

            self.assertEqual(bottom_payload["summary"]["metrics"]["bottom_maj@2"], 0.0)
            self.assertEqual(
                bottom_payload["docs"][0]["selected_repeat_indices"],
                [2, 3],
            )
            self.assertEqual(
                bottom_summary_json.parent.name,
                "prefix2_bottom2of4_g0.5_p0.9",
            )


class CorrelationPathsTest(unittest.TestCase):
    def test_resolve_paths_defaults_to_p_named_dtr_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "results_2026-03-25T00-00-00.json").write_text(
                json.dumps(results_payload(repeats=4)),
                encoding="utf-8",
            )
            (run_dir / "samples_aime24_custom_2026-03-25T00-00-00.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )

            dtr_path, _results_path, samples_path, output_plot, output_json = resolve_paths(
                Namespace(
                    run_dir=run_dir,
                    dtr_path=None,
                    results_path=None,
                    samples_path=None,
                    prefix_len=None,
                    g=0.5,
                    p=0.9,
                    num_bins=5,
                    output_plot=None,
                    output_json=None,
                )
            )

            self.assertEqual(dtr_path, run_dir / "dtr" / "dtr_g0.5_p0.9.json")
            self.assertEqual(
                samples_path,
                run_dir / "samples_aime24_custom_2026-03-25T00-00-00.jsonl",
            )
            self.assertEqual(
                output_plot,
                run_dir / "dtr_pass1_correlation" / "dtr_pass1_correlation_bins5.png",
            )
            self.assertEqual(
                output_json,
                run_dir / "dtr_pass1_correlation" / "dtr_pass1_correlation_bins5.json",
            )

    def test_resolve_paths_uses_prefix_suffix_for_prefix_corr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "results_2026-03-25T00-00-00.json").write_text(
                json.dumps(results_payload(repeats=4)),
                encoding="utf-8",
            )
            (run_dir / "samples_aime24_custom_2026-03-25T00-00-00.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )

            dtr_path, _results_path, _samples_path, output_plot, output_json = (
                resolve_paths(
                    Namespace(
                        run_dir=run_dir,
                        dtr_path=None,
                        results_path=None,
                        samples_path=None,
                        prefix_len=2,
                        g=0.5,
                        p=0.9,
                        num_bins=5,
                        output_plot=None,
                        output_json=None,
                    )
                )
            )

            self.assertIsNone(dtr_path)
            self.assertEqual(
                output_plot,
                run_dir
                / "dtr_pass1_correlation"
                / "dtr_pass1_correlation_prefix2_bins5.png",
            )
            self.assertEqual(
                output_json,
                run_dir
                / "dtr_pass1_correlation"
                / "dtr_pass1_correlation_prefix2_bins5.json",
            )

    def test_resolve_paths_rejects_prefix_len_with_dtr_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            with self.assertRaisesRegex(
                ValueError, "cannot use --dtr-path together with --prefix-len"
            ):
                resolve_paths(
                    Namespace(
                        run_dir=run_dir,
                        dtr_path=run_dir / "dtr.json",
                        results_path=None,
                        samples_path=None,
                        prefix_len=2,
                        g=0.5,
                        p=0.9,
                        num_bins=5,
                        output_plot=None,
                        output_json=None,
                    )
                )

    def test_correlation_core_path_loads_dtr_bins_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = (
                Path(tmpdir)
                / "results_deep_think_tokens"
                / "aime24_custom"
                / "gpt-oss-120b"
                / "0"
                / "20260325T000000Z"
            )
            write_run_fixture(
                run_dir=run_dir,
                repeats=4,
                sample_rows=[
                    sample_row(
                        doc_id=0,
                        prompt="prompt",
                        completions=["42", "42", "0", "0"],
                    )
                ],
                matrices={
                    (0, 0): deep_divergence(4),
                    (0, 1): medium_divergence(),
                    (0, 2): shallow_divergence(4),
                    (0, 3): shallow_divergence(4),
                },
                num_tokens={
                    (0, 0): 4,
                    (0, 1): 4,
                    (0, 2): 4,
                    (0, 3): 4,
                },
            )
            dtr_path = write_dtr_rows(run_dir)
            args = Namespace(
                run_dir=run_dir,
                dtr_path=None,
                results_path=None,
                samples_path=None,
                prefix_len=None,
                g=0.5,
                p=0.9,
                num_bins=2,
                output_plot=None,
                output_json=None,
            )

            resolved_dtr_path, results_path, samples_path, _output_plot, output_json = (
                resolve_paths(args)
            )
            self.assertEqual(resolved_dtr_path, dtr_path)
            assert resolved_dtr_path is not None

            dtr_by_key = load_dtr_by_key(resolved_dtr_path)
            rows = load_sequence_results(dtr_by_key, samples_path, reasoning_tags=None)
            self.assertEqual([row.dtr for row in rows], [0.0, 0.0, 0.5, 1.0])

            bins = make_bins(rows, 2)
            self.assertEqual([entry.count for entry in bins], [2, 2])
            self.assertAlmostEqual(
                pearson_r(
                    [entry.mean_dtr for entry in bins],
                    [entry.pass_at_1 for entry in bins],
                ),
                1.0,
            )

            summary_path = output_json
            write_summary_json(
                run_dir=run_dir,
                task_name="aime24_custom",
                model_name="openai/gpt-oss-120b",
                dtr_path=resolved_dtr_path,
                results_path=results_path,
                samples_path=samples_path,
                output_path=summary_path,
                rows=rows,
                bins=bins,
                binned_pearson=1.0,
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dtr_path"], str(resolved_dtr_path))
            self.assertEqual(payload["dtr_scope"], "full")
            self.assertIsNone(payload["prefix_len"])
            self.assertEqual(payload["num_sequences"], 4)
            self.assertEqual(payload["num_bins"], 2)
            self.assertEqual(payload["pearson_r_binned"], 1.0)
            self.assertEqual(payload["bins"][0]["mean_dtr"], 0.0)

    def test_correlation_core_path_loads_prefix_dtr_bins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = (
                Path(tmpdir)
                / "results_deep_think_tokens"
                / "aime24_custom"
                / "gpt-oss-120b"
                / "0"
                / "20260325T000000Z"
            )
            write_run_fixture(
                run_dir=run_dir,
                repeats=4,
                sample_rows=[
                    sample_row(
                        doc_id=0,
                        prompt="prompt",
                        completions=["42", "42", "0", "0"],
                    )
                ],
                matrices={
                    (0, 0): deep_divergence(4),
                    (0, 1): medium_divergence(),
                    (0, 2): shallow_divergence(4),
                    (0, 3): shallow_divergence(4),
                },
                num_tokens={
                    (0, 0): 4,
                    (0, 1): 4,
                    (0, 2): 4,
                    (0, 3): 4,
                },
            )

            prefix_dtr_by_key = load_prefix_dtr_by_key(
                run_dir,
                prefix_len=2,
                g=0.5,
                p=0.9,
            )

            self.assertEqual(prefix_dtr_by_key[(0, 0)], 1.0)
            self.assertEqual(prefix_dtr_by_key[(0, 1)], 0.5)
            self.assertEqual(prefix_dtr_by_key[(0, 2)], 0.0)
            self.assertEqual(prefix_dtr_by_key[(0, 3)], 0.0)


if __name__ == "__main__":
    unittest.main()
