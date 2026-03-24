import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from src.dtr.jsd import (
    build_heatmap_subtitle,
    build_heatmap_title,
    build_token_labels,
    escape_token_label,
    heatmap_path,
    render_existing_heatmaps,
    resolve_heatmap_dir,
    resolve_output_dir,
)
from src.plot.jsd_heatmap import pick_tick_indices


class FakeTokenizer:
    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ):
        del skip_special_tokens
        del clean_up_tokenization_spaces
        return {11: "x", 12: "", 13: "\n"}.get(token_ids[0], "?")

    def convert_ids_to_tokens(self, token_ids, *, skip_special_tokens: bool):
        del skip_special_tokens
        return [f"T{token_ids[0]}"]


class JsdHeatmapHelpersTest(unittest.TestCase):
    def test_escape_token_label_and_build_token_labels(self):
        tokenizer = FakeTokenizer()
        response_token_ids = torch.tensor([11, 12, 13], dtype=torch.long)

        labels = build_token_labels(tokenizer, response_token_ids)

        self.assertEqual(labels, ["x", "T12", "\\n"])
        self.assertEqual(escape_token_label("a\tb"), "a\\tb")

    def test_title_subtitle_and_heatmap_path(self):
        self.assertEqual(
            build_heatmap_title(
                doc_id=3, repeat_index=4, hidden_state_mode="normed_normed"
            ),
            "doc=3 rep=4 mode=normed_normed",
        )
        self.assertEqual(
            build_heatmap_subtitle(num_tokens=5, num_layers=7),
            "5 tokens x 7 layers (top=last, bottom=0)",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(
                heatmap_path(Path(tmpdir), 3, 4),
                Path(tmpdir) / "doc3_rep4.png",
            )
            self.assertEqual(
                resolve_heatmap_dir(Path(tmpdir), None),
                (Path(tmpdir) / "heatmaps").resolve(),
            )

    def test_resolve_output_dir_uses_mode_and_token_block_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "run"
            args = Namespace(
                run_dir=run_dir,
                output_dir=None,
                hidden_state_mode="raw_normed",
                token_block_size=64,
            )

            self.assertEqual(
                resolve_output_dir(args),
                (run_dir / "jsd_matrices" / "raw_normed_tb64").resolve(),
            )

    def test_pick_tick_indices_validates_and_limits_label_count(self):
        with self.assertRaisesRegex(ValueError, "max_labels must be positive"):
            pick_tick_indices(5, 0)
        self.assertEqual(pick_tick_indices(5, 1), [0])
        self.assertEqual(pick_tick_indices(5, 2), [0, 4])
        self.assertLessEqual(len(pick_tick_indices(9, 3)), 3)

    def test_render_existing_heatmaps_reads_cached_jsd_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "jsd_matrices"
            heatmap_dir = output_dir / "heatmaps"
            output_dir.mkdir(parents=True)
            torch.save(
                {
                    "doc_id": 3,
                    "repeat_index": 4,
                    "model_path": "dummy/model",
                    "hidden_state_mode": "normed_normed",
                    "response_token_ids": torch.tensor([11, 12], dtype=torch.long),
                    "jsd_matrix": torch.tensor(
                        [[0.1, 0.4], [0.2, 0.8]], dtype=torch.float32
                    ),
                },
                output_dir / "doc3_rep4.pt",
            )

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=FakeTokenizer(),
            ):
                with patch("src.dtr.jsd.render_heatmap") as render_heatmap_mock:
                    render_existing_heatmaps(
                        output_dir=output_dir,
                        heatmap_dir=heatmap_dir,
                        heatmap_cell_width=None,
                        heatmap_cell_height=None,
                        max_token_labels=2,
                        max_layer_labels=2,
                        heatmap_font_size=14,
                        heatmap_vmax=None,
                    )

            render_heatmap_mock.assert_called_once()
            call_kwargs = render_heatmap_mock.call_args.kwargs
            self.assertEqual(call_kwargs["token_labels"], ["x", "T12"])
            self.assertEqual(call_kwargs["output_path"], heatmap_dir / "doc3_rep4.png")


if __name__ == "__main__":
    unittest.main()
