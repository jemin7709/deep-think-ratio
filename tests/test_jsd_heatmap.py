import tempfile
import unittest
from pathlib import Path

import torch

from src.dtr.jsd import (
    build_heatmap_subtitle,
    build_heatmap_title,
    build_token_labels,
    escape_token_label,
    heatmap_path,
)


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
            build_heatmap_title(doc_id=3, repeat_index=4, hidden_state_mode="normed_normed"),
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


if __name__ == "__main__":
    unittest.main()
