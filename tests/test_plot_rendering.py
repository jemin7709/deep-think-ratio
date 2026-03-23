import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch

from src.plot.dtr_pass1_correlation import plot_to_png
from src.plot.jsd_heatmap import render_heatmap


@dataclass(frozen=True)
class CorrelationBin:
    bin_index: int
    mean_dtr: float
    pass_at_1: float


class PlotRenderingTest(unittest.TestCase):
    def test_plot_to_png_writes_dtr_correlation_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "plot.png"
            bins = [
                CorrelationBin(bin_index=1, mean_dtr=0.2, pass_at_1=0.4),
                CorrelationBin(bin_index=2, mean_dtr=0.6, pass_at_1=0.8),
            ]

            plot_to_png(
                bins=bins,
                pearson=1.0,
                output_path=output_path,
                title="DTR vs Pass@1",
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_render_heatmap_writes_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "heatmap.png"

            render_heatmap(
                jsd_matrix=torch.tensor([[0.1, 0.4], [0.2, 0.8]], dtype=torch.float32),
                token_labels=["tok0", "tok1"],
                title="doc=0 rep=0 mode=normed_normed",
                subtitle="2 tokens x 2 layers (top=last, bottom=0)",
                output_path=output_path,
                max_token_labels=2,
                max_layer_labels=2,
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
