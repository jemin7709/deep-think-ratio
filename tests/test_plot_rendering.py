import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import torch
from PIL import Image

from src.plot.dtr_length_scatter import CORRECT_POINT_FILL, POINT_FILL, _expand_axis_range
from src.plot.dtr_length_scatter import plot_to_png as plot_dtr_length_scatter_to_png
from src.plot.dtr_pass1_correlation import plot_to_png
from src.plot.jsd_heatmap import render_heatmap


@dataclass(frozen=True)
class CorrelationBin:
    bin_index: int
    mean_dtr: float
    pass_at_1: float


@dataclass(frozen=True)
class ScatterPoint:
    dtr: float
    response_length: int
    is_correct: bool


def scatter_mean_y_pixel(points: list[ScatterPoint], mean_length: float) -> int:
    height = 760
    margin_top = 110
    margin_bottom = 120
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_height = plot_bottom - plot_top
    lengths = [float(point.response_length) for point in points]
    y_low, y_high = _expand_axis_range(min(lengths), max(lengths), 1.0, clamp_low=0.0)
    ratio = (mean_length - y_low) / (y_high - y_low)
    return round(plot_bottom - ratio * plot_height)


class PlotRenderingTest(unittest.TestCase):
    def test_expand_axis_range_can_clamp_lower_bound(self):
        low, high = _expand_axis_range(895.0, 26095.0, 1.0, clamp_low=0.0)

        self.assertEqual(low, 0.0)
        self.assertGreater(high, 26095.0)

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

    def test_plot_to_png_writes_dtr_length_scatter_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scatter.png"
            points = [
                ScatterPoint(dtr=0.2, response_length=10, is_correct=True),
                ScatterPoint(dtr=0.8, response_length=180, is_correct=False),
            ]

            plot_dtr_length_scatter_to_png(
                points=points,
                pearson=0.98,
                output_path=output_path,
                title="DTR vs Response Length",
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_plot_to_png_colors_correct_vs_incorrect_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scatter.png"
            points = [
                ScatterPoint(dtr=0.2, response_length=10, is_correct=True),
                ScatterPoint(dtr=0.8, response_length=180, is_correct=False),
            ]
            plot_dtr_length_scatter_to_png(
                points=points,
                pearson=0.98,
                output_path=output_path,
                title="DTR vs Response Length",
            )

            image = Image.open(output_path).convert("RGB")
            pixels = {
                image.getpixel((x, y))
                for x in range(image.width)
                for y in range(image.height)
            }
            self.assertIn(CORRECT_POINT_FILL, pixels)
            self.assertIn(POINT_FILL, pixels)

    def test_plot_to_png_draws_group_mean_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scatter.png"
            points = [
                ScatterPoint(dtr=0.1, response_length=10, is_correct=True),
                ScatterPoint(dtr=0.2, response_length=50, is_correct=True),
                ScatterPoint(dtr=0.8, response_length=80, is_correct=False),
                ScatterPoint(dtr=0.9, response_length=100, is_correct=False),
            ]

            plot_dtr_length_scatter_to_png(
                points=points,
                pearson=0.5,
                output_path=output_path,
                title="DTR vs Response Length",
            )

            image = Image.open(output_path).convert("RGB")
            correct_y = scatter_mean_y_pixel(
                points,
                fmean(
                    point.response_length for point in points if point.is_correct
                ),
            )
            incorrect_y = scatter_mean_y_pixel(
                points,
                fmean(
                    point.response_length for point in points if not point.is_correct
                ),
            )

            correct_pixels = sum(
                1
                for x in range(135, image.width - 70)
                if image.getpixel((x, correct_y)) == CORRECT_POINT_FILL
            )
            incorrect_pixels = sum(
                1
                for x in range(135, image.width - 70)
                if image.getpixel((x, incorrect_y)) == POINT_FILL
            )

            self.assertGreater(correct_pixels, 50)
            self.assertGreater(incorrect_pixels, 50)

    def test_plot_to_png_draws_single_group_mean_line_without_other_group(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scatter.png"
            points = [
                ScatterPoint(dtr=0.1, response_length=10, is_correct=True),
                ScatterPoint(dtr=0.9, response_length=50, is_correct=True),
            ]

            plot_dtr_length_scatter_to_png(
                points=points,
                pearson=1.0,
                output_path=output_path,
                title="DTR vs Response Length",
            )

            image = Image.open(output_path).convert("RGB")
            mean_y = scatter_mean_y_pixel(
                points,
                fmean(point.response_length for point in points),
            )
            line_pixels = sum(
                1
                for x in range(135, image.width - 70)
                if image.getpixel((x, mean_y)) == CORRECT_POINT_FILL
            )

            self.assertGreater(line_pixels, 50)

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
