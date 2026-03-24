"""DTR-pass@1 상관 plot을 PNG로 렌더링한다."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from PIL import Image
from PIL import ImageDraw

from src.plot.common import FontLike, load_font, measure_text


class CorrelationBin(Protocol):
    """DTR correlation plot에 필요한 최소 bin 인터페이스."""

    @property
    def bin_index(self) -> int: ...

    @property
    def mean_dtr(self) -> float: ...

    @property
    def pass_at_1(self) -> float: ...


BACKGROUND = (255, 255, 255)
AXIS = (54, 61, 70)
GRID = (214, 219, 223)
LINE = (25, 102, 132)
POINT_FILL = (227, 108, 72)
POINT_OUTLINE = (123, 52, 28)
TEXT = (39, 44, 52)


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if not xs or not ys:
        raise ValueError("fit_line requires at least one point")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    if variance_x == 0.0:
        return 0.0, mean_y
    slope = covariance / variance_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def text_size(text: str, font: FontLike) -> tuple[int, int]:
    return measure_text(text, font)


def plot_to_png(
    bins: Sequence[CorrelationBin],
    pearson: float,
    output_path: Path,
    title: str,
) -> None:
    width = 1100
    height = 760
    margin_left = 120
    margin_right = 70
    margin_top = 110
    margin_bottom = 120
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = load_font(32)
    label_font = load_font(22)
    tick_font = load_font(18)
    note_font = load_font(20)

    xs = [entry.mean_dtr for entry in bins]
    ys = [entry.pass_at_1 for entry in bins]
    x_min = min(xs)
    x_max = max(xs)
    x_pad = max((x_max - x_min) * 0.12, 0.02)
    x_low = x_min - x_pad
    x_high = x_max + x_pad

    def x_to_pixel(value: float) -> int:
        ratio = (value - x_low) / (x_high - x_low)
        return round(plot_left + ratio * plot_width)

    def y_to_pixel(value: float) -> int:
        ratio = value
        return round(plot_bottom - ratio * plot_height)

    for step in range(6):
        y_value = step / 5
        y = y_to_pixel(y_value)
        draw.line([(plot_left, y), (plot_right, y)], fill=GRID, width=1)
        tick = f"{y_value:.1f}"
        tick_w, tick_h = text_size(tick, tick_font)
        draw.text((plot_left - tick_w - 14, y - tick_h / 2), tick, fill=TEXT, font=tick_font)

    for step in range(5):
        x_value = x_low + (x_high - x_low) * step / 4
        x = x_to_pixel(x_value)
        draw.line([(x, plot_bottom), (x, plot_top)], fill=GRID, width=1)
        tick = f"{x_value:.3f}"
        tick_w, tick_h = text_size(tick, tick_font)
        draw.text((x - tick_w / 2, plot_bottom + 14), tick, fill=TEXT, font=tick_font)

    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=AXIS, width=3)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=AXIS, width=3)

    points = [(x_to_pixel(x), y_to_pixel(y)) for x, y in zip(xs, ys, strict=True)]
    draw.line(points, fill=LINE, width=5, joint="curve")

    slope, intercept = fit_line(xs, ys)
    fit_start = (x_low, intercept + slope * x_low)
    fit_end = (x_high, intercept + slope * x_high)
    draw.line(
        [
            (x_to_pixel(fit_start[0]), y_to_pixel(fit_start[1])),
            (x_to_pixel(fit_end[0]), y_to_pixel(fit_end[1])),
        ],
        fill=(78, 140, 166),
        width=3,
    )

    point_radius = 10
    for entry, (x, y) in zip(bins, points, strict=True):
        draw.ellipse(
            [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
            fill=POINT_FILL,
            outline=POINT_OUTLINE,
            width=2,
        )
        label = f"Q{entry.bin_index}"
        label_w, label_h = text_size(label, note_font)
        draw.text((x - label_w / 2, y - 34 - label_h / 2), label, fill=TEXT, font=note_font)

    title_w, _ = text_size(title, title_font)
    draw.text(((width - title_w) / 2, 34), title, fill=TEXT, font=title_font)

    pearson_text = f"Pearson r = {pearson:.3f}"
    pearson_w, pearson_h = text_size(pearson_text, note_font)
    pearson_box = [
        (plot_left + 20, plot_top + 20),
        (plot_left + 48 + pearson_w, plot_top + 42 + pearson_h),
    ]
    draw.rounded_rectangle(pearson_box, radius=16, fill=(236, 241, 244))
    draw.text(
        (pearson_box[0][0] + 14, pearson_box[0][1] + 10),
        pearson_text,
        fill=TEXT,
        font=note_font,
    )

    x_label = f"Average DTR ({len(bins)} quantile bins)"
    x_label_w, x_label_h = text_size(x_label, label_font)
    draw.text(
        ((width - x_label_w) / 2, height - 70 - x_label_h / 2),
        x_label,
        fill=TEXT,
        font=label_font,
    )

    y_label = "Pass@1"
    label_image = Image.new("RGBA", (160, 60), (255, 255, 255, 0))
    ImageDraw.Draw(label_image).text((0, 0), y_label, fill=TEXT, font=label_font)
    rotated = label_image.rotate(90, expand=True)
    image.paste(rotated, (32, (height - rotated.height) // 2), rotated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
