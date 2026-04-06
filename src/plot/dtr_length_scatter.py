"""DTR-response length scatter plot을 PNG로 렌더링한다."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Protocol

from PIL import Image
from PIL import ImageDraw

from src.plot.common import draw_rotated_text, load_font, measure_text


class ScatterPoint(Protocol):
    """DTR-length scatter plot에 필요한 최소 point 인터페이스."""

    @property
    def dtr(self) -> float: ...

    @property
    def response_length(self) -> int: ...


BACKGROUND = (255, 255, 255)
AXIS = (54, 61, 70)
GRID = (214, 219, 223)
FIT_LINE = (47, 111, 132)
CORRECT_POINT_FILL = (67, 122, 201)
CORRECT_POINT_OUTLINE = (32, 72, 142)
POINT_FILL = (227, 108, 72)
POINT_OUTLINE = (123, 52, 28)
TEXT = (39, 44, 52)
NOTE_FILL = (236, 241, 244)


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


def _expand_axis_range(
    low: float,
    high: float,
    minimum_pad: float,
    *,
    clamp_low: float | None = None,
) -> tuple[float, float]:
    span = high - low
    pad = max(span * 0.08, minimum_pad)
    if span == 0.0:
        expanded_low, expanded_high = low - pad, high + pad
    else:
        expanded_low, expanded_high = low - pad, high + pad
    if clamp_low is not None:
        expanded_low = max(clamp_low, expanded_low)
    return expanded_low, expanded_high


def _draw_horizontal_dashed_line(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    left: int,
    right: int,
    fill: tuple[int, int, int],
    width: int = 3,
    dash: int = 14,
    gap: int = 8,
) -> None:
    x = left
    while x <= right:
        draw.line([(x, y), (min(x + dash - 1, right), y)], fill=fill, width=width)
        x += dash + gap


def _group_mean_lengths(
    points: Sequence[ScatterPoint],
) -> list[tuple[str, float, tuple[int, int, int]]]:
    correct_lengths = [
        float(point.response_length)
        for point in points
        if getattr(point, "is_correct", None) is True
    ]
    incorrect_lengths = [
        float(point.response_length)
        for point in points
        if getattr(point, "is_correct", None) is False
    ]
    mean_rows: list[tuple[str, float, tuple[int, int, int]]] = []
    if correct_lengths:
        mean_rows.append(("correct mean", fmean(correct_lengths), CORRECT_POINT_FILL))
    if incorrect_lengths:
        mean_rows.append(("incorrect mean", fmean(incorrect_lengths), POINT_FILL))
    return mean_rows


def plot_to_png(
    points: Sequence[ScatterPoint],
    pearson: float,
    output_path: Path,
    title: str,
    y_label: str = "Response Length (tokens)",
) -> None:
    if not points:
        raise ValueError("plot_to_png requires at least one point")

    width = 1100
    height = 760
    margin_left = 135
    margin_right = 70
    margin_top = 110
    margin_bottom = 120
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    image = Image.new("RGBA", (width, height), BACKGROUND + (255,))
    draw = ImageDraw.Draw(image)
    title_font = load_font(32)
    label_font = load_font(22)
    tick_font = load_font(18)
    note_font = load_font(20)

    xs = [point.dtr for point in points]
    ys = [float(point.response_length) for point in points]
    x_low, x_high = _expand_axis_range(min(xs), max(xs), 0.02)
    y_low, y_high = _expand_axis_range(min(ys), max(ys), 1.0, clamp_low=0.0)

    def x_to_pixel(value: float) -> int:
        ratio = (value - x_low) / (x_high - x_low)
        return round(plot_left + ratio * plot_width)

    def y_to_pixel(value: float) -> int:
        ratio = (value - y_low) / (y_high - y_low)
        return round(plot_bottom - ratio * plot_height)

    for step in range(6):
        y_value = y_low + (y_high - y_low) * step / 5
        y = y_to_pixel(y_value)
        draw.line([(plot_left, y), (plot_right, y)], fill=GRID, width=1)
        tick = f"{y_value:.0f}"
        tick_w, tick_h = measure_text(tick, tick_font)
        draw.text(
            (plot_left - tick_w - 14, y - tick_h / 2),
            tick,
            fill=TEXT,
            font=tick_font,
        )

    for step in range(5):
        x_value = x_low + (x_high - x_low) * step / 4
        x = x_to_pixel(x_value)
        draw.line([(x, plot_bottom), (x, plot_top)], fill=GRID, width=1)
        tick = f"{x_value:.3f}"
        tick_w, tick_h = measure_text(tick, tick_font)
        draw.text((x - tick_w / 2, plot_bottom + 14), tick, fill=TEXT, font=tick_font)

    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=AXIS, width=3)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=AXIS, width=3)

    slope, intercept = fit_line(xs, ys)
    fit_start = (x_low, intercept + slope * x_low)
    fit_end = (x_high, intercept + slope * x_high)
    draw.line(
        [
            (x_to_pixel(fit_start[0]), y_to_pixel(fit_start[1])),
            (x_to_pixel(fit_end[0]), y_to_pixel(fit_end[1])),
        ],
        fill=FIT_LINE,
        width=4,
    )

    mean_lines = _group_mean_lengths(points)
    label_positions: list[int] = []
    for label, mean_length, color in mean_lines:
        mean_y = y_to_pixel(mean_length)
        _draw_horizontal_dashed_line(
            draw,
            y=mean_y,
            left=plot_left,
            right=plot_right,
            fill=color,
        )

        label_w, label_h = measure_text(label, tick_font)
        label_x = plot_right - label_w - 16
        label_y = max(plot_top + 8, mean_y - label_h - 6)
        if label_positions and abs(label_y - label_positions[-1]) < label_h + 6:
            label_y = min(plot_bottom - label_h - 8, label_positions[-1] + label_h + 6)
        label_positions.append(label_y)
        label_box = [
            (label_x - 8, label_y - 3),
            (label_x + label_w + 8, label_y + label_h + 3),
        ]
        draw.rounded_rectangle(label_box, radius=8, fill=NOTE_FILL)
        draw.text((label_x, label_y), label, fill=color, font=tick_font)

    # 회귀선은 분포 경향만 보조적으로 보여주고, 점의 정오 색상은 직접 읽혀야 한다.
    point_radius = 6
    for point in points:
        x = x_to_pixel(point.dtr)
        y = y_to_pixel(float(point.response_length))
        is_correct = getattr(point, "is_correct", None) is True
        draw.ellipse(
            [
                (x - point_radius, y - point_radius),
                (x + point_radius, y + point_radius),
            ],
            fill=CORRECT_POINT_FILL if is_correct else POINT_FILL,
            outline=CORRECT_POINT_OUTLINE if is_correct else POINT_OUTLINE,
            width=2,
        )

    title_w, _ = measure_text(title, title_font)
    draw.text(((width - title_w) / 2, 34), title, fill=TEXT, font=title_font)

    pearson_text = f"Pearson r = {pearson:.3f}"
    pearson_w, pearson_h = measure_text(pearson_text, note_font)
    pearson_box = [
        (plot_left + 20, plot_top + 20),
        (plot_left + 48 + pearson_w, plot_top + 42 + pearson_h),
    ]
    draw.rounded_rectangle(pearson_box, radius=16, fill=NOTE_FILL)
    draw.text(
        (pearson_box[0][0] + 14, pearson_box[0][1] + 10),
        pearson_text,
        fill=TEXT,
        font=note_font,
    )

    x_label = "DTR"
    x_label_w, x_label_h = measure_text(x_label, label_font)
    draw.text(
        ((width - x_label_w) / 2, height - 70 - x_label_h / 2),
        x_label,
        fill=TEXT,
        font=label_font,
    )

    draw_rotated_text(
        image,
        y_label,
        (38, (height - 255) // 2),
        label_font,
        TEXT,
        angle=90,
        trim=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
