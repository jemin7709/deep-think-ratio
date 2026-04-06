"""문항 난이도 산출물의 scatter / bucket plot을 PNG로 렌더링한다."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image
from PIL import ImageDraw

from src.plot.common import load_font, measure_text


class DifficultyPoint(Protocol):
    """scatter plot이 필요한 최소 인터페이스."""

    @property
    def difficulty_score(self) -> float: ...

    @property
    def mean_dtr(self) -> float: ...

    @property
    def difficulty_bucket(self) -> str: ...


class BucketStat(Protocol):
    """bucket 요약 plot이 필요한 인터페이스."""

    @property
    def bucket(self) -> str: ...

    @property
    def num_problems(self) -> int | float: ...

    @property
    def mean_accuracy(self) -> float: ...

    @property
    def mean_dtr(self) -> float: ...

    @property
    def mean_response_length(self) -> float: ...


BACKGROUND = (255, 255, 255)
AXIS = (54, 61, 70)
GRID = (214, 219, 223)
POINT_FILL = (227, 108, 72)
POINT_OUTLINE = (123, 52, 28)
TEXT = (39, 44, 52)
NOTE_FILL = (236, 241, 244)
EASY_FILL = (53, 128, 81)
MEDIUM_FILL = (214, 163, 64)
HARD_FILL = (199, 59, 59)
EASY_BAND_FILL = (227, 241, 231)
MEDIUM_BAND_FILL = (248, 240, 214)
HARD_BAND_FILL = (247, 224, 224)
REFERENCE_LINE = (126, 134, 143)
ACCURACY_BAR_FILL = (67, 122, 201)


@dataclass(frozen=True)
class DifficultyBand:
    bucket: str
    start: float
    end: float
    fill: tuple[int, int, int]
    short_label: str
    long_label: str


def _bucket_color(bucket: str) -> tuple[int, int, int]:
    if bucket == "easy":
        return EASY_FILL
    if bucket == "hard":
        return HARD_FILL
    return MEDIUM_FILL


def difficulty_bands() -> tuple[DifficultyBand, ...]:
    return (
        DifficultyBand(
            bucket="easy",
            start=0.0,
            end=0.25,
            fill=EASY_BAND_FILL,
            short_label="easy",
            long_label="easy | acc >= 0.75",
        ),
        DifficultyBand(
            bucket="medium",
            start=0.25,
            end=0.75,
            fill=MEDIUM_BAND_FILL,
            short_label="medium",
            long_label="medium | 0.25 < acc < 0.75",
        ),
        DifficultyBand(
            bucket="hard",
            start=0.75,
            end=1.0,
            fill=HARD_BAND_FILL,
            short_label="hard",
            long_label="hard | acc <= 0.25",
        ),
    )


def difficulty_scatter_x_label() -> str:
    return "Difficulty Score (= 1 - accuracy, 0=easiest, 1=hardest)"


def difficulty_note_text(spearman: float) -> str:
    return f"Spearman ρ = {spearman:.3f}"


def _expand_axis_range(
    low: float,
    high: float,
    minimum_pad: float,
    *,
    clamp_low: float | None = None,
    clamp_high: float | None = None,
) -> tuple[float, float]:
    span = high - low
    pad = max(span * 0.08, minimum_pad)
    if span == 0.0:
        expanded_low, expanded_high = low - pad, high + pad
    else:
        expanded_low, expanded_high = low - pad, high + pad
    if clamp_low is not None:
        expanded_low = max(clamp_low, expanded_low)
    if clamp_high is not None:
        expanded_high = min(clamp_high, expanded_high)
    if expanded_low == expanded_high:
        expanded_low -= minimum_pad
        expanded_high += minimum_pad
    return expanded_low, expanded_high


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
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


def _render_title(draw: ImageDraw.ImageDraw, width: int, title: str, font) -> None:
    title_w, _ = measure_text(title, font)
    draw.text(
        ((width - title_w) / 2, 34),
        title,
        fill=TEXT,
        font=font,
    )


def plot_scatter_to_png(
    points: Sequence[DifficultyPoint],
    spearman: float,
    output_path: Path,
    title: str,
) -> None:
    if not points:
        raise ValueError("plot_scatter_to_png requires at least one point")

    width = 1100
    height = 760
    margin_left = 135
    margin_right = 70
    margin_top = 170
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

    xs = [point.difficulty_score for point in points]
    ys = [point.mean_dtr for point in points]
    x_low, x_high = _expand_axis_range(min(xs), max(xs), 0.05, clamp_low=0.0, clamp_high=1.0)
    y_low, y_high = _expand_axis_range(min(ys), max(ys), 0.02, clamp_low=0.0, clamp_high=1.0)

    def x_to_pixel(value: float) -> int:
        ratio = (value - x_low) / (x_high - x_low)
        return round(plot_left + ratio * plot_width)

    def y_to_pixel(value: float) -> int:
        ratio = (value - y_low) / (y_high - y_low)
        return round(plot_bottom - ratio * plot_height)

    for band in difficulty_bands():
        band_left = x_to_pixel(band.start)
        band_right = x_to_pixel(band.end)
        draw.rectangle(
            [(band_left, plot_top), (band_right, plot_bottom)],
            fill=band.fill,
        )

    for reference_x in (0.0, 0.25, 0.75, 1.0):
        x = x_to_pixel(reference_x)
        draw.line([(x, plot_top), (x, plot_bottom)], fill=REFERENCE_LINE, width=2)

    for step in range(6):
        y_value = y_low + (y_high - y_low) * step / 5
        y = y_to_pixel(y_value)
        draw.line([(plot_left, y), (plot_right, y)], fill=GRID, width=1)
        tick = f"{y_value:.3f}"
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
        draw.text(
            (x - tick_w / 2, plot_bottom + 14),
            tick,
            fill=TEXT,
            font=tick_font,
        )

    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=AXIS, width=3)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=AXIS, width=3)

    top_axis_y = plot_top - 42
    draw.line([(plot_left, top_axis_y), (plot_right, top_axis_y)], fill=AXIS, width=2)
    for difficulty in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x_to_pixel(difficulty)
        draw.line([(x, top_axis_y), (x, top_axis_y - 7)], fill=AXIS, width=2)
        accuracy_tick = f"{1.0 - difficulty:.2f}"
        tick_w, tick_h = measure_text(accuracy_tick, tick_font)
        draw.text(
            (x - tick_w / 2, top_axis_y - 12 - tick_h),
            accuracy_tick,
            fill=TEXT,
            font=tick_font,
        )
    top_label = "Accuracy"
    top_label_w, top_label_h = measure_text(top_label, label_font)
    draw.text(
        ((width - top_label_w) / 2, top_axis_y - 44 - top_label_h / 2),
        top_label,
        fill=TEXT,
        font=label_font,
    )

    for band in difficulty_bands():
        label_w, label_h = measure_text(band.long_label, tick_font)
        center_x = (x_to_pixel(band.start) + x_to_pixel(band.end)) / 2
        draw.text(
            (center_x - label_w / 2, plot_top - 22 - label_h / 2),
            band.long_label,
            fill=TEXT,
            font=tick_font,
        )

    slope, intercept = _fit_line(xs, ys)
    fit_start = (x_low, intercept + slope * x_low)
    fit_end = (x_high, intercept + slope * x_high)
    draw.line(
        [
            (x_to_pixel(fit_start[0]), y_to_pixel(fit_start[1])),
            (x_to_pixel(fit_end[0]), y_to_pixel(fit_end[1])),
        ],
        fill=(47, 111, 132),
        width=4,
    )

    point_radius = 7
    for point in points:
        x = x_to_pixel(point.difficulty_score)
        y = y_to_pixel(point.mean_dtr)
        color = _bucket_color(point.difficulty_bucket)
        draw.ellipse(
            [
                (x - point_radius, y - point_radius),
                (x + point_radius, y + point_radius),
            ],
            fill=color,
            outline=POINT_OUTLINE,
            width=2,
        )

    _render_title(draw, width, title, title_font)

    note = difficulty_note_text(spearman)
    note_width, note_height = measure_text(note, note_font)
    note_box = [
        (plot_left + 20, plot_top + 20),
        (plot_left + 44 + note_width, plot_top + 42 + note_height),
    ]
    draw.rounded_rectangle(note_box, radius=16, fill=NOTE_FILL)
    draw.text(
        (note_box[0][0] + 12, note_box[0][1] + 10),
        note,
        fill=TEXT,
        font=note_font,
    )

    x_label = difficulty_scatter_x_label()
    x_label_w, x_label_h = measure_text(x_label, label_font)
    draw.text(
        ((width - x_label_w) / 2, height - 70 - x_label_h / 2),
        x_label,
        fill=TEXT,
        font=label_font,
    )

    y_label = "Mean DTR"
    y_label_image = Image.new("RGBA", (120, 50), (255, 255, 255, 0))
    ImageDraw.Draw(y_label_image).text((0, 0), y_label, fill=TEXT, font=label_font)
    rotated = y_label_image.rotate(90, expand=True)
    image.alpha_composite(rotated, (25, (height - rotated.height) // 2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def plot_bucket_summary_to_png(
    bucket_summaries: Sequence[BucketStat],
    output_path: Path,
    title: str,
) -> None:
    if not bucket_summaries:
        raise ValueError("plot_bucket_summary_to_png requires at least one bucket")

    width = 1100
    height = 760
    margin_left = 140
    margin_right = 70
    margin_top = 110
    margin_bottom = 140
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

    summary_count = len(bucket_summaries)
    buckets = list(bucket_summaries)
    xs = [index * 3 for index in range(summary_count)]
    group_width = plot_width / max(1, len(xs))
    bar_width = group_width / 3
    max_height = plot_height - 20

    y_min = 0.0
    y_max = 1.0
    def y_to_pixel(value: float) -> int:
        ratio = (value - y_min) / (y_max - y_min)
        return round(plot_bottom - ratio * max_height)

    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=AXIS, width=3)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=AXIS, width=3)

    for step in range(6):
        y_value = y_max * (1.0 - step / 5)
        y = y_to_pixel(y_value)
        draw.line([(plot_left, y), (plot_right, y)], fill=GRID, width=1)
        tick = f"{y_value:.1f}"
        tick_w, tick_h = measure_text(tick, tick_font)
        draw.text(
            (plot_left - tick_w - 14, y - tick_h / 2),
            tick,
            fill=TEXT,
            font=tick_font,
        )

    for bucket_index, bucket in enumerate(buckets):
        center_x = plot_left + bucket_index * group_width + group_width / 2
        x0 = center_x - group_width / 2 + 10
        x_acc = x0 + 10
        x_dtr = x_acc + bar_width
        acc_height = (bucket.mean_accuracy) * max_height
        dtr_height = (bucket.mean_dtr) * max_height
        num_height = min(0.8, bucket.mean_response_length / max(
            1.0,
            max((entry.mean_response_length for entry in buckets),
            default=1.0),
        )) * (max_height / 2)

        acc_top = y_to_pixel(acc_height / max_height)
        dtr_top = y_to_pixel(dtr_height / max_height)
        num_top = y_to_pixel(0.6 + num_height / max_height)

        x_mid = x0 + group_width - 20
        draw.rectangle(
            [(x_acc, acc_top), (x_acc + bar_width, plot_bottom)],
            fill=ACCURACY_BAR_FILL,
            outline=POINT_OUTLINE,
        )
        draw.rectangle(
            [(x_dtr, dtr_top), (x_dtr + bar_width, plot_bottom)],
            fill=(47, 111, 132),
            outline=POINT_OUTLINE,
        )
        draw.rectangle(
            [(x_mid, num_top), (x_mid + (bar_width * 0.6), plot_bottom)],
            fill=(123, 52, 28),
            outline=POINT_OUTLINE,
        )

        label = bucket.bucket
        label_w, label_h = measure_text(label, tick_font)
        draw.text(
            (center_x - label_w / 2, plot_bottom + 24),
            label,
            fill=TEXT,
            font=tick_font,
        )
        count_text = str(bucket.num_problems)
        count_w, count_h = measure_text(count_text, tick_font)
        draw.text(
            (center_x - count_w / 2, plot_bottom + 48),
            count_text,
            fill=TEXT,
            font=tick_font,
        )

        draw.text(
            (x_acc, acc_top - 16),
            f"{bucket.mean_accuracy:.2f}",
            fill=TEXT,
            font=tick_font,
        )
        draw.text(
            (x_dtr, dtr_top - 16),
            f"{bucket.mean_dtr:.2f}",
            fill=TEXT,
            font=tick_font,
        )
        draw.text(
            (x_mid, num_top - 16),
            f"{bucket.mean_response_length:.0f}",
            fill=TEXT,
            font=tick_font,
        )

    _render_title(draw, width, title, title_font)

    legend = "blue: mean_accuracy | teal: mean_dtr | dark red: relative mean_response_length"
    legend_w, legend_h = measure_text(legend, tick_font)
    legend_box = [
        (plot_left + 20, plot_top - 48),
        (plot_left + 34 + legend_w, plot_top - 18 + legend_h),
    ]
    draw.rounded_rectangle(legend_box, radius=12, fill=NOTE_FILL)
    draw.text(
        (legend_box[0][0] + 10, legend_box[0][1] + 4),
        legend,
        fill=TEXT,
        font=tick_font,
    )

    x_label = "Difficulty Bucket"
    x_label_w, x_label_h = measure_text(x_label, label_font)
    draw.text(
        ((width - x_label_w) / 2, height - 90 - x_label_h / 2),
        x_label,
        fill=TEXT,
        font=label_font,
    )

    y_label = "Metric"
    y_label_image = Image.new("RGBA", (90, 50), (255, 255, 255, 0))
    ImageDraw.Draw(y_label_image).text((0, 0), y_label, fill=TEXT, font=label_font)
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(rotated, (32, (height - rotated.height) // 2), rotated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
