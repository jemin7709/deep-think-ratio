"""JSD matrix를 heatmap PNG로 렌더링한다."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from torch import Tensor

from src.plot.common import (
    FontLike,
    RGBColor,
    draw_rotated_text,
    load_font,
    measure_text,
    measure_text_bbox,
    rotated_text_size,
)


TOKEN_LABEL_ANGLE = -45
TOKEN_LABEL_X_OFFSET = 7
PALETTE_STOPS: list[tuple[float, RGBColor]] = [
    (0.0, (255, 232, 138)),
    (0.25, (210, 225, 168)),
    (0.5, (124, 194, 203)),
    (0.75, (55, 116, 181)),
    (1.0, (17, 47, 113)),
]


def choose_cell_width(
    num_tokens: int,
    user_value: int | None,
    font: FontLike,
    token_labels: list[str],
) -> int:
    if user_value is not None:
        return user_value
    widest_rotated_ink = max(
        (rotated_text_size(label, font, angle=TOKEN_LABEL_ANGLE)[0] for label in token_labels),
        default=0,
    )
    width_by_token_count = min(24, math.ceil(1400 / max(num_tokens, 1)))
    return max(min(widest_rotated_ink + 6, 24), width_by_token_count)


def choose_cell_height(num_layers: int, user_value: int | None) -> int:
    if user_value is not None:
        return user_value
    return max(6, min(20, math.ceil(800 / max(num_layers, 1))))


def pick_tick_indices(num_items: int, max_labels: int) -> list[int]:
    if max_labels <= 0:
        raise ValueError(f"max_labels must be positive, got {max_labels}")
    if num_items <= 0:
        return []
    if num_items <= max_labels:
        return list(range(num_items))
    if max_labels == 1:
        return [0]
    if max_labels == 2:
        return [0, num_items - 1]

    last_index = num_items - 1
    interior_slots = max_labels - 2
    indices = [0]
    for slot in range(1, interior_slots + 1):
        position = round(slot * last_index / (interior_slots + 1))
        if position > indices[-1]:
            indices.append(position)
    if indices[-1] != last_index:
        indices.append(last_index)
    return indices


def lerp(start: int, end: int, weight: float) -> int:
    return round(start + (end - start) * weight)


def palette_color(value: float, vmin: float, vmax: float) -> RGBColor:
    if vmax <= vmin:
        return PALETTE_STOPS[-1][1]

    normalized = min(max((value - vmin) / (vmax - vmin), 0.0), 1.0)
    for (start_pos, start_color), (end_pos, end_color) in zip(
        PALETTE_STOPS,
        PALETTE_STOPS[1:],
        strict=False,
    ):
        if normalized <= end_pos:
            weight = (normalized - start_pos) / (end_pos - start_pos)
            return (
                lerp(start_color[0], end_color[0], weight),
                lerp(start_color[1], end_color[1], weight),
                lerp(start_color[2], end_color[2], weight),
            )
    return PALETTE_STOPS[-1][1]


def draw_colorbar(
    image: Image.Image,
    *,
    x0: int,
    y0: int,
    height: int,
    width: int,
    vmin: float,
    vmax: float,
    font: FontLike,
) -> None:
    draw = ImageDraw.Draw(image)
    for row in range(height):
        fraction = 1.0 - row / max(height - 1, 1)
        value = vmin + (vmax - vmin) * fraction
        draw.rectangle(
            [(x0, y0 + row), (x0 + width, y0 + row)],
            fill=palette_color(value, vmin, vmax),
        )

    draw.rectangle([(x0, y0), (x0 + width, y0 + height)], outline=(80, 80, 80))
    draw.text((x0 + width + 10, y0 - 8), f"{vmax:.4f}", font=font, fill=(0, 0, 0))
    draw.text((x0 + width + 10, y0 + height - 8), f"{vmin:.4f}", font=font, fill=(0, 0, 0))
    draw_rotated_text(
        image,
        "JSD",
        (x0 + width + 48, y0 + height // 2 - 16),
        font,
        (0, 0, 0),
    )


def render_heatmap(
    *,
    jsd_matrix: Tensor,
    token_labels: list[str],
    title: str,
    subtitle: str,
    output_path: Path,
    cell_width: int | None = None,
    cell_height: int | None = None,
    max_token_labels: int | None = None,
    max_layer_labels: int = 24,
    font_size: int = 14,
    vmax: float | None = None,
) -> None:
    matrix = jsd_matrix.detach().cpu()
    num_tokens, num_layers = matrix.shape
    layer_labels = [str(layer_index) for layer_index in range(num_layers)]
    font = load_font(font_size)
    title_font = load_font(font_size + 4)
    resolved_cell_width = choose_cell_width(num_tokens, cell_width, font, token_labels)
    resolved_cell_height = choose_cell_height(num_layers, cell_height)

    max_token_label_height = max(
        (rotated_text_size(label, font, angle=TOKEN_LABEL_ANGLE)[1] for label in token_labels),
        default=0,
    )
    max_token_label_width = max(
        (rotated_text_size(label, font, angle=TOKEN_LABEL_ANGLE)[0] for label in token_labels),
        default=0,
    )
    heatmap_width = num_tokens * resolved_cell_width
    heatmap_height = num_layers * resolved_cell_height
    left_margin = max(112, math.ceil(max_token_label_width * 0.7) + 20)
    top_margin = 76
    bottom_margin = max(210, max_token_label_height + 104)
    right_margin = max(132, math.ceil(max_token_label_width * 0.7) + 48)
    colorbar_width = 24

    scratch = Image.new("RGBA", (1, 1), (255, 255, 255, 255))
    draw = ImageDraw.Draw(scratch)

    vmin = 0.0
    vmax_value = float(matrix.max().item()) if vmax is None else vmax
    title_bbox = measure_text_bbox(title, title_font)
    subtitle_bbox = measure_text_bbox(subtitle, font)
    image_width = max(
        left_margin + heatmap_width + right_margin + colorbar_width,
        left_margin + max(title_bbox[2], subtitle_bbox[2]) + 24,
    )
    image = Image.new(
        "RGBA",
        (image_width, top_margin + heatmap_height + bottom_margin),
        (255, 255, 255, 255),
    )
    draw = ImageDraw.Draw(image)
    draw.text((left_margin, 18), title, font=title_font, fill=(0, 0, 0))
    draw.text((left_margin, 44), subtitle, font=font, fill=(70, 70, 70))

    for token_index in range(num_tokens):
        x0 = left_margin + token_index * resolved_cell_width
        x1 = x0 + resolved_cell_width
        for layer_index in range(num_layers):
            y0 = top_margin + (num_layers - 1 - layer_index) * resolved_cell_height
            y1 = y0 + resolved_cell_height
            value = float(matrix[token_index, layer_index].item())
            draw.rectangle([(x0, y0), (x1, y1)], fill=palette_color(value, vmin, vmax_value))

    draw.rectangle(
        [(left_margin, top_margin), (left_margin + heatmap_width, top_margin + heatmap_height)],
        outline=(90, 90, 90),
    )

    token_tick_indices = (
        range(num_tokens)
        if max_token_labels is None
        else pick_tick_indices(num_tokens, max_token_labels)
    )
    for token_index in token_tick_indices:
        x0 = left_margin + token_index * resolved_cell_width
        x = x0 + resolved_cell_width // 2
        draw.line(
            [(x, top_margin + heatmap_height), (x, top_margin + heatmap_height + 6)],
            fill=(60, 60, 60),
            width=1,
        )
        draw_rotated_text(
            image,
            token_labels[token_index],
            (x0 + TOKEN_LABEL_X_OFFSET, top_margin + heatmap_height + 12),
            font,
            (0, 0, 0),
            angle=TOKEN_LABEL_ANGLE,
            trim=True,
        )

    for layer_index in pick_tick_indices(num_layers, max_layer_labels):
        y = top_margin + (num_layers - 1 - layer_index) * resolved_cell_height + resolved_cell_height // 2
        draw.line([(left_margin - 6, y), (left_margin, y)], fill=(60, 60, 60), width=1)
        label_width, label_height = measure_text(layer_labels[layer_index], font)
        draw.text(
            (left_margin - 12 - label_width, y - label_height // 2),
            layer_labels[layer_index],
            font=font,
            fill=(0, 0, 0),
        )

    token_axis_width, _ = measure_text("Token", title_font)
    draw.text(
        (
            left_margin + heatmap_width // 2 - token_axis_width // 2,
            top_margin + heatmap_height + bottom_margin - 36,
        ),
        "Token",
        font=title_font,
        fill=(0, 0, 0),
    )
    draw_rotated_text(
        image,
        "Layer (top=last)",
        (28, top_margin + heatmap_height // 2 - 58),
        title_font,
        (0, 0, 0),
    )
    draw_colorbar(
        image,
        x0=left_margin + heatmap_width + 28,
        y0=top_margin,
        height=heatmap_height,
        width=colorbar_width,
        vmin=vmin,
        vmax=vmax_value,
        font=font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
