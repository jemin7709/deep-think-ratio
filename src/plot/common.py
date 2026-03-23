"""Pillow 기반 plot에서 공통으로 쓰는 텍스트 렌더링 유틸."""

from __future__ import annotations

from typing import TypeAlias

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


RGBColor: TypeAlias = tuple[int, int, int]
FontLike: TypeAlias = ImageFont.FreeTypeFont | ImageFont.ImageFont
TEXT_PADDING = 8
ROTATED_TEXT_SUPERSAMPLE = 3


def load_font(size: int) -> FontLike:
    return ImageFont.truetype("DejaVuSans.ttf", size=size)


def measure_text_bbox(text: str, font: FontLike) -> tuple[int, int, int, int]:
    scratch = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    bbox = ImageDraw.Draw(scratch).textbbox((0, 0), text, font=font)
    return (
        int(bbox[0]),
        int(bbox[1]),
        int(bbox[2]),
        int(bbox[3]),
    )


def measure_text(text: str, font: FontLike) -> tuple[int, int]:
    left, top, right, bottom = measure_text_bbox(text, font)
    return right - left, bottom - top


def build_rotated_text_image(
    text: str,
    font: FontLike,
    fill: RGBColor,
    angle: int,
    *,
    trim: bool,
) -> Image.Image:
    supersample = ROTATED_TEXT_SUPERSAMPLE if angle % 90 else 1
    scaled_font = (
        font.font_variant(size=font.size * supersample)
        if isinstance(font, ImageFont.FreeTypeFont)
        else font
    )
    left, top, right, bottom = measure_text_bbox(text, scaled_font)
    padding = TEXT_PADDING * supersample
    text_image = Image.new(
        "RGBA",
        (right - left + padding * 2, bottom - top + padding * 2),
        (255, 255, 255, 0),
    )
    ImageDraw.Draw(text_image).text(
        (padding - left, padding - top),
        text,
        font=scaled_font,
        fill=fill,
    )
    rotated = text_image.rotate(
        angle,
        expand=True,
        resample=Image.Resampling.BICUBIC,
    )
    if trim:
        alpha_bbox = rotated.getchannel("A").getbbox()
        if alpha_bbox is not None:
            rotated = rotated.crop(alpha_bbox)
    return downsample_rotated_text_image(rotated, supersample)


def downsample_rotated_text_image(image: Image.Image, supersample: int) -> Image.Image:
    if supersample == 1:
        return image
    return image.resize(
        (
            max(1, round(image.width / supersample)),
            max(1, round(image.height / supersample)),
        ),
        resample=Image.Resampling.LANCZOS,
    )


def rotated_text_size(text: str, font: FontLike, angle: int = 90) -> tuple[int, int]:
    return build_rotated_text_image(
        text,
        font,
        (0, 0, 0),
        angle,
        trim=True,
    ).size


def draw_rotated_text(
    image: Image.Image,
    text: str,
    position: tuple[int, int],
    font: FontLike,
    fill: RGBColor,
    *,
    angle: int = 90,
    trim: bool = False,
) -> None:
    rotated = build_rotated_text_image(
        text,
        font,
        fill,
        angle,
        trim=trim,
    )
    image.alpha_composite(rotated, dest=position)
