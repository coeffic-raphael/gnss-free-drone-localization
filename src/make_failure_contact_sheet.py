"""Create a contact sheet for retrieval failure cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def open_thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_height: int,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def first_available(row: dict[str, str], names: list[str]) -> str:
    for name in names:
        if name in row:
            return name
    raise KeyError(f"None of these columns were found: {', '.join(names)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("failures_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--max-cases", type=int, default=8)
    args = parser.parse_args()

    with args.failures_csv.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))[: args.max_cases]
    if not rows:
        raise SystemExit("No rows found")
    reference_path_key = first_available(
        rows[0],
        ["motion_viterbi_reference_frame_path", "temporal_reference_frame_path"],
    )
    reference_dataset_key = first_available(
        rows[0],
        ["motion_viterbi_reference_dataset", "temporal_reference_dataset"],
    )
    reference_frame_key = first_available(
        rows[0],
        ["motion_viterbi_reference_frame_count", "temporal_reference_frame_count"],
    )
    rank_key = first_available(rows[0], ["motion_viterbi_rank", "temporal_rank"])
    error_key = first_available(
        rows[0],
        ["motion_viterbi_position_error_m", "temporal_position_error_m", "position_error_m"],
    )

    thumb_size = (360, 210)
    text_width = 380
    row_height = 280
    header_height = 60
    width = thumb_size[0] * 2 + text_width + 60
    height = header_height + row_height * len(rows)
    sheet = Image.new("RGB", (width, height), "#f7f7f2")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    draw.text((20, 20), "Worst retrieval failures: query vs selected reference", font=title_font, fill="#111111")
    draw.text((20, 42), "Left: GNSS-denied query frame | Middle: selected reference frame | Right: metrics", font=font, fill="#555555")

    for idx, row in enumerate(rows):
        top = header_height + idx * row_height
        query_path = Path(row["query_frame_path"])
        reference_path = Path(row[reference_path_key])
        query = open_thumb(query_path, thumb_size)
        reference = open_thumb(reference_path, thumb_size)
        sheet.paste(query, (20, top + 25))
        sheet.paste(reference, (40 + thumb_size[0], top + 25))

        draw.rectangle((20, top + 25, 20 + thumb_size[0], top + 25 + thumb_size[1]), outline="#cccccc")
        draw.rectangle((40 + thumb_size[0], top + 25, 40 + thumb_size[0] * 2, top + 25 + thumb_size[1]), outline="#cccccc")
        draw.text((20, top + 5), f"#{idx + 1} query frame {row.get('query_frame_count')}", font=font, fill="#111111")
        draw.text((40 + thumb_size[0], top + 5), f"reference {row.get(reference_dataset_key)} frame {row.get(reference_frame_key)}", font=font, fill="#111111")

        error = float(row[error_key])
        info = (
            f"error: {error:.2f} m\n"
            f"dino error: {float(row.get('dino_position_error_m', 0.0)):.2f} m\n"
            f"rank: {row.get(rank_key, '?')}\n"
            f"LightGlue matches: {row.get('lg_match_count', '?')}\n"
            f"inliers: {row.get('lg_inlier_count', '?')}\n"
            f"inlier ratio: {float(row.get('lg_inlier_ratio', 0.0)):.3f}"
        )
        text_x = 60 + thumb_size[0] * 2
        text_y = top + 25
        for line in info.splitlines():
            draw_wrapped(draw, line, (text_x, text_y), font, "#222222", text_width - 30, 18)
            text_y += 24

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output_png)
    print(f"wrote: {args.output_png}")


if __name__ == "__main__":
    main()
