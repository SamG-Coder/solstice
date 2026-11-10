"""Tiles rendered views into one sheet, to look at together. Plain Python, with Pillow.

    python tools/blender/gt3r/sheet.py FOLDER OUT.png [name ...] [--columns N] [--width PX]
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    args = sys.argv[1:]
    columns, width = 3, 640
    for flag in ("--columns", "--width"):
        if flag in args:
            i = args.index(flag)
            value = int(args[i + 1])
            del args[i:i + 2]
            if flag == "--columns":
                columns = value
            else:
                width = value
    folder, out, names = Path(args[0]), args[1], args[2:]
    tiles = []
    for name in names:
        image = Image.open(folder / f"{name}.png").convert("RGB")
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.LANCZOS)
        ImageDraw.Draw(image).text((6, 4), name, fill=(255, 255, 0))
        tiles.append(image)
    rows = (len(tiles) + columns - 1) // columns
    tall = max(t.height for t in tiles)
    sheet = Image.new("RGB", (columns * width, rows * tall), (20, 20, 20))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * width, (i // columns) * tall))
    sheet.save(out)
    print(f"sheet: {out} {sheet.size}")


main()
