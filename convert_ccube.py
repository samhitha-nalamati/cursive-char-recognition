"""
Converts C-Cube (.chr format) files into this project's canonical layout:

    data/processed/<dataset_name>/images/000001.png, 000002.png, ...
    data/processed/<dataset_name>/labels.csv   ("filename,label" rows)

C-Cube .chr format (per the dataset's documented spec): the file is a
sequence of character records back to back, no separators. Each record is:

    Line 1: 5 integers -> width height baseline_to_upperline upper_extreme lower_extreme
    Next `height` lines: a row of `width` characters, each '0' (white) or '1' (black)
    Final line: the ground-truth class label for this character

NOTE: C-Cube's label alphabet may use different casing/encoding conventions
than this project's canonical A-Z/a-z/0-9 split (e.g. some releases encode
case via a separate field, or as distinct label strings like "a_lower"). Open
one of the source .chr files after downloading and check the label lines
your version produces (a print statement is included below) -- adjust
`normalize_label()` if the raw labels don't already match single characters.
"""
import argparse
import os
import csv
from PIL import Image

from labels import CLASS_TO_IDX


def normalize_label(raw_label: str) -> str:
    """
    Maps a raw C-Cube label string to one of this project's canonical
    62 classes. Adjust this if your downloaded copy uses a different
    label convention than a bare single character.
    """
    raw_label = raw_label.strip()
    if raw_label in CLASS_TO_IDX:
        return raw_label
    # Common alternative: label given as e.g. "a" vs "A" already fine;
    # some distributions prefix/suffix case markers -- extend here as needed.
    raise ValueError(
        f"Unrecognized C-Cube label {raw_label!r}. Inspect the raw .chr file "
        "and extend normalize_label() to map it to one of the 62 canonical classes."
    )


def parse_chr_file(path: str):
    """Yields (PIL.Image, label) for every character record in a .chr file."""
    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        header = lines[i].split()
        if len(header) < 2:
            i += 1
            continue
        width, height = int(header[0]), int(header[1])
        i += 1

        bitmap_rows = lines[i:i + height]
        i += height

        label_line = lines[i]
        i += 1

        # Build a PIL image from the 0/1 bitmap rows.
        img = Image.new("L", (width, height), color=255)
        pixels = img.load()
        for y, row in enumerate(bitmap_rows):
            row = row.strip()
            for x, ch in enumerate(row[:width]):
                if ch == "1":
                    pixels[x, y] = 0  # black stroke on white background

        yield img, label_line.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chr_files", nargs="+", required=True,
                         help="One or more .chr files (e.g. train.chr test.chr)")
    parser.add_argument("--out_dir", default="data/processed/ccube")
    parser.add_argument("--target_size", type=int, default=32,
                         help="Output images are padded to square then resized to this size")
    args = parser.parse_args()

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    count = 0
    skipped = 0
    for chr_path in args.chr_files:
        for img, raw_label in parse_chr_file(chr_path):
            try:
                label = normalize_label(raw_label)
            except ValueError as e:
                skipped += 1
                if skipped <= 5:
                    print(f"Skipping: {e}")
                continue

            # Pad to square (preserve aspect ratio) then resize.
            w, h = img.size
            side = max(w, h)
            square = Image.new("L", (side, side), color=255)
            square.paste(img, ((side - w) // 2, (side - h) // 2))
            square = square.resize((args.target_size, args.target_size), Image.BILINEAR)

            filename = f"{count:06d}.png"
            square.save(os.path.join(img_dir, filename))
            rows.append((filename, label))
            count += 1

    with open(os.path.join(args.out_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"Converted {count} characters -> {args.out_dir} ({skipped} skipped due to unrecognized labels)")


if __name__ == "__main__":
    main()
