# Cursive Character Classifier (C-Cube + CHoiCe)

A CNN classifier for isolated handwritten characters (62 classes: `0-9`,
`a-z`, `A-Z`), built to combine three sources:

| Dataset | What it actually is | Size |
|---|---|---|
| **C-Cube** | Real cursive characters, manually segmented from cursive handwritten words | 57,293 chars |
| **CHoiCe** | Complex handwritten characters from the CEDAR dataset, cursive-focused | 62 classes, ≥40 imgs/class |

## Pipeline

```
raw dataset (3 different formats)
  -> convert_*.py           normalizes to: images/*.png (32x32 grayscale) + labels.csv
  -> dataset.py              loads any combination of processed folders as one dataset
  -> model.py (CharCNN)      small CNN classifier
  -> train.py / infer.py
```

## 1. Get the raw data

- **C-Cube**: https://www.idiap.ch/en/dataset/ccc — download the `.chr` files (train/test).
- **CHoiCe**: https://github.com/skyatmoon/CHoiCe-Dataset — clone/download; contains `V0.3/data`, `V0.3/data-bin`, `label.txt`.

## 2. Convert each source to the canonical format

```bash
python convert_ccube.py --chr_files path/to/train.chr path/to/test.chr \
    --out_dir data/processed/ccube

python convert_choice.py --choice_root path/to/CHoiCe-Dataset \
    --out_dir data/processed/choice

```

**Important — check label formats before trusting the conversion:**
- `convert_ccube.py`'s `normalize_label()` assumes C-Cube's raw label
  lines are already single characters (`a`, `A`, `7`, etc.). Open one
  downloaded `.chr` file and check the label lines match this — if C-Cube
  encodes case or class differently in your release, extend
  `normalize_label()` accordingly (the script will raise a clear error
  telling you exactly which raw label it couldn't map).
- `convert_choice.py` reads `label.txt` directly rather than assuming
  folder order, so it should adapt automatically to whatever mapping your
  copy uses — but do a spot check on a couple of output PNGs vs. their
  assigned label after conversion.

## 3. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

## 4. Train

```bash
# Cursive-only (recommended primary run)
python train.py --datasets data/processed/ccube data/processed/choice --epochs 30

```

Checkpoints save to `checkpoints/last.pt` / `checkpoints/best.pt` (best =
highest validation accuracy). At the end of training, the script prints
the worst-performing classes — for cursive this is usually genuinely
informative, since it'll surface classic cursive confusions like `n`/`u`,
`a`/`o`, or `C`/`c` rather than random noise.

## 5. Run inference

```bash
python infer.py --checkpoint checkpoints/best.pt --image samples/my_char.png --topk 3
```

Expects a single, roughly-centered character on a light background.

## Suggested experiments

Since you're combining multiple sources, the interesting project-level
question isn't just "what's my accuracy" but **how do the sources
interact**:
   Train on C-Cube alone, then CHoiCe alone, then both combined — compare
   validation accuracy and the worst-class list. Do they make similar
   mistakes (suggesting a genuinely hard cursive ambiguity) or different
   ones (suggesting dataset-specific artifacts)?
