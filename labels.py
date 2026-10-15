"""
Canonical 62-class label space shared by every dataset in this project:
digits 0-9, lowercase a-z, uppercase A-Z.

All dataset converters map their native labels into this space so C-Cube,
CHoiCe, and EMNIST can be trained on individually or combined.
"""

CLASSES = list("0123456789") + list("abcdefghijklmnopqrstuvwxyz") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)  # 62

assert NUM_CLASSES == 62
