import os
import re
from typing import Dict, List, Tuple
from torchvision import datasets

DATASET_ROOT = "/workspace/stdex/Images"
TARGET_CATEGORY_NAME = "n02085620-Chihuahua"   # Subcategory folder name to inspect

def list_subfolders(root: str) -> List[str]:
    return sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    )

def list_subfolders_numeric_sorted(root: str) -> List[str]:
    def key_fn(x):
        m = re.search(r'\d+', x)
        return int(m.group()) if m else 10**9
    return sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
        key=key_fn
    )

def build_one_based_numeric_map(root: str) -> Dict[str, int]:
    """Define the first folder by ascending numeric part as class 1 (1-based)."""
    names = list_subfolders_numeric_sorted(root)
    return {name: i+1 for i, name in enumerate(names)}

def build_imagefolder_train_map(root: str) -> Dict[str, int]:
    """Reproduce ImageFolder's 0-based mapping used in training (alphabetical order)."""
    ds = datasets.ImageFolder(root=root)
    return ds.class_to_idx  # dict: class_name -> index (0-based)

def inspect_mapping(root: str, target_name: str, topk: int = 10) -> None:
    m1 = build_one_based_numeric_map(root)     # 1-based, numeric order
    m0 = build_imagefolder_train_map(root)     # 0-based, alphabetical order (used in training)

    # 1) Print part of the 1-based mapping to verify "the first subcategory = 1"
    names_numeric = list_subfolders_numeric_sorted(root)
    print("=== 1-based (numeric-sorted) preview ===")
    for i, n in enumerate(names_numeric[:topk], start=1):
        print(f"  {i:3d} -> {n}")
    if names_numeric:
        print(f"[Check] First folder under numeric sort = '{names_numeric[0]}' -> class 1")
    print()

    # 2) Print part of the training-time 0-based mapping (alphabetical order)
    names_alpha = list_subfolders(root)
    print("=== 0-based (alphabetical, ImageFolder) preview ===")
    for n in names_alpha[:topk]:
        print(f"  {m0[n]:3d} -> {n}")
    if names_alpha:
        print(f"[Check] First folder under alphabetical sort = '{names_alpha[0]}' -> class 0")
    print()

    # 3) Locate the target category index in both mappings
    if target_name not in m1:
        raise ValueError(f"'{target_name}' is not in the 1-based numeric-order mapping.")
    if target_name not in m0:
        raise ValueError(f"'{target_name}' is not in the 0-based training mapping.")

    idx1 = m1[target_name]          # 1-based (for display / your "positive class ID")
    idx0 = m0[target_name]          # 0-based (**must be used for model forward/backprop/comparison**)

    print("=== Target category ===")
    print(f"  Name:        {target_name}")
    print(f"  1-based idx: {idx1}   (numeric order; for display / your defined 'correct class ID')")
    print(f"  0-based idx: {idx0}   (alphabetical order; use this for model argmax / saliency backprop)")
    print()
    print("NOTE:")
    print("  • If you insist on 'the first subcategory = class 1 (1-based)', that is a **display-layer** convention;")
    print("  • but model inference/training uses **0-based** indices (ImageFolder alphabetical order).")
    print("  • Therefore, when comparing prediction correctness or generating class saliency maps, always convert your 1-based index to the training-time 0-based index.")

if __name__ == "__main__":
    inspect_mapping(DATASET_ROOT, TARGET_CATEGORY_NAME, topk=10)
