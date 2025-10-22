#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path
from typing import Any

import torch


def coerce_label(x: Any) -> float | None:
    """Try to coerce various label representations to a float in {0.0, 1.0}."""
    try:
        # Tensor scalar
        if hasattr(x, "item"):
            return float(x.item())
        # Bool or numeric
        if isinstance(x, (bool, int, float)):
            return float(x)
    except Exception:
        pass
    return None


def read_label_from_pt(path: str) -> float | None:
    obj = torch.load(path, map_location="cpu")
    # Common patterns:
    # 1) (image_tensor, label_tensor)
    if isinstance(obj, tuple) and len(obj) == 2:
        return coerce_label(obj[1])
    # 2) dict with 'label'
    if isinstance(obj, dict) and "label" in obj:
        return coerce_label(obj["label"])
    # 3) raw label tensor
    return coerce_label(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache",
        required=True,
        help="Path to cache directory, e.g. learning_cache/",
    )
    ap.add_argument(
        "--pattern",
        default="*.pt",
        help="Glob pattern for files (default: *.pt)",
    )
    ap.add_argument(
        "--max-list",
        type=int,
        default=30,
        help="Max examples to print per class",
    )
    ap.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional path to write CSV: file,label",
    )
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    if not cache_dir.exists():
        raise SystemExit(f"Cache directory not found: {cache_dir}")

    # Gather files (sorted for stable output)
    files = sorted(glob.glob(os.path.join(str(cache_dir), args.pattern)))
    if not files:
        raise SystemExit(
            f"No files matched pattern {args.pattern} in {cache_dir}"
        )

    zeros, ones, others, errors = [], [], [], []

    for f in files:
        try:
            lab = read_label_from_pt(f)
            if lab is None:
                others.append((f, "unreadable_label"))
                continue
            # Be tolerant to small numeric noise; map to 0/1
            mapped = 1.0 if lab >= 0.5 else 0.0
            if mapped == 0.0:
                zeros.append(f)
            else:
                ones.append(f)

            # If you want to strictly separate exact 0.0/1.0, uncomment:
            # if lab == 0.0: zeros.append(f)
            # elif lab == 1.0: ones.append(f)
            # else: others.append((f, lab))

        except Exception as e:
            errors.append((f, repr(e)))

    # Summary
    total = len(files)
    print(f"\nScanned {total} files in: {cache_dir}")
    print(f"  Label 0.0 (localized):   {len(zeros)}")
    print(f"  Label 1.0 (delocalized): {len(ones)}")
    print(f"  Other/unknown:           {len(others)}")
    print(f"  Read errors:             {len(errors)}")

    # Class ratio
    if (len(zeros) + len(ones)) > 0:
        frac_pos = len(ones) / (len(zeros) + len(ones))
        print(
            f"\nClass ratio (pos=1.0): {frac_pos:.3f}  "
            f"({len(ones)} / {len(zeros) + len(ones)})"
        )

    # Show examples
    def preview(lst, title):
        print(f"\n{title} (showing up to {args.max_list}):")
        for f in lst[: args.max_list]:
            print("  ", Path(f).name)

    preview(zeros, "Examples with label 0.0")
    preview(ones, "Examples with label 1.0")

    if others:
        print("\nFiles with non-binary/unknown labels (first 10):")
        for f, lab in others[:10]:
            print("  ", Path(f).name, "->", lab)

    if errors:
        print("\nFiles with read errors (first 10):")
        for f, e in errors[:10]:
            print("  ", Path(f).name, "->", e)

    # Optional CSV
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            fh.write("file,label\n")
            for f in zeros:
                fh.write(f"{Path(f).name},0\n")
            for f in ones:
                fh.write(f"{Path(f).name},1\n")
            for f, lab in others:
                fh.write(f"{Path(f).name},{lab}\n")
        print(f"\nWrote CSV: {out}")


if __name__ == "__main__":
    main()
