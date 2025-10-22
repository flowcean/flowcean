#!/usr/bin/env python3
import csv
import os
import re
import sys

WORD_STATIC = re.compile(r"\bstatic\b", re.IGNORECASE)
WORD_DYNAMIC = re.compile(r"\bdynamic\b", re.IGNORECASE)

def parse_map_from_info(info_path: str) -> str:
    """Return map name from a line like 'map: warehouse'."""
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().strip().startswith("map:"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return "unknown"

def parse_obstacles_flags(info_path: str) -> tuple[str, str]:
    """
    Return ('x' or 'o', 'x' or 'o') for static,dynamic.
    Primary source: the line starting with 'obstacles:' (case-insensitive).
    Fallback: the second non-empty line of the file.
    """
    static_flag = "o"
    dynamic_flag = "o"

    lines = []
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
    except FileNotFoundError:
        return static_flag, dynamic_flag

    # Prefer an 'obstacles:' line
    obstacle_line = None
    for ln in lines:
        if ln.lower().strip().startswith("obstacles:"):
            obstacle_line = ln
            break

    # Fallback to second non-empty line if needed
    if obstacle_line is None:
        non_empty = [ln for ln in lines if ln.strip()]
        if len(non_empty) >= 2:
            obstacle_line = non_empty[1]

    if obstacle_line:
        text = obstacle_line.lower()
        if WORD_STATIC.search(text):
            static_flag = "x"
        if WORD_DYNAMIC.search(text):
            dynamic_flag = "x"

    return static_flag, dynamic_flag

def parse_delocalizations_from_metadata(metadata_path: str) -> int:
    """
    Extract message_count for topic '/delocalizations' from metadata.yaml,
    using simple text parsing (no PyYAML dependency).
    """
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return 0

    target_idx = None
    for i, line in enumerate(lines):
        if "name:" in line and "/delocalizations" in line:
            target_idx = i
            break

    if target_idx is None:
        return 0

    for j in range(target_idx + 1, len(lines)):
        s = lines[j].strip()
        if s.startswith("- topic_metadata:"):
            break
        if s.startswith("message_count:"):
            try:
                return int(s.split(":", 1)[1].strip())
            except Exception:
                digits = "".join(ch for ch in s if ch.isdigit())
                return int(digits) if digits else 0
    return 0

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 summarize_obstacles_and_delocalizations.py /path/to/wgtl")
        sys.exit(1)

    root = os.path.expanduser(sys.argv[1])
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        sys.exit(1)

    rows = []
    for entry in sorted(os.listdir(root)):
        bag_dir = os.path.join(root, entry)
        if not os.path.isdir(bag_dir):
            continue

        info_path = os.path.join(bag_dir, "info.txt")
        meta_path = os.path.join(bag_dir, "metadata.yaml")
        if not os.path.isfile(meta_path):
            # Not a bag folder
            continue

        map_name = parse_map_from_info(info_path)
        static_flag, dynamic_flag = parse_obstacles_flags(info_path)
        deloc_count = parse_delocalizations_from_metadata(meta_path)

        # One row per bag (even if multiple share the same map)
        rows.append((map_name, static_flag, dynamic_flag, deloc_count))

    out_csv = os.path.join(root, "bag_map_obstacle_delocalizations.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["map", "static", "dynamic", "/delocalizations"])
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to: {out_csv}")

if __name__ == "__main__":
    main()

