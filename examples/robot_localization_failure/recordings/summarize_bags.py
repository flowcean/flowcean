#!/usr/bin/env python3
import os
import sys
import csv

def parse_info_txt(info_path: str):
    """
    Returns (map_name, static_flag, dynamic_flag)
    static_flag/dynamic_flag are 'x' if present in obstacles line, else 'o'.
    """
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            # Keep only non-empty, stripped lines
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Warning: could not read {info_path}: {e}")
        return ("", "o", "o")

    # Defaults
    map_name = ""
    static_flag = "o"
    dynamic_flag = "o"

    # 1) Map is always on the first line (e.g., "map: warehouse")
    if len(lines) >= 1:
        first = lines[0]
        # Be case-insensitive; accept anything before ':' as label
        if ":" in first:
            label, value = first.split(":", 1)
            if label.strip().lower() == "map":
                map_name = value.strip()
        else:
            # Fallback: if line doesn't have ':', take whole line after "map"
            low = first.lower()
            if low.startswith("map"):
                map_name = first.split(" ", 1)[-1].strip()

    # 2) Obstacles info is on the second line (e.g., "obstacles: static (...) and dynamic (...)")
    if len(lines) >= 2:
        second = lines[1].lower()
        # Just check for presence of the words
        if "static" in second:
            static_flag = "x"
        if "dynamic" in second:
            dynamic_flag = "x"

    return (map_name, static_flag, dynamic_flag)


def main():
    # Root directory can be provided as arg, else current working dir
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory.")
        sys.exit(1)

    rows = []
    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue  # skip files like summarize_topics.py, CSVs, etc.

        info_path = os.path.join(folder, "info.txt")
        if not os.path.isfile(info_path):
            continue  # skip if no info.txt

        map_name, static_flag, dynamic_flag = parse_info_txt(info_path)
        rows.append({
            "Name": entry,
            "Map": map_name,
            "Static": static_flag,
            "Dynamic": dynamic_flag,
        })

    out_csv = os.path.join(root, "bag_map_obstacle_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Map", "Static", "Dynamic"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()

