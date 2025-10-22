#!/usr/bin/env python3
import os
import argparse
import csv
from collections import defaultdict

try:
    import yaml  # PyYAML
except ImportError:
    raise SystemExit("PyYAML is required. Install with:\n  pip install pyyaml")

def find_bag_dirs(root):
    """Immediate subdirectories that contain a metadata.yaml."""
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "metadata.yaml")):
            out.append(path)
    return out

def read_metadata(meta_path):
    with open(meta_path, "r") as f:
        return yaml.safe_load(f)

def extract_topic_counts_and_duration(meta):
    """
    Returns:
      topic_counts: list[(topic_name, count)]
      duration_seconds: float
    """
    info = meta.get("rosbag2_bagfile_information", {})
    # duration in ns -> seconds (float)
    dur_ns = info.get("duration", {}).get("nanoseconds", 0) or info.get("duration", {}).get("nanoseconds_since_epoch", 0)
    # Some metadata variants use nested 'files[0].duration.nanoseconds' — prefer top-level if present
    if not dur_ns and "files" in info and info["files"]:
        dur_ns = info["files"][0].get("duration", {}).get("nanoseconds", 0)

    duration_seconds = float(dur_ns) / 1e9 if dur_ns else 0.0

    items = info.get("topics_with_message_count", [])
    topic_counts = []
    for item in items:
        tmeta = item.get("topic_metadata", {})
        name = tmeta.get("name", "")
        count = int(item.get("message_count", 0))
        if name:
            topic_counts.append((name, count))

    return topic_counts, duration_seconds

def main():
    ap = argparse.ArgumentParser(description="Create topic_counts_wide.csv from ROS 2 bag metadata.")
    ap.add_argument("root", nargs="?", default=".", help="Folder containing bag subfolders (default: current dir).")
    ap.add_argument("--out", default="topic_counts_wide.csv", help="Output CSV filename.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    bag_dirs = find_bag_dirs(root)
    if not bag_dirs:
        raise SystemExit(f"No bag directories with metadata.yaml found under: {root}")

    topics_all = set()
    bag_topic_map = defaultdict(dict)   # bag -> {topic: count}
    bag_duration_map = {}               # bag -> duration_seconds

    for bag_path in bag_dirs:
        bag_name = os.path.basename(bag_path)
        meta_path = os.path.join(bag_path, "metadata.yaml")
        try:
            meta = read_metadata(meta_path)
            topic_counts, duration_seconds = extract_topic_counts_and_duration(meta)
        except Exception as e:
            print(f"WARNING: failed to read {meta_path}: {e}")
            continue

        bag_duration_map[bag_name] = duration_seconds
        for topic, count in topic_counts:
            topics_all.add(topic)
            bag_topic_map[bag_name][topic] = count

    topics_sorted = sorted(topics_all)

    # Write wide CSV with one row per bag, topics as columns + duration_seconds + TOTAL
    headers = ["bag", "duration_seconds"] + topics_sorted + ["TOTAL"]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for bag in sorted(bag_topic_map.keys()):
            total = 0
            row = [bag, f"{bag_duration_map.get(bag, 0.0):.6f}"]
            for t in topics_sorted:
                val = bag_topic_map[bag].get(t, 0)
                row.append(val)
                total += val
            row.append(total)
            writer.writerow(row)

    print(f"✔ Wrote {args.out}")

if __name__ == "__main__":
    main()

