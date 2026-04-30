"""
Remove data pairs labeled 'no' from JSONL files and delete corresponding files/folders.

For each 'no' pair:
  - Delete the pair image files and point npy files from disk
  - Remove the line from the JSONL

If a subfolder has no remaining data pairs after removal, delete the subfolder.
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

DRY_RUN = False  # Set to True to preview without deleting

CONFIGS = [
    {
        "jsonl": "/mnt/disk1/datasets/drag_data/train_json/pexels_tdv2_all.jsonl",
        "data_root": "/mnt/disk1/datasets/drag_data/selectframe/pexels_tdv2",
    },
    {
        "jsonl": "/mnt/disk1/datasets/drag_data/train_json/OpenVid-1M_all.jsonl",
        "data_root": "/mnt/disk1/datasets/drag_data/selectframe/OpenVid-1M",
    },
]


def process_jsonl(jsonl_path, data_root):
    print(f"\n{'='*60}")
    print(f"Processing: {jsonl_path}")
    print(f"Data root:  {data_root}")
    print(f"{'='*60}")

    with open(jsonl_path, "r") as f:
        lines = f.readlines()

    kept_lines = []
    removed_count = 0
    deleted_files = 0
    # Track which folders still have data pairs
    folder_has_kept = defaultdict(bool)

    for line in lines:
        entry = json.loads(line)
        folder = entry["folder"]

        if entry.get("label") == "no":
            removed_count += 1
            # Collect files to delete for this pair
            files_to_delete = list(entry["pair"]) + [entry["src_points"], entry["tgt_points"]]
            for rel_path in files_to_delete:
                full_path = os.path.join(data_root, rel_path)
                if os.path.exists(full_path):
                    if DRY_RUN:
                        print(f"  [DRY RUN] Would delete file: {full_path}")
                    else:
                        os.remove(full_path)
                    deleted_files += 1
        else:
            kept_lines.append(line)
            folder_has_kept[folder] = True

    # Rewrite the JSONL without 'no' entries
    if DRY_RUN:
        print(f"  [DRY RUN] Would rewrite {jsonl_path} ({len(lines)} -> {len(kept_lines)} lines)")
    else:
        with open(jsonl_path, "w") as f:
            f.writelines(kept_lines)

    # Find folders that had 'no' entries and check if they should be removed
    all_folders = set()
    for line in lines:
        entry = json.loads(line)
        all_folders.add(entry["folder"])

    removed_folders = 0
    for folder in sorted(all_folders):
        if not folder_has_kept[folder]:
            folder_path = os.path.join(data_root, folder)
            if os.path.isdir(folder_path):
                if DRY_RUN:
                    print(f"  [DRY RUN] Would delete folder: {folder_path}")
                else:
                    shutil.rmtree(folder_path)
                removed_folders += 1

    print(f"\nResults:")
    print(f"  Total entries:    {len(lines)}")
    print(f"  Removed (no):     {removed_count}")
    print(f"  Kept:             {len(kept_lines)}")
    print(f"  Files deleted:    {deleted_files}")
    print(f"  Folders removed:  {removed_folders}")


if __name__ == "__main__":
    if DRY_RUN:
        print("*** DRY RUN MODE - no changes will be made ***\n")

    for cfg in CONFIGS:
        process_jsonl(cfg["jsonl"], cfg["data_root"])

    print("\nDone!")
