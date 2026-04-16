"""
对 JSONL 中 label=uncertain 的记录，计算点对位移比例，
低均值且低方差的标记为 no（运动不显著）。

位移归一化为图像对角线的比例，消除分辨率差异。
低均值但高方差的保留（说明有局部显著运动）。

用法:
python displacement_filter.py \
  --jsonl /mnt/disk1/datasets/drag_data/train_json/pexels_tdv2_all.jsonl \
  --root_dir /mnt/disk1/datasets/drag_data/selectframe/pexels_tdv2 \
  --min_mean_ratio 0.02 \
  --min_std_ratio 0.02
"""
import json
import re
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def run(jsonl_path, root_dir, min_mean_ratio, min_std_ratio):
    pattern = re.compile(r"stride_(\d+)_frame_(\d+)\.png")

    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    uncertain_count = sum(1 for r in records if r.get("label") == "uncertain")
    stats = {"checked": 0, "rejected": 0}

    pbar = tqdm(records, desc="位移筛选", total=len(records))
    for r in pbar:
        if r.get("label") != "uncertain":
            continue

        pair = r.get("pair", [])
        if len(pair) != 2:
            continue

        # 优先从 src_points/tgt_points 字段读 npy 路径
        npy1_rel = r.get("src_points")
        npy2_rel = r.get("tgt_points")

        if npy1_rel and npy2_rel:
            npy1 = os.path.join(root_dir, npy1_rel)
            npy2 = os.path.join(root_dir, npy2_rel)
        else:
            # fallback: 从 pair 文件名推断
            m1 = pattern.search(os.path.basename(pair[0]))
            m2 = pattern.search(os.path.basename(pair[1]))
            if not m1 or not m2:
                continue
            sid = m1.group(1)
            fid1 = m1.group(2)
            fid2 = m2.group(2)
            folder = os.path.join(root_dir, r.get("folder", os.path.dirname(pair[0])))

            npy1 = os.path.join(folder, f"pred_track_stride_{sid}_frame_{fid1}.npy")
            if not os.path.exists(npy1):
                npy1 = os.path.join(folder, f"pred_track_frame_{fid1}.npy")
            npy2 = os.path.join(folder, f"pred_track_stride_{sid}_frame_{fid2}.npy")
            if not os.path.exists(npy2):
                npy2 = os.path.join(folder, f"pred_track_frame_{fid2}.npy")

        # 从 pair 文件名取 frame_id 用于找 original_frame
        m1 = pattern.search(os.path.basename(pair[0]))
        if not m1:
            continue
        fid1 = m1.group(2)
        folder_abs = os.path.join(root_dir, r.get("folder", os.path.dirname(pair[0])))
        orig1 = os.path.join(folder_abs, f"original_frame_{fid1}.png")

        if not os.path.exists(npy1) or not os.path.exists(npy2) or not os.path.exists(orig1):
            continue

        # 只读图片头信息拿分辨率，不解码像素
        try:
            with Image.open(orig1) as img:
                w, h = img.size
        except Exception:
            continue
        diag = np.sqrt(h ** 2 + w ** 2)

        points1 = np.load(npy1)  # (N, 2)
        points2 = np.load(npy2)  # (N, 2)
        if len(points1) != len(points2) or len(points1) == 0:
            r["label"] = "no"
            stats["checked"] += 1
            stats["rejected"] += 1
            pbar.set_postfix(uncertain=uncertain_count, checked=stats["checked"], rejected=stats["rejected"])
            continue
        displacements = np.linalg.norm(points2 - points1, axis=-1)  # (N,)

        # 归一化为对角线比例
        disp_ratio = displacements / diag
        mean_ratio = float(disp_ratio.mean())
        std_ratio = float(disp_ratio.std())

        stats["checked"] += 1
        r["disp_mean_ratio"] = round(mean_ratio, 4)
        r["disp_std_ratio"] = round(std_ratio, 4)

        # 低均值且低方差 -> 整体运动不显著，淘汰
        if mean_ratio < min_mean_ratio and std_ratio < min_std_ratio:
            r["label"] = "no"
            stats["rejected"] += 1

        pbar.set_postfix(uncertain=uncertain_count, checked=stats["checked"], rejected=stats["rejected"])

    # 原地写回
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"检查 {stats['checked']} 条 uncertain, 淘汰 {stats['rejected']} 条 "
          f"(mean_ratio < {min_mean_ratio} 且 std_ratio < {min_std_ratio})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按点对位移比例过滤 uncertain 记录")
    parser.add_argument("--jsonl", required=True, help="JSONL 文件路径")
    parser.add_argument("--root_dir", required=True, help="数据根目录 (JSONL 中的相对路径基于此目录)")
    parser.add_argument("--min_mean_ratio", type=float, default=0.02,
                        help="最小平均位移比例 (相对于图像对角线，默认 0.02)")
    parser.add_argument("--min_std_ratio", type=float, default=0.02,
                        help="最小位移标准差比例 (相对于图像对角线，默认 0.02)")
    args = parser.parse_args()
    run(args.jsonl, args.root_dir, args.min_mean_ratio, args.min_std_ratio)
