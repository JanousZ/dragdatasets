import os
import re
import json
import argparse
import random
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def load_birefnet(device="cuda:0"):
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    model = model.to(device).to(torch.float32).eval()
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, transform


def get_foreground_mask(model, transform, img_path, device="cuda:0", dilate_radius=15):
    """返回原始分辨率的前景概率图 (H, W)，值域 [0, 1]，前景边缘向外膨胀 dilate_radius 像素"""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid()  # 取最后一层输出
    mask = preds[0, 0].cpu().numpy()
    # resize 回原始分辨率
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0
    # 膨胀前景边缘
    if dilate_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_radius * 2 + 1, dilate_radius * 2 + 1))
        mask_resized = cv2.dilate(mask_resized.astype(np.float32), kernel)
    return mask_resized


def count_bg_points(mask, points, fg_threshold=0.5):
    """统计落在背景区域的点数，同时返回每个点的前景/背景标记"""
    h, w = mask.shape
    bg_count = 0
    point_labels = []  # True=前景, False=背景
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        xi = max(0, min(xi, w - 1))
        yi = max(0, min(yi, h - 1))
        is_fg = mask[yi, xi] >= fg_threshold
        if not is_fg:
            bg_count += 1
        point_labels.append(is_fg)
    return bg_count, point_labels


def save_verbose_image(orig_path, mask, points, point_labels, save_path):
    """
    生成可视化图：原图上叠加半透明 mask，画出追踪点。
    绿色圆圈=前景点，红色圆圈=背景点。
    """
    img = cv2.imread(orig_path)
    if img is None:
        return
    h, w = img.shape[:2]

    # 半透明叠加 mask（前景蓝色高亮）
    mask_color = np.zeros_like(img)
    mask_u8 = (mask * 255).astype(np.uint8)
    mask_color[:, :, 0] = mask_u8  # 蓝色通道 = 前景概率
    overlay = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)

    # 画追踪点
    for (x, y), is_fg in zip(points, point_labels):
        xi, yi = int(round(x)), int(round(y))
        color = (0, 220, 0) if is_fg else (0, 0, 255)  # 绿=前景, 红=背景
        cv2.circle(overlay, (xi, yi), 8, color, -1)
        cv2.circle(overlay, (xi, yi), 8, (255, 255, 255), 2)

    cv2.imwrite(save_path, overlay)



def auto_filter(root_dir, output_jsonl, device="cuda:0", bg_ratio_threshold=0.5, fg_threshold=0.5,
                dilate_radius=15, verbose=False, verbose_samples=20):
    """
    自动筛选：对每个 pair，用 BiRefNet 判断追踪点是否在前景上。
    如果两帧中任一帧有超过 bg_ratio_threshold 比例的点落在背景，标记为 no。

    Args:
        root_dir: 数据根目录 (包含各视频子文件夹)
        output_jsonl: 输出文件路径
        device: 推理设备
        bg_ratio_threshold: 背景点占比超过此值则判定为不合格
        fg_threshold: 前景概率低于此值视为背景
        dilate_radius: 前景 mask 向外膨胀像素数，容忍边缘附近的点
        verbose: 是否保存可视化抽样图
        verbose_samples: 淘汰/通过各抽样多少张
    """
    pattern = re.compile(r"stride_(\d+)_frame_(\d+)\.png")

    print("加载 BiRefNet 模型...")
    model, transform = load_birefnet(device)
    print("模型加载完成。")

    # 读取已处理的记录，避免重复
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    existing_pairs = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    pair = record.get("pair", [])
                    if len(pair) == 2:
                        existing_pairs.add(tuple(pair))
                except json.JSONDecodeError:
                    continue

    # 缓存：同一文件夹内的 mask 可以复用
    mask_cache = {}

    stats = {"total": 0, "rejected": 0, "passed": 0}

    # verbose 模式：边跑边保存，用蓄水池抽样保持均匀
    if verbose:
        verbose_dir = os.path.join(os.path.dirname(os.path.abspath(output_jsonl)), "verbose_check")
        rejected_dir = os.path.join(verbose_dir, "rejected")
        passed_dir = os.path.join(verbose_dir, "passed")
        os.makedirs(rejected_dir, exist_ok=True)
        os.makedirs(passed_dir, exist_ok=True)
        # 蓄水池：保存当前抽样的文件名，用于替换
        reservoir_rejected = []  # [(index_in_stream, save_name)]
        reservoir_passed = []
        rejected_seen = 0
        passed_seen = 0

    with open(output_jsonl, "a", encoding="utf-8") as f_out:
        for root, dirs, files in sorted(os.walk(root_dir)):
            # 收集当前文件夹的 stride 图像
            img_dict = {}
            for f in files:
                match = pattern.search(f)
                if match:
                    stride_id = match.group(1)
                    frame_id = match.group(2)
                    rel_path = os.path.relpath(os.path.join(root, f), root_dir)
                    img_dict.setdefault(stride_id, []).append((frame_id, rel_path))

            if not img_dict:
                continue

            mask_cache.clear()

            for stride_id, frame_list in img_dict.items():
                if len(frame_list) < 2:
                    continue

                frame_list.sort(key=lambda x: int(x[0]))
                # 组合 pair（与 manual_select 一致）
                from itertools import combinations
                pairs = list(combinations(frame_list, 2))

                for (fid1, rel_path1), (fid2, rel_path2) in pairs:
                    pair_key = (rel_path1, rel_path2)
                    if pair_key in existing_pairs:
                        continue

                    rel_folder = os.path.relpath(root, root_dir)

                    # 找对应的 original_frame 和 npy（用绝对路径做文件操作）
                    orig1 = os.path.join(root, f"original_frame_{fid1}.png")
                    orig2 = os.path.join(root, f"original_frame_{fid2}.png")

                    # npy: 新命名优先，fallback 旧命名
                    npy1_rel = os.path.join(rel_folder, f"pred_track_stride_{stride_id}_frame_{fid1}.npy")
                    npy1 = os.path.join(root_dir, npy1_rel)
                    if not os.path.exists(npy1):
                        npy1_rel = os.path.join(rel_folder, f"pred_track_frame_{fid1}.npy")
                        npy1 = os.path.join(root_dir, npy1_rel)
                    npy2_rel = os.path.join(rel_folder, f"pred_track_stride_{stride_id}_frame_{fid2}.npy")
                    npy2 = os.path.join(root_dir, npy2_rel)
                    if not os.path.exists(npy2):
                        npy2_rel = os.path.join(rel_folder, f"pred_track_frame_{fid2}.npy")
                        npy2 = os.path.join(root_dir, npy2_rel)

                    if not all(os.path.exists(p) for p in [orig1, orig2, npy1, npy2]):
                        continue

                    points1 = np.load(npy1)  # (N, 2), x-y 坐标
                    points2 = np.load(npy2)

                    # 获取前景 mask（带缓存）
                    if orig1 not in mask_cache:
                        mask_cache[orig1] = get_foreground_mask(model, transform, orig1, device, dilate_radius)
                    if orig2 not in mask_cache:
                        mask_cache[orig2] = get_foreground_mask(model, transform, orig2, device, dilate_radius)

                    mask1 = mask_cache[orig1]
                    mask2 = mask_cache[orig2]

                    n_points = len(points1)
                    bg1, labels1 = count_bg_points(mask1, points1, fg_threshold)
                    bg2, labels2 = count_bg_points(mask2, points2, fg_threshold)

                    # 两帧中取较严重的那个
                    max_bg_ratio = max(bg1, bg2) / n_points if n_points > 0 else 0
                    is_rejected = max_bg_ratio >= bg_ratio_threshold

                    label = "no" if is_rejected else "uncertain"
                    record = {
                        "folder": rel_folder,
                        "stride": stride_id,
                        "pair": [rel_path1, rel_path2],
                        "src_points": npy1_rel,
                        "tgt_points": npy2_rel,
                        "label": label,
                        "bg_points_frame1": bg1,
                        "bg_points_frame2": bg2,
                        "total_points": n_points,
                        "max_bg_ratio": round(max_bg_ratio, 3),
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()

                    stats["total"] += 1
                    if is_rejected:
                        stats["rejected"] += 1
                    else:
                        stats["passed"] += 1

                    # verbose: 蓄水池抽样，边跑边保存
                    if verbose:
                        # 选背景点更多的那帧作为代表
                        if bg1 >= bg2:
                            v_orig, v_mask, v_pts, v_labels = orig1, mask1, points1, labels1
                        else:
                            v_orig, v_mask, v_pts, v_labels = orig2, mask2, points2, labels2

                        folder_name = os.path.basename(root)
                        save_name = f"{folder_name}_s{stride_id}_bg{record['max_bg_ratio']}.jpg"

                        if is_rejected:
                            rejected_seen += 1
                            if len(reservoir_rejected) < verbose_samples:
                                # 池未满，直接保存
                                save_verbose_image(v_orig, v_mask, v_pts, v_labels,
                                                   os.path.join(rejected_dir, save_name))
                                reservoir_rejected.append(save_name)
                            else:
                                # 蓄水池替换
                                j = random.randint(0, rejected_seen - 1)
                                if j < verbose_samples:
                                    old_name = reservoir_rejected[j]
                                    old_path = os.path.join(rejected_dir, old_name)
                                    if os.path.exists(old_path):
                                        os.remove(old_path)
                                    save_verbose_image(v_orig, v_mask, v_pts, v_labels,
                                                       os.path.join(rejected_dir, save_name))
                                    reservoir_rejected[j] = save_name
                        else:
                            passed_seen += 1
                            if len(reservoir_passed) < verbose_samples:
                                save_verbose_image(v_orig, v_mask, v_pts, v_labels,
                                                   os.path.join(passed_dir, save_name))
                                reservoir_passed.append(save_name)
                            else:
                                j = random.randint(0, passed_seen - 1)
                                if j < verbose_samples:
                                    old_name = reservoir_passed[j]
                                    old_path = os.path.join(passed_dir, old_name)
                                    if os.path.exists(old_path):
                                        os.remove(old_path)
                                    save_verbose_image(v_orig, v_mask, v_pts, v_labels,
                                                       os.path.join(passed_dir, save_name))
                                    reservoir_passed[j] = save_name

                    if stats["total"] % 50 == 0:
                        print(f"进度: {stats['total']} pairs | 淘汰: {stats['rejected']} | 待人工: {stats['passed']}")

    print(f"\n完成! 共 {stats['total']} pairs, 自动淘汰 {stats['rejected']}, 待人工审核 {stats['passed']}")
    print(f"结果保存至: {output_jsonl}")

    if verbose:
        print(f"可视化抽样保存至: {verbose_dir}/")
        print(f"  rejected/ : {len(reservoir_rejected)} 张")
        print(f"  passed/   : {len(reservoir_passed)} 张")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用 BiRefNet 自动筛选背景点过多的 pair")
    parser.add_argument("--root_dir", required=True, help="数据根目录")
    parser.add_argument("--output_jsonl", required=True, help="输出 JSONL 路径")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--bg_ratio", type=float, default=0.5,
                        help="背景点占比阈值，超过则淘汰 (默认 0.5，即超过一半的点在背景)")
    parser.add_argument("--fg_threshold", type=float, default=0.3,
                        help="前景概率阈值，低于此值判为背景 (默认 0.3)")
    parser.add_argument("--dilate_radius", type=int, default=15,
                        help="前景 mask 向外膨胀像素数 (默认 15)，设 0 关闭膨胀")
    parser.add_argument("--verbose", action="store_true",
                        help="开启可视化抽样，保存淘汰/通过的样例图到 verbose_check/")
    parser.add_argument("--verbose_samples", type=int, default=20,
                        help="淘汰/通过各抽样保存多少张 (默认 20)")
    args = parser.parse_args()

    auto_filter(
        root_dir=args.root_dir,
        output_jsonl=args.output_jsonl,
        device=args.device,
        bg_ratio_threshold=args.bg_ratio,
        fg_threshold=args.fg_threshold,
        dilate_radius=args.dilate_radius,
        verbose=args.verbose,
        verbose_samples=args.verbose_samples,
    )