import os
import json
import re
import cv2
import numpy as np
from itertools import combinations
import argparse

def select_tracked_video_pairs(root_dir, output_jsonl):
    PREVIEW_PATH = "./current_preview.jpg" # 建议在 VS Code 中点开此图
    pattern = re.compile(r"stride_(\d+)_frame_(\d+)\.png")

    #如果output_jsonl不存在，先创建
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    if not os.path.exists(output_jsonl):
        open(output_jsonl, 'w').close()

    #先记录所有的已经标注的folder，防止二次标注
    try:
        with open(output_jsonl, 'r', encoding='utf-8') as f_in:
            existing_folders = set()
            for line in f_in:
                try:
                    record = json.loads(line.strip())
                    existing_folders.add(record.get("folder", ""))
                except json.JSONDecodeError:
                    continue
    except:
        existing_folders = set()

    with open(output_jsonl, 'a', encoding='utf-8') as f_out:
        for root, dirs, files in os.walk(root_dir):
            rel_folder = os.path.relpath(root, root_dir)
            if rel_folder in existing_folders:
                print(f"跳过已标注的文件夹: {rel_folder}")
                continue
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

            print(f"\n>>> 当前文件夹: {rel_folder}")

            for stride_id, frame_list in img_dict.items():
                if len(frame_list) < 2:
                    continue

                frame_list.sort(key=lambda x: int(x[0]))
                pairs = list(combinations(frame_list, 2))

                for (fid1, rel_path1), (fid2, rel_path2) in pairs:
                    abs_path1 = os.path.join(root_dir, rel_path1)
                    abs_path2 = os.path.join(root_dir, rel_path2)
                    img1 = cv2.imread(abs_path1)
                    img2 = cv2.imread(abs_path2)
                    if img1 is None or img2 is None:
                        continue

                    # npy: 新命名优先，fallback 旧命名
                    npy1_rel = os.path.join(rel_folder, f"pred_track_stride_{stride_id}_frame_{fid1}.npy")
                    if not os.path.exists(os.path.join(root_dir, npy1_rel)):
                        npy1_rel = os.path.join(rel_folder, f"pred_track_frame_{fid1}.npy")
                    npy2_rel = os.path.join(rel_folder, f"pred_track_stride_{stride_id}_frame_{fid2}.npy")
                    if not os.path.exists(os.path.join(root_dir, npy2_rel)):
                        npy2_rel = os.path.join(rel_folder, f"pred_track_frame_{fid2}.npy")

                    # 图像处理
                    h, w = 400, 400
                    vis = np.hstack((cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))))

                    # 保存预览图 (你在远程侧边栏看这个文件)
                    cv2.imwrite(PREVIEW_PATH, vis)

                    # --- 关键修改：使用终端输入替代 waitKey ---
                    print(f"\n[待标注] Stride: {stride_id}")
                    print(f"1: {os.path.basename(rel_path1)}")
                    print(f"2: {os.path.basename(rel_path2)}")

                    # 程序会在这里停住，等待你在终端输入
                    user_choice = input("确认保留? [y:是 / n:跳过 / q:退出]: ").lower().strip()

                    def make_record(label):
                        return {
                            "folder": rel_folder,
                            "stride": stride_id,
                            "pair": [rel_path1, rel_path2],
                            "src_points": npy1_rel,
                            "tgt_points": npy2_rel,
                            "label": label
                        }

                    if user_choice == 'y':
                        f_out.write(json.dumps(make_record("yes"), ensure_ascii=False) + "\n")
                        f_out.flush()
                        print(" [已记录 YES]")

                    elif user_choice == 'n':
                        f_out.write(json.dumps(make_record("no"), ensure_ascii=False) + "\n")
                        f_out.flush()
                        print(" [已记录 No]")
                        continue

                    elif user_choice == 'q':
                        print("程序已手动停止。")
                        return
                    else:
                        f_out.write(json.dumps(make_record("no"), ensure_ascii=False) + "\n")
                        f_out.flush()
                        print("输入无效，默认跳过...")

    print("\n所有文件夹标注完成！")

def review_uncertain_pairs(root_dir, output_jsonl):
    """读取 JSONL 中 BiRefNet 标记为 uncertain 的记录，逐条人工审核，结果追加到同一文件"""
    PREVIEW_PATH = "./current_preview.jpg"

    if not os.path.exists(output_jsonl):
        print("JSONL 文件不存在，跳过 uncertain 审核。")
        return

    # 读取所有记录，找出 uncertain 的，同时收集已经被人工审核过的 pair
    all_records = []
    reviewed_pairs = set()  # 已经有 yes/no 最终判定的 pair
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                all_records.append(record)
                pair = tuple(record.get("pair", []))
                if record.get("label") in ("yes", "no"):
                    reviewed_pairs.add(pair)
            except json.JSONDecodeError:
                continue

    # 筛选出还没被人工审核的 uncertain 记录
    uncertain_todo = []
    for r in all_records:
        if r.get("label") == "uncertain" and tuple(r.get("pair", [])) not in reviewed_pairs:
            uncertain_todo.append(r)

    print(f"\n共 {len(uncertain_todo)} 条 uncertain 记录待人工审核")
    if not uncertain_todo:
        return

    with open(output_jsonl, 'a', encoding='utf-8') as f_out:
        for idx, record in enumerate(uncertain_todo):
            rel_path1, rel_path2 = record["pair"]
            stride_id = record["stride"]

            abs_path1 = os.path.join(root_dir, rel_path1)
            abs_path2 = os.path.join(root_dir, rel_path2)
            img1 = cv2.imread(abs_path1)
            img2 = cv2.imread(abs_path2)
            if img1 is None or img2 is None:
                print(f"无法读取图片，跳过: {rel_path1} / {rel_path2}")
                continue

            h, w = 400, 400
            vis = np.hstack((cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))))
            cv2.imwrite(PREVIEW_PATH, vis)

            print(f"\n[{idx+1}/{len(uncertain_todo)}] 文件夹: {record.get('folder', '')}")
            print(f"  Stride: {stride_id} | bg_ratio: {record.get('max_bg_ratio', '?')}")
            print(f"  1: {os.path.basename(rel_path1)}")
            print(f"  2: {os.path.basename(rel_path2)}")

            user_choice = input("确认保留? [y:是 / n:跳过 / q:退出]: ").lower().strip()

            out_record = {**record}
            if user_choice == 'y':
                out_record["label"] = "yes"
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()
                print(" [已记录 YES]")
            elif user_choice == 'q':
                print("程序已手动停止。")
                return
            else:
                out_record["label"] = "no"
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f_out.flush()
                print(" [已记录 No]" if user_choice == 'n' else "输入无效，默认标记为 No")

    print("\n所有 uncertain 记录审核完成！")


# --- 配置 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    args = parser.parse_args()
    select_tracked_video_pairs(
        root_dir=args.root_dir,
        output_jsonl=args.output_jsonl
    )

    #审核 BiRefNet 标记为 uncertain 的记录
    review_uncertain_pairs(args.root_dir, args.output_jsonl)