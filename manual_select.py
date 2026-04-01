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
            if root in existing_folders:
                print(f"跳过已标注的文件夹: {root}")
                continue
            img_dict = {}
            for f in files:
                match = pattern.search(f)
                if match:
                    stride_id = match.group(1)
                    full_path = os.path.join(root, f)
                    img_dict.setdefault(stride_id, []).append(full_path)
            
            if not img_dict:
                continue

            print(f"\n>>> 当前文件夹: {root}")
            
            for stride_id, path_list in img_dict.items():
                if len(path_list) < 2:
                    continue
                
                pairs = list(combinations(sorted(path_list), 2))
                
                for img_path1, img_path2 in pairs:
                    img1 = cv2.imread(img_path1)
                    img2 = cv2.imread(img_path2)
                    if img1 is None or img2 is None:
                        continue

                    # 图像处理
                    h, w = 400, 400
                    vis = np.hstack((cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))))
                    
                    # 保存预览图 (你在远程侧边栏看这个文件)
                    cv2.imwrite(PREVIEW_PATH, vis)
                    
                    # --- 关键修改：使用终端输入替代 waitKey ---
                    print(f"\n[待标注] Stride: {stride_id}")
                    print(f"1: {os.path.basename(img_path1)}")
                    print(f"2: {os.path.basename(img_path2)}")
                    
                    # 程序会在这里停住，等待你在终端输入
                    user_choice = input("确认保留? [y:是 / n:跳过 / q:退出]: ").lower().strip()
                    
                    if user_choice == 'y':
                        record = {
                            "folder": root,
                            "stride": stride_id,
                            "pair": [img_path1, img_path2],
                            "label": "yes"
                        }
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f_out.flush()
                        print(" [已记录 YES]")
                    
                    elif user_choice == 'n':
                        os.remove(img_path1)
                        os.remove(img_path2)
                        continue
                    
                    elif user_choice == 'q':
                        print("程序已手动停止。")
                        return
                    else:
                        print("输入无效，默认跳过...")
                        os.remove(img_path1)
                        os.remove(img_path2)

    print("\n所有文件夹标注完成！")

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