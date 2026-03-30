import os
import json
import random
import re
import cv2
import numpy as np
from itertools import combinations
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

def select_tracked_video(root_dir, output_file, total_limit=900):
    """
    从selectvideo中采样一部分视频，按照magictime、pixabay、celebv三个类别进行1:1:1的均衡抽样，并将结果写入output_file。
    仅适配OpenVid-1M数据集。
    """
    def get_video_info(path):
        """获取视频的时长和分辨率"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        
        # 获取参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        duration = (frame_count / fps) if fps > 0 else 0
        cap.release()
        
        return {
            "duration": duration,
            "width": width,
            "height": height
        }
    
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm')
    
    # 初始化三个类别的容器
    pools = {
        "magictime": [],
        "pixabay": [],
        "celebv": []
    }
    
    print(f"开始扫描目录: {root_dir} ...")

    # 1. 递归遍历并分类存入对应的池子,只挑选30秒，长边不超过1300的视频
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith(video_extensions):
                continue
            
            file_lower = file.lower()
            full_path = os.path.abspath(os.path.join(root, file))
            file_name_no_ext = os.path.splitext(file)[0]
            item = {"video": file_name_no_ext, "full_path": full_path}

            video_info = get_video_info(full_path)
            if video_info is not None:
                if video_info["duration"] > 30:
                    continue

                long_edge = max(video_info["width"], video_info["height"])
                if long_edge > 1300:
                    continue

            if 'magictime' in file_lower:
                pools["magictime"].append(item)
            elif 'pixabay' in file_lower:
                pools["pixabay"].append(item)
            elif 'celebv' in file_lower:
                pools["celebv"].append(item)

    # 2. 计算平衡数量
    # 找出三个类别中现有的最小数量
    min_available = min(len(pools["magictime"]), len(pools["pixabay"]), len(pools["celebv"]))
    
    # 根据 total_limit 计算每个类别理想的分配额度
    ideal_per_class = total_limit // 3
    
    # 最终每个类别抽取的数量：不能超过自己有的，也不能超过理想额度
    sample_size = min(min_available, ideal_per_class)
    
    if sample_size == 0:
        print("错误: 其中一个类别的视频数量为 0，无法按 1:1:1 比例抽样。")
        # 这里可以选择继续处理其他类别，或者报错退出
    
    # 3. 执行均衡抽样
    final_results = []
    print(f"\n--- 抽样统计 (比例 1:1:1) ---")
    for label, pool in pools.items():
        sampled_items = random.sample(pool, sample_size)
        final_results.extend(sampled_items)
        print(f"类别 [{label}]: 共有 {len(pool)} 个 -> 抽取 {sample_size} 个")

    # 4. 打乱并写入
    random.shuffle(final_results)

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in final_results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("-" * 30)
    print(f"处理完成！最终文件包含 {len(final_results)} 条记录。")
    print(f"保存至: {output_file}")

def select_tracked_video_pairs(root_dir, output_jsonl):
    PREVIEW_PATH = "./current_preview.jpg" # 建议在 VS Code 中点开此图
    pattern = re.compile(r"stride_(\d+)_frame_(\d+)\.png")

    #先记录所有的已经标注的folder，防止二次标注
    with open(output_jsonl, 'r', encoding='utf-8') as f_in:
        existing_folders = set()
        for line in f_in:
            try:
                record = json.loads(line.strip())
                existing_folders.add(record.get("folder", ""))
            except json.JSONDecodeError:
                continue
    
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
                        record = {
                            "folder": root,
                            "stride": stride_id,
                            "pair": [img_path1, img_path2],
                            "label": "no"
                        }
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f_out.flush()
                        print(" [已跳过 NO]")
                        continue
                    
                    elif user_choice == 'q':
                        print("程序已手动停止。")
                        return
                    else:
                        record = {
                            "folder": root,
                            "stride": stride_id,
                            "pair": [img_path1, img_path2],
                            "label": "no"
                        }
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f_out.flush()
                        print("输入无效，默认跳过...")

    print("\n所有文件夹标注完成！")

class DragDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        self.crop_size = 512
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    self.data.append(record)
                except json.JSONDecodeError:
                    continue

    def image_preprocess(self, img_path1, img_path2, src_points, tgt_points):
        # 加载图像
        src_image = cv2.imread(img_path1) # H, W, C (BGR)
        tgt_image = cv2.imread(img_path2)
        h_orig, w_orig, _ = src_image.shape

        # 1. 计算点集的边界
        all_points = np.concatenate([src_points, tgt_points], axis=0)
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)

        # 2. 确定需要涵盖的最小正方形区域 (ROI)
        content_w = max_x - min_x
        content_h = max_y - min_y
        base_side = max(content_w, content_h, self.crop_size)
        
        # 限制 base_side 不能超过原图的短边（防止缩放崩溃）
        base_side = min(base_side, h_orig, w_orig)

        # 3. 确定 ROI 的中心并修正偏移量
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        
        x_offset = int(center_x - base_side / 2)
        y_offset = int(center_y - base_side / 2)

        # 边界修正：确保 ROI 在原图内
        x_offset = max(0, min(x_offset, w_orig - int(base_side)))
        y_offset = max(0, min(y_offset, h_orig - int(base_side)))
        actual_side = int(base_side)

        # 4. 裁剪并缩放到 512x512
        src_crop = src_image[y_offset : y_offset + actual_side, x_offset : x_offset + actual_side]
        tgt_crop = tgt_image[y_offset : y_offset + actual_side, x_offset : x_offset + actual_side]
        
        # 如果实际裁剪的尺寸不是 512，则进行缩放
        scale = self.crop_size / actual_side
        if actual_side != self.crop_size:
            src_crop = cv2.resize(src_crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            tgt_crop = cv2.resize(tgt_crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)

        # 5. 更新坐标：(原始坐标 - 偏移) * 缩放比例
        new_src_points = (src_points - np.array([x_offset, y_offset])) * scale
        new_tgt_points = (tgt_points - np.array([x_offset, y_offset])) * scale

        # 6. 转换为 Torch Tensor 格式 [-1, 1]
        # 转换为 RGB -> 归一化到 [0, 1] -> 变换到 [-1, 1]
        def to_tensor(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB
            img_tensor = torch.from_numpy(img).float() # [H, W, C]
            img_tensor = (img_tensor / 127.5) - 1.0    # 归一化到 [-1, 1]
            return img_tensor

        src_tensor = to_tensor(src_crop)
        tgt_tensor = to_tensor(tgt_crop)

        return src_tensor, new_src_points, tgt_tensor, new_tgt_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError("索引超出范围")
        
        record = self.data[idx]
        dir = record["folder"]
        stride = record["stride"]
        img_path1, img_path2 = record["pair"]
        label = record["label"]

        frame1 = os.path.basename(img_path1).split('_frame_')[1].split('.png')[0]
        frame2 = os.path.basename(img_path2).split('_frame_')[1].split('.png')[0]
        pred_track_path1 = os.path.join(dir, f"pred_track_frame_{frame1}.npy")
        pred_track_path2 = os.path.join(dir, f"pred_track_frame_{frame2}.npy")
        src_points = np.load(pred_track_path1) if os.path.exists(pred_track_path1) else None
        tgt_points = np.load(pred_track_path2) if os.path.exists(pred_track_path2) else None
        img_path1 = os.path.join(dir, f"original_frame_{frame1}.png")
        img_path2 = os.path.join(dir, f"original_frame_{frame2}.png")

        src_image, src_points, tgt_image, tgt_points = self.image_preprocess(img_path1, img_path2, src_points, tgt_points)

        # 将src_points和tgt_points转换point_map
        src_points_map = torch.zeros((src_image.shape[0], src_image.shape[1], 1), dtype=torch.float32) # [H, W, 1]
        tgt_points_map = torch.zeros((tgt_image.shape[0], tgt_image.shape[1], 1), dtype=torch.float32) # [H, W, 1]
        for i, src_point in enumerate(src_points):
            x, y = int(src_point[0]), int(src_point[1]) # [w, h]
            src_points_map[y, x, 0] = (i + 1) * 1.0 # 注意坐标顺序是 (y, x)
        
        for i, tgt_point in enumerate(tgt_points):
            x, y = int(tgt_point[0]), int(tgt_point[1]) # [w, h]
            tgt_points_map[y, x, 0] = (i + 1) * 1.0 # 注意坐标顺序是 (y, x)

        item = {
            "src_image": src_image,
            "tgt_image": tgt_image,
            "src_points": src_points,
            "tgt_points": tgt_points,
            "src_points_map": src_points_map,
            "tgt_points_map": tgt_points_map,
        }
        return item

class RGBDVideoSequenceDataset:
    def __init__(self, root_dir, output_dir="processed_videos", fps=30):
        """
        root_dir: rgbd-dataset 文件夹路径
        output_dir: 视频保存路径
        fps: 合成视频的帧率
        """
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 匹配规则解释：
        # ^(.+) : 类别_实例 (如 apple_1)
        # _(\d+) : 片段编号 (track_id)
        # _(\d+) : 帧编号 (frame_id)
        # \.png$ : 必须以数字+扩展名结尾，排除了 _depth.png 和 _mask.png
        self.img_pattern = re.compile(r"^(.+)_(\d+)_(\d+)\.png$")

    def process_all(self):
        """核心入口：遍历大类和实例"""
        if not self.root_dir.exists():
            print(f"错误：找不到根目录 {self.root_dir}")
            return

        # 获取所有大类文件夹 (如 apple, banana, bowl...)
        categories = [d for d in self.root_dir.iterdir() if d.is_dir()]
        print(f"找到 {len(categories)} 个类别。")

        for cat in tqdm(categories, desc="总进度"):
            # 获取每个类下的实例 (如 apple_1, apple_2...)
            instances = [d for d in cat.iterdir() if d.is_dir()]
            for inst in instances:
                self._process_instance(inst)

    def _process_instance(self, instance_path):
        """处理单个实例文件夹，按 track 分组"""
        all_files = list(instance_path.glob("*.png"))
        video_tracks = {}

        for f in all_files:
            fname = f.name
            
            # 1. 第一层过滤：排除包含特定关键字的文件
            if any(key in fname for key in ["mask", "depth", "loc", "normal"]):
                continue

            # 2. 第二层过滤：正则匹配 ID 提取
            # 注意：RGB-D 命名通常是 apple_1_1_1.png (inst_track_frame)
            match = self.img_pattern.search(fname)
            if match:
                # 根据文件名解析：实例_片段_帧
                # apple_1_1_155.png -> apple_1(inst), 1(track), 155(frame)
                name_prefix = match.group(1)
                track_id = match.group(2)
                frame_id = int(match.group(3))

                if track_id not in video_tracks:
                    video_tracks[track_id] = []
                
                video_tracks[track_id].append({
                    'path': f,
                    'frame_id': frame_id
                })

        # 3. 开始合成
        for track_id, frames in video_tracks.items():
            # 必须按帧序号排序，否则视频画面会乱跳
            frames.sort(key=lambda x: x['frame_id'])
            
            # 生成文件名：类名_实例名_trackID.mp4
            video_name = f"{instance_path.name}_track{track_id}.mp4"
            self._write_video(video_name, frames)

    def _write_video(self, video_name, frames):
        """使用 OpenCV 写入视频文件"""
        if not frames:
            return

        save_path = self.output_dir / video_name
        
        # 读取第一帧以确定视频的分辨率
        sample_img = cv2.imread(str(frames[0]['path']))
        if sample_img is None:
            return
        
        h, w, _ = sample_img.shape

        # 定义视频编码器 (MP4V 兼容性较好)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(str(save_path), fourcc, self.fps, (w, h))

        for frame_data in frames:
            img = cv2.imread(str(frame_data['path']))
            if img is not None:
                out.write(img)
            else:
                print(f"警告：无法读取图像 {frame_data['path']}")

        out.release()
        # print(f"已完成: {video_name}")

# --- 配置 ---
if __name__ == "__main__":
    # select_tracked_video(
    #     root_dir='/mnt/disk1/datasets/drag_data/selectvideo', 
    #     output_file='tracked_videos.jsonl', 
    #     total_limit=1000  # 你希望最终得到的总条数
    # )

    select_tracked_video_pairs(
        root_dir='/mnt/disk1/datasets/drag_data/selectframe', 
        output_jsonl='paired_frames.jsonl'
    )
    print("请取消注释需要执行的函数，并确保路径正确！")

    # dragdatasets = DragDataset(jsonl_file='paired_frames.jsonl')

    # processor = RGBDVideoSequenceDataset(
    #     root_dir="/mnt/disk1/datasets/rgbd-dataset", 
    #     output_dir="/mnt/disk1/datasets/rgbd-dataset/videos",
    #     fps=30
    # )
    # processor.process_all()