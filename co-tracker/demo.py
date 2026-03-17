# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import json
import cv2
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
import time

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def sample_track_points(video, pred_tracks, pred_visibility, video_width=None, video_height=None, num_samples=None, grid_size=20):

    def calculate_total_displacement(pred_tracks, mode='path_length'):
        """
        tracks: [1, F, N, 2] (Batch, Frames, Number_of_points, XY_coordinates)
        mode: 'path_length' (累积路径) 或 'net_distance' (首尾净位移)
        """
        # 移除 Batch 维度方便计算 -> [F, N, 2]
        t = pred_tracks.squeeze(0)
        F, N, _ = t.shape

        if mode == 'net_distance':
            # 计算最后一帧与第一帧的 L2 距离
            # dist = sqrt((x_end - x_start)^2 + (y_end - y_start)^2)
            displacement = torch.norm(t[-1] - t[0], dim=-1) # 形状: [N]
            
        elif mode == 'path_length':
            # 1. 计算相邻帧之间的位移向量 [F-1, N, 2]
            # t[1:] 是第 2 帧到最后一帧，t[:-1] 是第 1 帧到倒数第二帧
            diffs = t[1:] - t[:-1]
            
            # 2. 计算每一步的欧氏距离 [F-1, N]
            step_distances = torch.norm(diffs, dim=-1)
            
            # 3. 对时间维度求和，得到每个点的总路径长度 [N]
            displacement = torch.sum(step_distances, dim=0)

        return displacement

    def filter_out_of_video(pred_tracks, video_width, video_height):
        # pred_tracks: [1, F, N, 2]
        t = pred_tracks.squeeze(0)  # [F, N, 2]
        x = t[..., 0]  # [F, N]
        y = t[..., 1]  # [F, N]

        out_of_bounds = (x < 0) | (x >= video_width) | (y < 0) | (y >= video_height)  # [F, N]
        ever_out_of_bounds = torch.any(out_of_bounds, dim=0)  # [N]

        return ever_out_of_bounds

    def sample_diverse_points(significant_points, significant_motion_mask, displacement, pred_tracks, num_samples):
        # significant_points: [M, 2] (t=0时的坐标)
        # displacement: [M] (对应点的总位移)
        # pred_tracks: [1, T, N, 2]
        
        device = significant_points.device
        M = significant_points.shape[0]
        
        if M <= num_samples:
            return significant_points.cpu().numpy().tolist()

        # --- 步骤 A: 提取运动方向特征 ---
        # 计算起点到终点的向量 [M, 2]
        start_pt = pred_tracks[0, 0, significant_motion_mask]
        end_pt = pred_tracks[0, -1, significant_motion_mask]
        motion_vectors = end_pt - start_pt # 运动矢量
        
        # 归一化运动矢量，使其只代表方向
        norms = torch.norm(motion_vectors, dim=1, keepdim=True) + 1e-6
        motion_directions = motion_vectors / norms 
        
        # --- 步骤 B: 构建多维特征用于采样 ---
        # 特征 = [归一化坐标(x, y), 运动方向(dx, dy) * 权重]
        # 权重决定了你有多在意“方向多样性”对比“空间分布”
        alpha = 0.5 
        feat_coords = significant_points / torch.max(significant_points) * 0.5 # 归一化位置
        features = torch.cat([feat_coords, motion_directions * alpha], dim=1) # [M, 4]

        norm_displacement = (displacement[significant_motion_mask] - displacement[significant_motion_mask].min()) / (displacement[significant_motion_mask].max() - displacement[significant_motion_mask].min() + 1e-6)
        beta = 1.2  # 调节因子：1.0是线性，>1.0 极其偏爱大位移
        score = norm_displacement ** beta

        # --- 步骤 C: 最远点采样 (FPS) ---
        selected_indices = []
        # 初始点选择位移最大的点，确保选到一个显著运动的点
        current_idx = torch.argmax(displacement[significant_motion_mask])
        selected_indices.append(current_idx.item())
        
        # 距离数组，记录每个点到已选点集的最短距离
        min_distances = torch.full((M,), 1e10, device=device)
        
        for _ in range(num_samples - 1):
            # 更新所有点到新加入点的距离
            current_feat = features[current_idx]
            dists = torch.norm(features - current_feat, dim=1)
            min_distances = torch.min(min_distances, dists)
            weighted_dists = min_distances * score
            
            # 选出距离当前点集最远的点
            current_idx = torch.argmax(weighted_dists)
            selected_indices.append(current_idx.item())

        sampled_points = significant_points[selected_indices]
        return sampled_points.cpu().numpy().tolist()
    
    def refine_points_with_corners(significant_points, first_frame, radius_x, radius_y, k=2):
        """
        significant_points: torch.Tensor [M, 2] (x, y)
        first_frame: numpy array [H, W, 3] 第一帧图像
        """
        device = significant_points.device
        M = significant_points.shape[0]
        k = k + radius_x * radius_y // 500

        # 1. 转换为灰度图计算特征点
        first_frame = first_frame.cpu().numpy().transpose(1, 2, 0).astype(np.uint8) # [H, W, 3]
        gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
        
        # maxCorners 设高一点，确保覆盖全图
        # qualityLevel 0.01 表示保留强度超过最强点 1% 的点
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=5000, qualityLevel=0.01, minDistance=3)
        
        if corners is None:
            # 如果没找到任何特征点，退回到你原来的随机逻辑
            k = k + radius_x * radius_y // 500
            offsets_x = (torch.rand(significant_points.shape[0], k, 1, device=significant_points.device) * 2 - 1) * radius_x
            offsets_y = (torch.rand(significant_points.shape[0], k, 1, device=significant_points.device) * 2 - 1) * radius_y
            offsets = torch.cat([offsets_x, offsets_y], dim=-1)
            new_sampled_points = significant_points.unsqueeze(1) + offsets
            new_sampled_points = new_sampled_points.view(-1, 2)
            unique_points = torch.unique(new_sampled_points, dim=0)
            unique_points[:, 0] = unique_points[:, 0].clamp(0, gray.shape[1] - 1)
            unique_points[:, 1] = unique_points[:, 1].clamp(0, gray.shape[0] - 1)
            unique_points = torch.unique(new_sampled_points, dim=0)
            return unique_points # 这里可以接你原来的随机 offset 代码
        
        # 转换为 [N, 2] 的 tensor
        corner_points = torch.from_numpy(corners.reshape(-1, 2)).to(device)
        
        refined_samples = []
        
        # 2. 对每个显著运动点，在半径内找最强的 K 个特征点
        for pt in significant_points:
            # 计算该点到所有特征点的距离
            dists_x = torch.abs(corner_points[:, 0] - pt[0])
            dists_y = torch.abs(corner_points[:, 1] - pt[1])
            
            # 筛选出在 Radius 范围内的特征点
            mask = (dists_x <= radius_x) & (dists_y <= radius_y)
            in_radius_corners = corner_points[mask]
            
            if in_radius_corners.shape[0] > 0:
                # 如果特征点多于 k 个，随机选 k 个或者选距离最近的
                # 这里建议随机选，增加样本多样性
                idx = torch.randperm(in_radius_corners.shape[0])[:k]
                refined_samples.append(in_radius_corners[idx])
            else:
                # 如果附近没特征点，保留原点并加上微小随机扰动
                offsets = (torch.rand(k, 2, device=device) * 2 - 1) * torch.tensor([radius_x, radius_y], device=device) * 0.5
                refined_samples.append(pt.unsqueeze(0) + offsets)
                
        # 合并结果
        new_sampled_points = torch.cat(refined_samples, dim=0)
        
        # 后处理：去重、裁剪边界
        unique_points = torch.unique(torch.round(new_sampled_points), dim=0)
        unique_points[:, 0] = unique_points[:, 0].clamp(0, gray.shape[1] - 1)
        unique_points[:, 1] = unique_points[:, 1].clamp(0, gray.shape[0] - 1)
        unique_points = torch.unique(unique_points, dim=0)
        return unique_points
    
    # 1. 计算每个点的总位移
    displacement = calculate_total_displacement(pred_tracks, mode='path_length')   #[N]
    final_displacement = calculate_total_displacement(pred_tracks, mode='net_distance')   #[N]
    # 2. 过滤掉视频边界外的点
    if video_width is not None and video_height is not None:
        out_of_bounds_mask = filter_out_of_video(pred_tracks, video_width, video_height)
        displacement[out_of_bounds_mask] = 0  
        final_displacement[out_of_bounds_mask] = 0

    # 过滤掉位移短的点，保留位移较大的点
    avg_motion = displacement.mean()
    std_motion = displacement.std()
    if num_samples is None:
        motion_threshold = avg_motion + 0.5 * std_motion
    else:
        motion_threshold = avg_motion
    
    final_motion_threshold = final_displacement.mean() * 0.5  
    significant_motion_mask = (displacement > motion_threshold) & (final_displacement > final_motion_threshold)
    significant_points = pred_tracks[0, 0, significant_motion_mask]

    if num_samples is None:
        # 进行点扩展
        unique_points = refine_points_with_corners(significant_points, video[0, 0], radius_x=video_width//grid_size, radius_y=video_height//grid_size, k=2)
        return unique_points.cpu().numpy().tolist()
    else:
        # 进行点筛选
        # if significant_points.shape[0] > num_samples:
        #     # 对significant_points进行displacement排序，选择位移最大的前 num_samples * 2 个点
        #     if significant_points.shape[0] > num_samples * 2:
        #         topk_indices = torch.topk(displacement[significant_motion_mask], k=num_samples * 2).indices
        #         significant_points = significant_points[topk_indices]
                
        #     indices = torch.randperm(significant_points.shape[0])[:num_samples]
        #     significant_points = significant_points[indices]
        significant_points = sample_diverse_points(significant_points, significant_motion_mask, displacement, pred_tracks, num_samples)
        return significant_points

def sample_frames(video, video_name, pred_tracks, pred_visibility, strides=[15, 60]):
    """
    在整个序列中，为每个指定的 stride 挑选一个最优的帧对 (t1, t2)。
    
    Args:
        video: 视频张量
        video_name: 视频名称
        pred_tracks: [1, T, N, 2] - 轨迹坐标
        pred_visibility: [1, T, N] - 可见性掩码 (1可见, 0遮挡)
        strides: List[int] - 想要抽取的帧间距，例如 [15, 60]
        
    Returns:
        dict: {stride: (t1, t2)} 包含了每个 stride 对应的最佳帧索引对
    """
    T = pred_tracks.shape[1]
    N = pred_tracks.shape[2]
    selected_pairs = {}

    for delta_t in strides:
        best_score = -1.0
        best_pair = None
        
        # 遍历所有可能的起始帧 t1
        for t1 in range(T - delta_t):
            t2 = t1 + delta_t
            
            # 1. 提取可见性并计算共视点数量 (Commonly Visible Points)
            # v1, v2 形状为 [N]
            v1 = pred_visibility[0, t1]
            v2 = pred_visibility[0, t2]
            common_mask = (v1 > 0.5) & (v2 > 0.5)
            num_common = common_mask.sum().item()
            
            # 如果这一对帧之间完全没有共视点，直接跳过
            if num_common == 0:
                continue
            
            # 2. 计算这些共视点的平均位移 (Motion)
            # p1, p2 形状为 [num_common, 2]
            p1 = pred_tracks[0, t1, common_mask]
            p2 = pred_tracks[0, t2, common_mask]
            motion = torch.norm(p1 - p2, dim=-1).mean().item()
            
            # 3. 评分机制：共视点比例 * log(位移 + 1)
            # 这样既保证了有足够的点进行训练，又保证了动作是有意义的（不是静止的）
            score = (1 / num_common) * torch.tensor(motion)
            
            if score > best_score:
                best_score = score
                best_pair = (t1, t2)
        
        # 将该 stride 下表现最好的一对存入结果
        selected_pairs[delta_t] = best_pair
        if best_pair is not None:
            save_annotated_frames(video, pred_tracks, pred_visibility, best_pair, delta_t, save_dir=os.path.join("/mnt/disk1/datasets/drag_data/selectframe", video_name))

    return selected_pairs

def save_annotated_frames(video_tensor, pred_tracks, pred_visibility, pair, stride, save_dir):
    """
    辅助函数：负责具体的绘制和写入磁盘操作
    """
    import colorsys
    def get_vibrant_colors(n):
        colors = []
        for i in range(n):
            # 在色相环上均匀分布 (0.0 到 1.0)
            hue = i / n
            # 固定高饱和度和高亮度，确保颜色极其鲜艳
            saturation = 0.9 
            brightness = 1.0
            
            # HSV 转 RGB (范围 0-1)
            rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
            # 转为 OpenCV 使用的 0-255 tuple
            colors.append(tuple(int(c * 255) for c in rgb))
            
        # 打乱颜色顺序，防止相邻的追踪点颜色太接近
        np.random.seed(42)
        np.random.shuffle(colors)
        return colors

    t1, t2 = pair
    N = pred_tracks.shape[2]
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成固定颜色种子，确保 t1 和 t2 的同一点颜色一致
    np.random.seed(42)
    colors = get_vibrant_colors(N)
    # colors = [tuple(map(int, np.random.randint(0, 255, 3).tolist())) for _ in range(N)]

    for t_idx in [t1, t2]:
        # Tensor -> BGR Numpy
        frame = video_tensor[0, t_idx].permute(1, 2, 0).cpu().numpy()
        if frame.max() <= 1.0: frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # 假设输入是RGB

        cv2.imwrite(os.path.join(save_dir, f"original_frame_{t_idx}.png"), frame)  # 保存原始帧
        # 保存 pred_tracks 和 pred_visibility
        np.save(os.path.join(save_dir, f"pred_track_frame_{t_idx}"), pred_tracks[0,t_idx].cpu().numpy())

        # 绘制每一个点
        for i in range(N):
            pos = pred_tracks[0, t_idx, i].int().cpu().numpy()
            vis = pred_visibility[0, t_idx, i].item()
            color = colors[i]
            center = (pos[0], pos[1])

            # if vis > 0.5:
            #     cv2.circle(frame, center, 5, color, -1) # 实心
            # else:
            #     cv2.circle(frame, center, 5, color, 2)  # 空心
            cv2.circle(frame, center, 5, color, -1)

        # 保存图片
        file_path = os.path.join(save_dir, f"stride_{stride}_frame_{t_idx}.png")
        cv2.imwrite(file_path, frame)
        print(f"Saved: {file_path}")

def get_adaptive_stride_pair(pred_tracks, video_dims, video_name, target_shift=0.10, min_s=5, max_s=100):
    """
    pred_tracks: [1, T, N, 2] Co-Tracker 的预测轨迹
    video_dims: (W, H)
    target_shift: 期望的平均位移比例 (0.10 代表 10% 的屏幕尺寸)
    """
    W, H = video_dims
    diag = np.sqrt(W**2 + H**2)
    T = pred_tracks.shape[1]
    N = pred_tracks.shape[2]
    
    # 1. 计算所有可能的帧对之间的平均位移
    # 为了效率，我们可以遍历起始帧 t，尝试不同的 stride
    best_score = -1e10
    best_pair = (0, min(T-1, 30)) # 默认保底
    
    # 2. 这里的搜索可以优化：
    # 我们希望找到一对帧 (t1, t2)，使得：
    # (a) 平均位移接近 target_shift * diag
    # (b) 所有点都在画面内 (Visibility)
    # (c) 轨迹尽可能直 (运动效率高)
    
    # 采样一部分起始帧进行快速搜索
    for t1 in range(0, T - min_s, 5): 
        for s in [15, 30, 45, 60, 80]: # 候选 Stride
            t2 = t1 + s
            if t2 >= T: break
            
            # 计算这一对帧的 N 个点的位移
            pts1 = pred_tracks[0, t1] # [N, 2]
            pts2 = pred_tracks[0, t2] # [N, 2]
            dist = torch.norm(pts2 - pts1, dim=-1) # [N]
            avg_dist = dist.mean().item()
            
            # 量化得分：
            # 1. 位移得分 (接近目标值最高)
            shift_ratio = avg_dist / diag
            dist_score = 1.0 - abs(shift_ratio - target_shift) / target_shift
            
            # 2. 边界保护 (如果有任何点跑出屏幕，大幅扣分)
            out_of_bounds = (pts2[:, 0] < 0) | (pts2[:, 0] >= W) | \
                            (pts2[:, 1] < 0) | (pts2[:, 1] >= H)
            visibility_score = 1.0 if not out_of_bounds.any() else 0.0
            
            # 总分
            current_score = dist_score * visibility_score
            
            if current_score > best_score:
                best_score = current_score
                best_pair = (t1, t2)
    
    save_annotated_frames(video, pred_tracks, pred_visibility, best_pair, best_pair[1]-best_pair[0], save_dir=os.path.join("/mnt/disk1/datasets/drag_data/selectframe", video_name))
                
    return best_pair # 返回找到的最佳 (start_frame, end_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="/home/yanzhang/Video-T1/final_results/A_charming_wooden_birdhouse,_painted_in_vibrant_hues_of_red_and_blue,_hangs_gracefully_from_a_sturdy.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--video_json",
        default="/home/yanzhang/dragdatasets/points_annotation_test.jsonl",
        # default=None,
        help="video paths",
    )
    parser.add_argument(
        "--video_dir",
        default="/mnt/disk1/datasets/drag_data/selectvideo",
        type=str,
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/scaled_offline.pth",
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=20, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
    )

    args = parser.parse_args()

    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
        else:
            if args.offline:
                window_len = 60
            else:
                window_len = 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    
    model = model.to(DEFAULT_DEVICE)

    # load json
    if args.video_json is not None:
        video_paths = []
        video_names = []

        with open(args.video_json, "r", encoding='utf-8') as f:
            for line in f:
                # 去掉行尾空格和换行符，并解析
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                
                # 提取字段
                video_paths.append(data["full_path"])
                video_names.append(data["video"])
    elif args.video_dir is not None:
        video_paths = []
        video_names = []
        for filename in os.listdir(args.video_dir):
            if filename.endswith(".mp4") or filename.endswith(".avi"):
                video_paths.append(os.path.join(args.video_dir, filename))
                video_names.append(filename.split(".")[0])
    else:
        video_paths = [args.video_path]
        video_names = [args.video_path.split("/")[-1].split(".")[0]]

    for i,video_path in enumerate(video_paths):
    # load the input video frame by frame

        np.random.seed(int(time.time()) + i)  # 每个视频使用不同的随机种子，确保结果多样但可复现
        #random_index = np.random.randint(0, len(video_paths))
        random_index = i
        # if "pixabay" not in video_paths[random_index]:
        #     continue
        video = read_video_from_path(video_paths[random_index])
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

        flipped_video = torch.flip(video, dims=[1]).to(DEFAULT_DEVICE)
        #segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
        #segm_mask = torch.from_numpy(segm_mask)[None, None]

        video = video.to(DEFAULT_DEVICE)

        # points = [ (128,128), (172,172), (256,256)]  # (w,h)
        # points = process_video_pair(video_path, sample_interval=15, num_points=10)
        # frame_idx = 0  # 这些点所在的帧
        # queries = torch.tensor([ [ [frame_idx, x, y] for (x, y) in points ] ]).to(DEFAULT_DEVICE).to(torch.float32)  # shape (1, N, 3)
        # pred_tracks, pred_visibility = model(video, queries=queries)

        pred_tracks, pred_visibility = model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            backward_tracking=args.backward_tracking,
            # segm_mask=segm_mask
        )
        # vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=2)
        # vis.visualize(
        #     video,
        #     pred_tracks,
        #     pred_visibility,
        #     query_frame=0 if args.backward_tracking else args.grid_query_frame,
        #     filename=f"{video_names[random_index]}_grid",
        # )

        points = sample_track_points(video, pred_tracks, pred_visibility, video_width=video.shape[-1], video_height=video.shape[-2], grid_size=args.grid_size)
        frame_idx = 0
        queries = torch.tensor([ [ [frame_idx, x, y] for (x, y) in points ] ]).to(DEFAULT_DEVICE).to(torch.float32)
        pred_tracks, pred_visibility = model(video, queries=queries)
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=2)
        vis.visualize(
            video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if args.backward_tracking else args.grid_query_frame,
            filename=f"{video_names[random_index]}_1st_sample",
        )

        points = sample_track_points(video, pred_tracks, pred_visibility, video_width=video.shape[-1], video_height=video.shape[-2], num_samples=10, grid_size=args.grid_size)
        frame_idx = 0
        queries = torch.tensor([ [ [frame_idx, x, y] for (x, y) in points ] ]).to(DEFAULT_DEVICE).to(torch.float32)
        pred_tracks, pred_visibility = model(video, queries=queries)
        
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=2)
        vis.visualize(
            video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if args.backward_tracking else args.grid_query_frame,
            filename=f"{video_names[random_index]}",
        )

        selected_frames = sample_frames(video, video_names[random_index], pred_tracks, pred_visibility, strides=[15, 60, video.shape[1]-1])
        # get_adaptive_stride_pair(pred_tracks, video_dims=(video.shape[-1], video.shape[-2]), video_name=video_names[random_index], target_shift=0.10)

        # 将视频进行反转播放，再跑一次
        video_name = video_names[random_index]
        video_name = video_name + "_flip"
        video = video.flip(dims=[1])
        pred_tracks, pred_visibility = model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            backward_tracking=args.backward_tracking,
            # segm_mask=segm_mask
        )
        # vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=2)
        # vis.visualize(
        #     video,
        #     pred_tracks,
        #     pred_visibility,
        #     query_frame=0 if args.backward_tracking else args.grid_query_frame,
        #     filename=f"{video_name}_grid",
        # )

        points = sample_track_points(video, pred_tracks, pred_visibility, video_width=video.shape[-1], video_height=video.shape[-2], grid_size=args.grid_size)
        frame_idx = 0
        queries = torch.tensor([ [ [frame_idx, x, y] for (x, y) in points ] ]).to(DEFAULT_DEVICE).to(torch.float32)
        pred_tracks, pred_visibility = model(video, queries=queries)
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=2)
        vis.visualize(
            video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if args.backward_tracking else args.grid_query_frame,
            filename=f"{video_name}_1st_sample",
        )

        points = sample_track_points(video, pred_tracks, pred_visibility, video_width=video.shape[-1], video_height=video.shape[-2], num_samples=10, grid_size=args.grid_size)
        frame_idx = 0
        queries = torch.tensor([ [ [frame_idx, x, y] for (x, y) in points ] ]).to(DEFAULT_DEVICE).to(torch.float32)
        pred_tracks, pred_visibility = model(video, queries=queries)
        
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=2)
        vis.visualize(
            video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if args.backward_tracking else args.grid_query_frame,
            filename=f"{video_name}",
        )

        selected_frames = sample_frames(video, video_name, pred_tracks, pred_visibility, strides=[15, 60, video.shape[1]-1])
