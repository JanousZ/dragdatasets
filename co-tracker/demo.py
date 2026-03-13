# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import json

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def sample_track_points(pred_tracks, pred_visibility, video_width=None, video_height=None, num_samples=None, grid_size=20):

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
    
    # 1. 计算每个点的总位移
    displacement = calculate_total_displacement(pred_tracks, mode='path_length')   #[N]
    # 2. 过滤掉视频边界外的点
    if video_width is not None and video_height is not None:
        out_of_bounds_mask = filter_out_of_video(pred_tracks, video_width, video_height)
        displacement[out_of_bounds_mask] = 0  
    
    avg_motion = displacement.mean()
    std_motion = displacement.std()
    if num_samples is None:
        motion_threshold = avg_motion + 0.5 * std_motion
    else:
        motion_threshold = 0.5 * avg_motion
    significant_motion_mask = (displacement > motion_threshold)
    significant_points = pred_tracks[0, 0, significant_motion_mask]

    if num_samples is None:
        radius_x = video_width // grid_size
        radius_y = video_height // grid_size
        k = 2 + radius_x * radius_y // 500
        #offsets_x = torch.randint(low=-radius_x, high=radius_x + 1, size=(significant_points.shape[0], k, 1)).to(significant_points.device)
        #offsets_y = torch.randint(low=-radius_y, high=radius_y + 1, size=(significant_points.shape[0], k, 1)).to(significant_points.device)
        offsets_x = (torch.rand(significant_points.shape[0], k, 1, device=significant_points.device) * 2 - 1) * radius_x
        offsets_y = (torch.rand(significant_points.shape[0], k, 1, device=significant_points.device) * 2 - 1) * radius_y
        offsets = torch.cat([offsets_x, offsets_y], dim=-1)
        new_sampled_points = significant_points.unsqueeze(1) + offsets
        new_sampled_points = new_sampled_points.view(-1, 2)
        unique_points = torch.unique(new_sampled_points, dim=0)
        unique_points[:, 0] = unique_points[:, 0].clamp(0, video_width - 1)
        unique_points[:, 1] = unique_points[:, 1].clamp(0, video_height - 1)
        unique_points = torch.unique(new_sampled_points, dim=0)
        return unique_points.cpu().numpy().tolist()
    else:
        if significant_points.shape[0] > num_samples:
            indices = torch.randperm(significant_points.shape[0])[:num_samples]
            significant_points = significant_points[indices]
        return significant_points.cpu().numpy().tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="/home/yanzhang/Video-T1/final_results/A_charming_wooden_birdhouse,_painted_in_vibrant_hues_of_red_and_blue,_hangs_gracefully_from_a_sturdy.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--video_json",
        #default="/home/yanzhang/dragdatasets/labeled_data.jsonl",
        default=None,
        help="video paths",
    )
    parser.add_argument(
        "--video_dir",
        default="/mnt/disk1/datasets/drag_data/selectdata",
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

        if "magictime" not in video_path:
            continue

        video = read_video_from_path(video_paths[np.random.randint(len(video_paths))])
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
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

        points = sample_track_points(pred_tracks, pred_visibility, video_width=video.shape[-1], video_height=video.shape[-2], grid_size=args.grid_size)
        frame_idx = 0
        queries = torch.tensor([ [ [frame_idx, x, y] for (x, y) in points ] ]).to(DEFAULT_DEVICE).to(torch.float32)
        pred_tracks, pred_visibility = model(video, queries=queries)
        points = sample_track_points(pred_tracks, pred_visibility, video_width=video.shape[-1], video_height=video.shape[-2], num_samples=15, grid_size=args.grid_size)
        frame_idx = 0
        queries = torch.tensor([ [ [frame_idx, x, y] for (x, y) in points ] ]).to(DEFAULT_DEVICE).to(torch.float32)
        pred_tracks, pred_visibility = model(video, queries=queries)
        
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=2)
        vis.visualize(
            video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if args.backward_tracking else args.grid_query_frame,
            filename=f"{video_names[i]}",
        )