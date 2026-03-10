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
from start_point import MotionSampler,process_video_pair

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="/home/yanzhang/Video-T1/final_results/A_close-up_shot_captures_a_kangaroo_in_its_natural_habitat,_its_fur_a_rich_blend_of_earthy_browns_an.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--video_json",
        #default="/home/yanzhang/dragdatasets/labeled_data.jsonl",
        default=None,
        help="video paths",
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
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
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
    else:
        video_paths = [args.video_path]
        video_names = [args.video_path.split("/")[-1].split(".")[0]]

    for i,video_path in enumerate(video_paths):
    # load the input video frame by frame
        video = read_video_from_path(video_path)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        #segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
        #segm_mask = torch.from_numpy(segm_mask)[None, None]

        video = video.to(DEFAULT_DEVICE)

        points = process_video_pair(video_path, stride=50)

        # points = [ (128,128), (172,172), (256,256)]  # (w,h)
        frame_idx = 0  # 这些点所在的帧
        queries = torch.tensor([ [ [frame_idx, x, y] for (x, y) in points ] ]).to(DEFAULT_DEVICE).to(torch.float32)  # shape (1, N, 3)
        pred_tracks, pred_visibility = model(video, queries=queries)

        # pred_tracks, pred_visibility = model(
        #     video,
        #     grid_size=args.grid_size,
        #     grid_query_frame=args.grid_query_frame,
        #     backward_tracking=args.backward_tracking,
        #     # segm_mask=segm_mask
        # )
        print("computed")
        print(f"pred_tracks shape: {pred_tracks.shape}, pred_visibility shape: {pred_visibility.shape}")

        # save a video with predicted tracks
        seq_name = video_path.split("/")[-1]
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=2)
        vis.visualize(
            video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if args.backward_tracking else args.grid_query_frame,
            filename=f"{video_names[i]}.mp4",
        )