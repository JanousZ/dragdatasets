import os
import json
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import imageio
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import torch.multiprocessing as mp
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import argparse

_model_cache = {}

def _init_raft(gpu_ids):
    global _model_cache
    current_process = mp.current_process()
    try:
        # 获取进程编号来分配 GPU
        worker_id = int(current_process.name.split('-')[-1]) - 1
    except:
        worker_id = 0
        
    target_gpu = gpu_ids[worker_id % len(gpu_ids)]
    device = f"cuda:{target_gpu}"
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights).to(device).eval()
    transforms = weights.transforms()
    
    _model_cache['model'] = model
    _model_cache['transforms'] = transforms
    _model_cache['device'] = device

def motionscore_videofilter(video_path, min_motion_score=2.0, max_motion_score=1000.0, sample_interval=5):
    model = _model_cache['model']
    transforms = _model_cache['transforms']
    device = _model_cache['device']

    try:
        reader = imageio.get_reader(video_path)
        # 优化：直接转为 float Tensor，减少后续转换开销
        frames = [torch.from_numpy(frame).permute(2, 0, 1).float().to(device) 
                  for i, frame in enumerate(reader) if i % sample_interval == 0]
        reader.close()
    except Exception:
        return str(video_path), 0.0

    if len(frames) < 2:
        return str(video_path), 0.0
    
    total_motion = 0.0
    count = 0
    with torch.no_grad():
        for i in range(len(frames) - 1):
            # 使用 TF.resize 替代 F.resize
            img1 = TF.resize(frames[i], [512, 512], antialias=True)
            img2 = TF.resize(frames[i+1], [512, 512], antialias=True)
            
            # RAFT 预处理
            img1, img2 = transforms(img1, img2)
            
            # 模型推理 [1, 2, H, W]
            flow = model(img1.unsqueeze(0), img2.unsqueeze(0))[-1][0]
            
            # 计算平均运动模长
            mag = torch.norm(flow, dim=0).mean().item()
            total_motion += mag
            count += 1
            del img1, img2, flow
        
        del frames
        torch.cuda.empty_cache()

    avg_motion = total_motion / count if count > 0 else 0.0

    if avg_motion < min_motion_score or avg_motion > max_motion_score:
        return str(video_path), 0.0
        
    return str(video_path), avg_motion

def run_quality_scoring(args):
    root_path = Path(args.root_dir)
    # 增加后缀兼容性
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    all_videos = [f for f in root_path.rglob("*") 
                  if f.suffix.lower() in video_extensions and "seg" in f.name]
    
    if not all_videos:
        print(f"未在 {args.root_dir} 下找到有效视频。")
        return

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    raw_results = []
    # 使用进程池处理
    with mp.Pool(processes=args.num_workers, initializer=_init_raft, initargs=(args.gpu_ids,)) as pool:
        # imap_unordered 对大批量任务更友好
        pbar = tqdm(pool.imap_unordered(motionscore_videofilter, all_videos), 
                    total=len(all_videos), desc=f"Scoring on GPUs: {args.gpu_ids}")
        for res in pbar:
            raw_results.append(res)

    # 按父文件夹归档（转为相对路径）
    video_groups = defaultdict(list)
    for v_path_str, score in raw_results:
        if score > 0:
            rel_path = os.path.relpath(v_path_str, args.root_dir)
            v_path_obj = Path(v_path_str)
            group_key = str(v_path_obj.parent)
            video_groups[group_key].append((rel_path, score))

    # 写入 JSONL
    saved_count = 0
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        # 按照路径排序，保证输出顺序相对稳定
        for group_path in sorted(video_groups.keys()):
            items = video_groups[group_path]
            # 每个独立路径下的文件夹取最高分的 top_k 个
            top_k = sorted(items, key=lambda x: x[1], reverse=True)[:args.top_k]
            for v_path, score in top_k:
                line = json.dumps({"video_path": v_path, "motion_score": score}, ensure_ascii=False)
                f.write(line + "\n")
                saved_count += 1

    print(f"\n筛选完成！共保存 {saved_count} 个视频路径到 {args.output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default="motionscore_select.jsonl")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[0])
    parser.add_argument("--top_k", type=int, default=3, help="每个文件夹取运动分数最高的 k 个视频")
    
    args = parser.parse_args()
    run_quality_scoring(args)