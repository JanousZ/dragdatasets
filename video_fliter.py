import os
import pandas as pd
from pathlib import Path
import numpy as np
import json
import re
import torch
from multiprocessing import Process, Queue
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from multiprocessing import Pool
import cv2
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置路径 ---
CSV_PATH = "/mnt/disk1/datasets/drag_data/OpenVid-1M.csv"
VIDEO_ROOT = "/mnt/disk1/datasets/drag_data" # 递归扫描的起点

# 单个视频处理函数
def process_single_video_crop(video_path):
    video_str = str(video_path)
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0',
        video_str
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        if not result.stdout.strip():
            return f"跳过（无法读取尺寸）: {video_str}"
        w, h = map(int, result.stdout.strip().split('x'))
    except Exception:
        return f"跳过（无法读取尺寸）: {video_str}"
    if w % 8 == 0 and h % 8 == 0:
        return None  # 已经是 8 的倍数，跳过
    target_w = w - (w % 8)
    target_h = h - (h % 8)
    temp_output = video_str + ".temp_crop.mp4"
    crop_cmd = [
        'ffmpeg', '-y', '-i', video_str,
        '-vf', f"crop={target_w}:{target_h}",
        '-c:v', 'libx264', '-crf', '18',
        '-c:a', 'copy', '-loglevel', 'error',
        temp_output
    ]
    try:
        subprocess.run(crop_cmd, check=True)
        os.replace(temp_output, video_str)
        return f"裁剪完成: {video_str} ({w}x{h} -> {target_w}x{target_h})"
    except subprocess.CalledProcessError:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return f"处理失败: {video_str}"

# 多线程版本

def process_videos_crop(root_dir, num_workers=128):
    """
    递归查找 root_dir 下所有视频，检查宽高是否为 8 的倍数，
    若不是则进行中心裁剪并覆盖原文件。多线程加速。
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
    all_files = list(Path(root_dir).rglob("*"))
    video_files = [f for f in all_files if f.suffix.lower() in video_extensions]
    print(f"共发现 {len(video_files)} 个视频文件，开始检查...")
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_video_crop, video_path): video_path for video_path in video_files}
        for f in tqdm(as_completed(futures), total=len(futures), desc="裁剪视频进度(多线程)"):
            res = f.result()
            if res:
                print(res)

def prepare_processing_static_list(csv_path, video_root):
    """
    1. 递归扫描视频文件
    2. 结合 CSV 的 caption 和 camera_motion 进行初步筛选
    """
    print(f"正在加载 CSV 文件: {csv_path}...")
    # 读入 CSV，确保包含 videoid, content (caption), 和 camera_motion 列
    df = pd.read_csv(csv_path)
    
    # 预筛选：只保留 camera_motion 为 static 的记录
    # 提示：如果 CSV 中该列有空格或大小写不一，使用 strip() 和 lower()
    df['camera motion'] = df['camera motion'].astype(str).str.strip().str.lower()
    static_df = df[df['camera motion'] == 'static']
    
    # 建立映射：videoid -> caption
    # 注意：如果 CSV 里的 videoid 不带后缀，我们需要手动补上 .mp4 或根据实际情况匹配
    # 这里假设匹配逻辑需要 videoid 与磁盘文件名（含后缀）完全一致
    caption_map = dict(zip(static_df['video'].astype(str), static_df['caption']))
    
    print(f"正在递归扫描根目录: {video_root} ...")
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov')
    
    data_to_analyze = []
    found_but_not_static = 0
    match_count = 0

    # 使用 os.walk 进行递归查找
    for root, dirs, files in os.walk(video_root):
        for file in files:
            if file.endswith(video_extensions):
                # 这里的 file 就是带后缀的文件名，如 "example_001.mp4"
                v_name_with_ext = file
                
                if v_name_with_ext in caption_map:
                    data_to_analyze.append({
                        'video': v_name_with_ext, # 只保留文件名，不含路径
                        'caption': caption_map[v_name_with_ext],
                        'full_path': os.path.join(root, file) # 保留完整路径供后续读取视频
                    })
                    match_count += 1
                else:
                    # 记录一下：在 CSV 中但被过滤掉（非 static）的情况
                    found_but_not_static += 1
                    # 删除该文件（如果需要的话，谨慎操作！）

                    # if v_name_with_ext in df['video'].astype(str).values:
                    #     os.remove(os.path.join(root, file))

    print(f"筛选完成！")
    print(f"- 成功匹配 (Static + 有描述): {match_count} 条")
    print(f"- 跳过 (非 Static 或不在 CSV 中): {found_but_not_static} 条")
    
    return pd.DataFrame(data_to_analyze)

FILTER_PROMPT = """
# Role
You are an expert in Computer Vision and Graphics, specializing in data curation for "Point-based Video Drag-editing" tasks.

# Task Overview
Your objective is to evaluate whether a video is suitable as training data for drag-based generation by synthesizing the "Video Content" and the provided "Text Description (Caption)." Even if the background is complex, the video is considered qualified as long as the "primary evolving subject" is distinct and its motion can be guided by "Start Points -> Target Points" displacements.

# Assessment Dimensions
Please provide a comprehensive judgment based on the following four dimensions:

1. **Main Subject & Background Stability**: 
   - Does the primary dynamic change concentrate on a single, prominent subject or a single part of the background?
   - **Background Requirement**: The background should ideally remain static (Stationary) to ensure the motion is derived from the subject itself. 
   - **Exception**: If the "primary change" intentionally occurs in the background (e.g., landscape morphing, clouds moving), it is also acceptable.
2. **Subject Category**:
   - Must fall into the following hierarchy: [Human, Animal, Daily Items, Nature/Landscape, Others].
   - Provide a specific detailed description (e.g., "A man wearing a heavy coat," "A rotating ceramic cup").
3. **Motion Taxonomy**:
   - Identify the dynamic attributes of the video. Select the most one appropriate category: [Translation (Position Change), Pose Change, Deformation, Rotation, Scaling, Boundary Expansion, Camera movement, Others].
4. **Point-based Drag Feasibility**:
   - Core Judgment: Can this motion be effectively represented by the displacement of a "Set of Start Points -> Set of Target Points"?
   - Scoring: 1-10 (A score of 7 or above indicates high-quality data).

# Output Format (Strict JSON)
Output ONLY the JSON result. Do not include any reasoning process to ensure the output is directly parsable:
{
  "is_single_changing_subject": bool,
  "subject": {
    "category": "Human/Animal/Daily Items/Nature/Landscape/Others",
    "detail": "string"
  },
  "motion_semantics": ["string"],
  "drag_feasibility": {
    "suitable": bool,
    "score": int,
    "reason": "string"
  }
}

Following are the video content and its caption for your analysis:
"""

def run_multigpu_labeling(df,
                          model_id = "/mnt/disk1/models/Qwen/Qwen2.5-VL-7B-Instruct", 
                          output_path="labeled_data.jsonl"):
    """
    主函数：将任务分发到 3 张显卡并行处理。
    """
    # 将 DataFrame 均匀切分成 3 份
    num_gpus = 3
    chunks = [df.iloc[i::num_gpus] for i in range(num_gpus)]
    
    # 使用队列来管理结果收集（可选，这里直接让每个进程写自己的临时文件更稳定）
    processes = []
    
    print(f"🚀 Starting parallel labeling on {num_gpus} GPUs...")
    
    for gpu_id in range(num_gpus):
        p = Process(
            target=_worker_inference, 
            args=(gpu_id, chunks[gpu_id], model_id, output_path)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    print(f"✅ All workers finished. Data saved to {output_path}")

def _worker_inference(gpu_id, data_chunk,
                      model_id, 
                      final_output_path):
    """
    工作进程：负责单张显卡的加载与推理。
    """
    # 1. 显卡绑定
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    # 2. 模型加载 (每个进程加载一个完整的 7B 模型)
    # A6000 有 48G，7B 模型占约 15-20G，完全跑得下
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map={"": device}, # 强制映射到当前进程的显卡
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    # 3. 结果保存路径 (每个进程先写临时文件，避免写入冲突)
    temp_output = f"{final_output_path}.gpu{gpu_id}"
    
    with open(temp_output, 'w', encoding='utf-8') as f:
        # 只在 GPU 0 上显示进度条
        iterator = tqdm(data_chunk.iterrows(), total=len(data_chunk), desc=f"GPU {gpu_id}") if gpu_id == 0 else data_chunk.iterrows()
        
        for _, row in iterator:
            try:
                # 构造消息
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "system", "text": FILTER_PROMPT},
                            {"type": "video", "video": row['full_path'], "fps": 1.0, "max_pixels": 360 * 480},
                            {"type": "text", "text": f"Reference Caption: {row['caption']}\n"}
                        ],
                    }
                ]
                
                # 预处理
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs, 
                    padding=True, return_tensors="pt"
                ).to(device)

                # 推理
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=512)
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

                # 解析 JSON
                json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
                analysis = json.loads(json_match.group()) if json_match else {"error": "no_json", "raw": output_text}
                
                # 写入结果
                result = {
                    "video": row['video'],
                    "full_path": row['full_path'],
                    "analysis": analysis
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                
            except Exception as e:
                # 记录错误跳过，保证进程不中断
                error_res = {"video": row['video'], "error": str(e)}
                f.write(json.dumps(error_res) + '\n')

    # 4. 合并逻辑（简单追加到主文件）
    _merge_to_main(temp_output, final_output_path)

def _merge_to_main(temp_file, main_file):
    """
    将临时文件内容合并到主输出文件。
    """
    import shutil
    with open(main_file, 'ab') as dest:
        with open(temp_file, 'rb') as src:
            shutil.copyfileobj(src, dest)
    os.remove(temp_file)

# process_videos_crop(VIDEO_ROOT) # 先处理视频裁剪，确保后续分析的视频尺寸符合要求

matched_df = prepare_processing_static_list(CSV_PATH, VIDEO_ROOT)

# 预览结果
if not matched_df.empty:
    print("\n[预览待分析数据]:")
    print(matched_df[['video', 'caption']].head())
else:
    print("未找到符合 static 筛选条件的视频，请检查 CSV 列名或内容。")

# test_df = matched_df.head(9) # 取前 9 条进行测试
# run_multigpu_labeling(test_df)