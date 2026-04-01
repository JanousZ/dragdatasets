import os
from pathlib import Path
from tqdm import tqdm
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

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
    
    # --- 新增功能：删除小于 500x500 的视频 ---
    if w < 500 or h < 500:
        try:
            os.remove(video_str)
            return f"已删除（尺寸过小 {w}x{h}）: {video_str}"
        except Exception as e:
            return f"删除失败 {video_str}: {e}"
    # ---------------------------------------

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

def process_videos_crop(root_dir, num_workers=32):
    """
    递归查找 root_dir 下所有视频，检查宽高是否为 8 的倍数，
    若不是则进行中心裁剪并覆盖原文件。多线程加速。
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
    all_files = list(Path(root_dir).rglob("*"))
    video_files = [f for f in all_files if f.suffix.lower() in video_extensions and "_seg" not in str(f)]
    print(f"共发现 {len(video_files)} 个视频文件，开始检查...")
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_video_crop, video_path): video_path for video_path in video_files}
        for f in tqdm(as_completed(futures), total=len(futures), desc="裁剪视频进度(多线程)"):
            res = f.result()
            if res:
                print(res)

def split_video_by_frames(video_path, frame_counts=[20, 60]):
    video_path = Path(video_path)
    video_name = video_path.stem

    # 同时获取总帧数和fps
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,r_frame_rate',
        '-of', 'json', str(video_path)
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        import json
        info = json.loads(result.stdout)['streams'][0]
        
        # 解析fps，格式通常是 "30/1" 或 "30000/1001"
        num, den = map(int, info['r_frame_rate'].split('/'))
        fps = num / den
        
        total_frames = int(info.get('nb_frames', 0))
        if total_frames == 0:
            # nb_frames 有时不准，fallback 用 ffprobe count
            probe_cmd2 = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-count_packets', '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0', str(video_path)
            ]
            total_frames = int(subprocess.check_output(probe_cmd2, text=True).strip())
    except Exception as e:
        return f"Probe Fail: {video_path} | {e}"

    for count in frame_counts:
        num_segments = total_frames // count
        if num_segments == 0:
            continue

        output_dir = os.path.join(str(video_path.parent), video_name)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(num_segments):
            start_frame = i * count
            start_time = start_frame / fps          # 换算为秒
            duration = count / fps                  # 片段时长（秒）

            output_file = os.path.join(output_dir, f"{video_name}_seg{i:03d}_{count}f.mp4")

            # 用 -ss/-t 时间定位，快速且稳定
            command = [
                'ffmpeg', '-y',
                '-ss', f"{start_time:.6f}",         # 输入前seek，极快
                '-i', str(video_path),
                '-t', f"{duration:.6f}",
                '-vf', 'crop=floor(iw/8)*8:floor(ih/8)*8',
                '-frames:v', str(count),            # 兜底，防止浮点误差多一帧
                '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
                '-an',                              # 切片通常不需要音频
                '-movflags', '+faststart',          # moov atom 写到文件头，防损坏
                output_file
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"切片失败: {output_file}\n{e.stderr[-300:]}")
                if os.path.exists(output_file):
                    os.remove(output_file)  # 删除损坏文件

    try:
        os.remove(str(video_path))
    except Exception as e:
        return f"Done (删除原文件失败): {video_name} | {e}"
    return f"Done: {video_name}"

def process_videos_split(root_dir, num_workers=32):
    """
    递归查找 root_dir 下所有视频，检查宽高是否为 8 的倍数，
    若不是则进行中心裁剪并覆盖原文件。多线程加速。
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
    all_files = list(Path(root_dir).rglob("*"))
    video_files = [f for f in all_files if f.suffix.lower() in video_extensions and "_seg" not in str(f)]
    print(f"共发现 {len(video_files)} 个视频文件，开始检查...")
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(split_video_by_frames, video_path, frame_counts=[20, 60]): video_path for video_path in video_files}
        for f in tqdm(as_completed(futures), total=len(futures), desc="裁剪视频长度(多线程)"):
            res = f.result()
            # if res:
            #     print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Score Filter to JSONL")
    parser.add_argument("--root_dir", type=str, required=True, help="数据集根目录")
    args = parser.parse_args()
    # 先处理视频裁剪，确保后续分析的视频尺寸符合要求
    process_videos_crop(args.root_dir) 
    # 时间裁剪，15帧/60帧一段
    process_videos_split(args.root_dir)