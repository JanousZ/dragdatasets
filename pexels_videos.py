import os
import requests
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# API 配置
VIDEO_BASE_URL = "https://api.pexels.com/videos/search"
IMAGE_BASE_URL = "https://api.pexels.com/v1/search"
MIN_FILE_SIZE = 10 * 1024  # 小于 10KB 视为下载不完整

# --- 下载统计 ---
stats = {"success": 0, "failed": 0, "skipped": 0, "filtered_duration": 0}

def get_best_url(data, media_type):
    """获取下载链接，根据类型区分处理"""
    if media_type == "video":
        files = data.get('video_files', [])
        # 优先寻找 512-1080 尺寸
        for f in files:
            if 512 <= f.get('height', 0) <= 1080 and 512 <= f.get('width', 0) <= 1080:
                return f['link']
        return files[0]['link'] if files else None
    else:
        # 图片处理：优先 large(~1280px)，更接近目标尺寸
        src = data.get('src', {})
        return src.get('large') or src.get('large2x') or src.get('original')

def download_file(args):
    """通用下载函数"""
    url, filename, file_id = args
    if os.path.exists(filename) and os.path.getsize(filename) > MIN_FILE_SIZE:
        stats["skipped"] += 1
        return
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        if os.path.getsize(filename) < MIN_FILE_SIZE:
            os.remove(filename)
            stats["failed"] += 1
            print(f"  [X] Incomplete: {file_id}")
        else:
            stats["success"] += 1
            print(f"  [√] Saved: {file_id}")
    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        stats["failed"] += 1
        print(f"  [X] Failed {file_id}: {e}")

def api_request_with_retry(base_url, headers, params, max_retries=3):
    for _ in range(max_retries):
        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"  [!] 限频，等待 {wait}s...")
                time.sleep(wait)
            else:
                break
        except Exception as e:
            print(f"  [!] 请求异常: {e}")
            time.sleep(2)
    return None

def fetch_and_queue(headers, path, kw, per_kw, max_duration, media_type):
    """搜索并获取任务列表"""
    os.makedirs(path, exist_ok=True)
    tasks = []
    page = 1
    found = 0
    base_url = VIDEO_BASE_URL if media_type == "video" else IMAGE_BASE_URL
    
    print(f"\n>>> 检索中 [{media_type}]: [{kw}] -> 存储至: {path}")

    while found < per_kw:
        params = {"query": kw, "per_page": 80, "page": page}
        data = api_request_with_retry(base_url, headers, params)
        if not data: break

        items = data.get('videos' if media_type == "video" else 'photos', [])
        if not items: break

        for item in items:
            if found >= per_kw: break
            
            # 视频过滤逻辑
            if media_type == "video":
                duration = item.get('duration', 0)
                if duration > max_duration:
                    stats["filtered_duration"] += 1
                    continue
            
            # --- 分辨率过滤逻辑 ---
            # image: 只要求最短边 >= 512（Pexels 原图一般都很大，不再设上限）
            if media_type == 'image':
                width = item.get('width', 0)
                height = item.get('height', 0)
                if min(width, height) < 512:
                    continue
            
            url = get_best_url(item, media_type)
            if url:
                ext = ".mp4" if media_type == "video" else ".jpg"
                save_name = os.path.join(path, f"{item['id']}{ext}")
                tasks.append((url, save_name, item['id']))
                found += 1
        page += 1
        time.sleep(0.5)
    return tasks

def recursive_search_and_download(data, current_path, headers, per_kw, max_duration, max_workers, media_type):
    if isinstance(data, dict):
        for key, value in data.items():
            safe_key = key.replace(" ", "_")
            new_path = os.path.join(current_path, safe_key)
            recursive_search_and_download(value, new_path, headers, per_kw, max_duration, max_workers, media_type)
    elif isinstance(data, list):
        for kw in data:
            tasks = fetch_and_queue(headers, current_path, kw, per_kw, max_duration, media_type)
            if tasks:
                print(f"正在并行下载 {len(tasks)} 个文件...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(download_file, t) for t in tasks]
                    for f in as_completed(futures):
                        f.result()

def parse_args():
    parser = argparse.ArgumentParser(description="从 Pexels 下载视频或图片")
    parser.add_argument("--api_key", type=str, default="xZ5YrEIVoaUsUzDNJ4S46iLbT9FWTGLdPpFkQkzQ4KyIHOTOhjQrJrlc", help="Pexels API Key")
    parser.add_argument("--json_file", type=str, default="target_distribution_v3.json")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--type", choices=["video", "image"], required=True, help="下载类型: video 或 image")
    parser.add_argument("--per_kw", type=int, default=20)
    parser.add_argument("--max_workers", type=int, default=12)
    parser.add_argument("--max_duration", type=int, default=30)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    headers = {"Authorization": args.api_key}

    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            target_distribution = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {args.json_file}")
        exit()

    start_time = time.time()

    recursive_search_and_download(
        target_distribution, args.save_dir, headers,
        args.per_kw, args.max_duration, args.max_workers, args.type
    )

    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*50}")
    print(f"[任务完成] 总耗时: {total_time:.2f} 分钟 | 类型: {args.type}")
    print(f"  成功: {stats['success']} | 失败: {stats['failed']} | 跳过: {stats['skipped']}")
    if args.type == "video":
        print(f"  过滤时长超{args.max_duration}s: {stats['filtered_duration']}")