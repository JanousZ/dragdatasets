import os
import requests
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://api.pexels.com/videos/search"
MIN_FILE_SIZE = 10 * 1024  # 小于 10KB 视为下载不完整

# --- 下载统计 ---
stats = {"success": 0, "failed": 0, "skipped": 0, "filtered_duration": 0}


def get_best_video_url(video_data):
    files = video_data.get('video_files', [])
    for f in files:
        if 512 <= f.get('height', 0) <= 1080 and 512 <= f.get('width', 0) <= 1080:
            return f['link']
    return files[0]['link'] if files else None


def download_single_video(args):
    url, filename, video_id = args
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
            print(f"  [X] Incomplete: {video_id}")
        else:
            stats["success"] += 1
            print(f"  [√] Saved: {video_id}")
    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        stats["failed"] += 1
        print(f"  [X] Failed {video_id}: {e}")


def api_request_with_retry(headers, params, max_retries=3):
    """带重试的 API 请求，处理 429 限频"""
    for _ in range(max_retries):
        try:
            resp = requests.get(BASE_URL, headers=headers, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"  [!] 限频，等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"  [!] API 返回 {resp.status_code}")
                break
        except Exception as e:
            print(f"  [!] 请求异常: {e}")
            time.sleep(2)
    return None


def fetch_and_queue(headers, path, kw, per_kw, max_duration):
    """搜索并获取下载任务列表"""
    os.makedirs(path, exist_ok=True)
    tasks = []
    page = 1
    found = 0

    print(f"\n>>> 检索中: [{kw}] -> 存储至: {path}")

    while found < per_kw:
        params = {"query": kw, "per_page": 80, "page": page}
        data = api_request_with_retry(headers, params)
        if not data:
            break

        videos = data.get('videos', [])
        if not videos:
            break

        for v in videos:
            if found >= per_kw:
                break
            duration = v.get('duration', 0)
            if duration > max_duration:
                stats["filtered_duration"] += 1
                continue
            url = get_best_video_url(v)
            if url:
                save_name = os.path.join(path, f"{v['id']}.mp4")
                tasks.append((url, save_name, v['id']))
                found += 1
        page += 1
        time.sleep(0.5)

    return tasks


def recursive_search_and_download(data, current_path, headers, per_kw, max_duration, max_workers):
    """
    核心递归逻辑：
    - 如果是 dict: 继续往下游走，拼接路径名
    - 如果是 list: 视为搜索词列表，开始在该路径下载
    """
    if isinstance(data, dict):
        for key, value in data.items():
            safe_key = key.replace(" ", "_")
            new_path = os.path.join(current_path, safe_key)
            recursive_search_and_download(value, new_path, headers, per_kw, max_duration, max_workers)

    elif isinstance(data, list):
        for kw in data:
            tasks = fetch_and_queue(headers, current_path, kw, per_kw, max_duration)
            if tasks:
                print(f"正在并行下载 {len(tasks)} 个视频...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(download_single_video, t) for t in tasks]
                    for f in as_completed(futures):
                        f.result()


def parse_args():
    parser = argparse.ArgumentParser(description="从 Pexels 按目标分布下载视频")
    parser.add_argument("--api_key", type=str, default="xZ5YrEIVoaUsUzDNJ4S46iLbT9FWTGLdPpFkQkzQ4KyIHOTOhjQrJrlc",
                        help="Pexels API Key")
    parser.add_argument("--json_file", type=str, default="target_distribution_v2.json",
                        help="搜索配置 JSON 文件路径")
    parser.add_argument("--save_dir", type=str, required=True
                        help="数据集保存根目录")
    parser.add_argument("--per_kw", type=int, default=20,
                        help="每个关键词的下载目标数")
    parser.add_argument("--max_workers", type=int, default=12,
                        help="并行下载线程数")
    parser.add_argument("--max_duration", type=int, default=30,
                        help="视频最大时长(秒)，超过则过滤")
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
        args.per_kw, args.max_duration, args.max_workers
    )

    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*50}")
    print(f"[任务完成] 总耗时: {total_time:.2f} 分钟")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['failed']}")
    print(f"  跳过(已存在): {stats['skipped']}")
    print(f"  过滤(超{args.max_duration}s): {stats['filtered_duration']}")