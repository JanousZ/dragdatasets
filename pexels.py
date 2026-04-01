import os
import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor

# --- 核心配置 ---
PEXELS_API_KEY = 'xZ5YrEIVoaUsUzDNJ4S46iLbT9FWTGLdPpFkQkzQ4KyIHOTOhjQrJrlc'
BASE_URL = "https://api.pexels.com/videos/search"
HEADERS = {"Authorization": PEXELS_API_KEY}

JSON_FILE = "target_distribution.json"  # 你的配置文件名
BASE_SAVE_PATH = "./A1_Dataset"        # 数据集根目录
TOTAL_NEEDED_PER_KW = 5               # 每个关键词的下载目标数
MAX_WORKERS = 12                       # 3x6000服务器建议开到12-16线程

def get_best_video_url(video_data):
    files = video_data.get('video_files', [])
    # 优先选择 720p 到 1080p 之间的视频，保证实验质量
    for f in files:
        if 512 <= f.get('height', 0) <= 1080 and 512 <= f.get('weight', 0) <= 1080:
            return f['link']
    return files[0]['link'] if files else None

def download_single_video(args):
    url, filename, video_id = args
    if os.path.exists(filename):
        return
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
        print(f"  [√] Saved: {video_id}")
    except Exception as e:
        print(f"  [X] Failed {video_id}: {e}")

def fetch_and_queue(path, kw):
    """搜索并获取下载任务列表"""
    os.makedirs(path, exist_ok=True)
    tasks = []
    page = 1
    found = 0
    
    print(f"\n>>> 检索中: [{kw}] -> 存储至: {path}")
    
    while found < TOTAL_NEEDED_PER_KW:
        params = {"query": kw, "per_page": 80, "page": page}
        try:
            resp = requests.get(BASE_URL, headers=HEADERS, params=params)
            if resp.status_code != 200: break
            
            videos = resp.json().get('videos', [])
            if not videos: break

            for v in videos:
                if found >= TOTAL_NEEDED_PER_KW: break
                url = get_best_video_url(v)
                if url:
                    # 使用视频ID作为文件名，防止重复
                    save_name = os.path.join(path, f"{v['id']}.mp4")
                    tasks.append((url, save_name, v['id']))
                    found += 1
            page += 1
            time.sleep(0.5) 
        except Exception: break
    return tasks

def recursive_search_and_download(data, current_path):
    """
    核心递归逻辑：
    - 如果是 dict: 继续往下游走，拼接路径名
    - 如果是 list: 视为搜索词列表，开始在该路径下载
    """
    all_tasks = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            # 自动处理 key 中的空格，方便 Linux 文件系统管理
            safe_key = key.replace(" ", "_")
            new_path = os.path.join(current_path, safe_key)
            recursive_search_and_download(value, new_path)
            
    elif isinstance(data, list):
        # 这一层已经是关键词了
        for kw in data:
            tasks = fetch_and_queue(current_path, kw)
            # 立即下载当前关键词的任务
            if tasks:
                print(f"正在并行下载 {len(tasks)} 个视频...")
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    executor.map(download_single_video, tasks)

if __name__ == "__main__":
    # 1. 读入你写好的 JSON
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            target_distribution = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {JSON_FILE}")
        exit()

    start_time = time.time()
    
    # 2. 启动递归下载
    recursive_search_and_download(target_distribution, BASE_SAVE_PATH)
    
    total_time = (time.time() - start_time) / 60
    print(f"\n[任务完成] 总耗时: {total_time:.2f} 分钟")