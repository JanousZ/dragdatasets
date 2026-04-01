import os
import subprocess
from scenedetect import detect, ContentDetector

# --- 配置区：对应你的 A1 目标分布 ---
TARGET_DISTRIBUTION = {
    "plant_growth": ["plant growth time-lapse static", "flower blooming macro timelapse"],
    "daily_objects": ["opening drawer static camera", "rotating chair 360", "folding clothes timelapse"],
    "animal_motion": ["cat yawning close up", "dog head tilt static", "bird wings flapping slow motion"],
    "human_pose": ["human stretching exercise static", "hand gestures close up static"]
}

# 下载限制
MAX_DOWNLOADS_PER_TAG = 4  # 每个关键词下载多少个
MIN_DURATION = 5            # 最短5秒
MAX_DURATION = 20           # 最长20秒，方便后续剪辑成5s
OUTPUT_DIR = "./my_drag_dataset"

def check_scene_cuts(video_path):
    """验证视频是否为单镜头（过滤 Q6）"""
    try:
        scene_list = detect(video_path, ContentDetector(threshold=30.0))
        # 如果检测到的场景数大于1，说明有镜头切换
        return len(scene_list) <= 1
    except Exception as e:
        print(f"检测失败: {e}")
        return False

def download_distribution():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for category, keywords in TARGET_DISTRIBUTION.items():
        cat_path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(cat_path, exist_ok=True)
        
        for kw in keywords:
            print(f"\n正在抓取类别 [{category}] 关键词: {kw}...")
            
            # 构造 yt-dlp 命令
            # --match-filter: 过滤时长
            # --max-downloads: 限制数量
            # --postprocessor-args: 强制转换格式方便处理
            node_path = "/usr/bin/node"
            SEARCH_POOL = 100  # 提高搜索池，因为 512p 以上且符合时长的视频更少
            TARGET_COUNT = 4

            cmd = [
                "yt-dlp",
                # 1. 显式指定 JS 运行时路径 (必须放在搜索参数之前)
                "--js-runtimes", f"node:{node_path}", 
                
                # 2. 搜索逻辑：扩大搜索池，但限制最终下载数
                f"ytsearch{SEARCH_POOL}:{kw}",
                "--max-downloads", str(TARGET_COUNT),
                
                # 3. 过滤逻辑：时长 + 分辨率 (确保短边 >= 512)
                "--match-filter", f"duration >= {MIN_DURATION} & duration <= {MAX_DURATION}",
                "--format", "bestvideo[height>=512][height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height>=512][ext=mp4]/best",
                
                # 4. 输出与基础配置
                "--output", f"{cat_path}/%(id)s.%(ext)s",
                "--no-warnings",   # 屏蔽已安装但仍弹出的冗余警告
                "--ignore-errors", # 遇到单个视频解析失败自动跳过下一个
                "--no-post-overwrites",
                "--force-ipv4"
            ]
            subprocess.run(cmd)

        # 下载完成后进行 Q6 过滤
        # print(f"正在对 {category} 进行切镜过滤...")
        # for video_file in os.listdir(cat_path):
        #     v_path = os.path.join(cat_path, video_file)
        #     if not check_scene_cuts(v_path):
        #         print(f"发现镜头切换，已删除: {video_file}")
        #         os.remove(v_path)

if __name__ == "__main__":
    download_distribution()
    print("\n抓取任务完成！视频已按类别存放在 my_drag_dataset 文件夹中。")