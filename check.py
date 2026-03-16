import numpy as np
import os
import cv2

def check_pred_track_npy(npy_dir, raw_image, npy):
    """
    检查npy是否正确储存了点的信息
    """
    raw_image_path = os.path.join(npy_dir, raw_image)
    npy_path = os.path.join(npy_dir, npy)

    if not os.path.exists(raw_image_path):
        print(f"Raw image not found: {raw_image_path}")
        return
    
    if not os.path.exists(npy_path):
        print(f"Numpy file not found: {npy_path}")
        return
    
    # Load the numpy file
    pred_tracks = np.load(npy_path)
    
    # 在对应的位置画圈

    image = cv2.imread(raw_image_path)
    for track in pred_tracks:
        x, y = int(track[0]), int(track[1])
        cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    cv2.imwrite(os.path.join(npy_dir, "pred_tracks_overlay.png"), image)
            
# check_pred_track_npy(
#     npy_dir="/mnt/disk1/datasets/drag_data/selectframe/magictime_7209165856029281573_001",
#     raw_image="original_frame_149.png",
#     npy="pred_track_frame_149.npy"
# )