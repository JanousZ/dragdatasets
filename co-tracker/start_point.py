import torch
import numpy as np
import imageio
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

class MotionSampler:
    def __init__(self, device="cuda"):
        self.device = device
        # 1. 加载轻量化 RAFT 模型，适合大规模处理
        self.weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=self.weights).to(device).eval()
        self.transforms = self.weights.transforms()

    def preprocess(self, img1, img2):
        # 将 numpy 数组转为 torch tensor 并进行 RAFT 预处理
        img1 = torch.from_numpy(img1).permute(2, 0, 1).to(self.device)
        img2 = torch.from_numpy(img2).permute(2, 0, 1).to(self.device)
        return self.transforms(img1, img2)

    @torch.no_grad()
    def get_flow_magnitude(self, frame1, frame2):
        """计算两帧间的光流模长"""
        img1_batch, img2_batch = self.preprocess(frame1, frame2)
        
        # RAFT 推理
        list_of_flows = self.model(img1_batch.unsqueeze(0), img2_batch.unsqueeze(0))
        flow = list_of_flows[-1][0]  # 取最后一次迭代的 [2, H, W]
        
        # 计算模长 sqrt(u^2 + v^2)
        magnitude = torch.norm(flow, dim=0)
        return flow, magnitude

    def sample_significant_points(self, magnitude, num_samples=50, grid_size=16, threshold_quantile=0.95):
        """
        优化后的采样策略：
        1. 动态阈值：保留运动显著区域。
        2. 随机网格采样：在每个网格内随机选点，消除方正感。
        3. 运动强度加权：让模长更大的点有更高概率被选中（可选）。
        """
        H, W = magnitude.shape
        margin = 20
        
        # 1. 筛选显著区域
        inner_mag = magnitude[margin:H-margin, margin:W-margin]
        if inner_mag.numel() == 0: return np.array([])
        
        thresh = torch.quantile(inner_mag, threshold_quantile)
        mask = magnitude > thresh
        
        # 排除边缘
        edge_mask = torch.zeros_like(mask)
        edge_mask[margin:H-margin, margin:W-margin] = 1
        mask = mask & edge_mask

        candidate_coords = torch.nonzero(mask) # [N, 2] -> (y, x)
        if len(candidate_coords) == 0: return np.array([])

        # --- 核心修改：打破方正感 ---
        # 2. 将点分配到网格，但不再取“第一个”，而是打乱顺序后取“第一个”
        # 这样每个网格里选中的点就是随机的，而不是左上角的点
        grid_y = candidate_coords[:, 0] // grid_size
        grid_x = candidate_coords[:, 1] // grid_size
        grid_indices = grid_y * (W // grid_size + 1) + grid_x

        # 随机打乱候选点顺序
        perm = torch.randperm(len(candidate_coords))
        shuffled_coords = candidate_coords[perm]
        shuffled_grid_indices = grid_indices[perm]

        # 在打乱后的数组中取唯一值，相当于在每个网格内随机抽样
        _, first_occurrence = np.unique(shuffled_grid_indices.cpu().numpy(), return_index=True)
        sampled_pts = shuffled_coords[first_occurrence]

        # 3. 最终数量控制
        if len(sampled_pts) > num_samples:
            # 再次打乱以确保最终点集的随机性
            final_idx = torch.randperm(len(sampled_pts))[:num_samples]
            sampled_pts = sampled_pts[final_idx]
            
        return sampled_pts.flip(1).cpu().numpy() # 返回 (x, y)
    
    def visualize_comparison(self, frame1, frame_target, flow, magnitude, sampled_points, save_path="pair_check.png"):
        """
        三栏对比可视化：
        1. 第一帧 + 采样 Handle Points
        2. 光流热力图
        3. 目标帧 + 预测的对应 Target Points
        """
        if len(sampled_points) == 0:
            print("警告：没有采集到样点，跳过可视化。")
            return

        plt.figure(figsize=(20, 6))

        # --- 1. Source Frame & Handle Points ---
        plt.subplot(1, 3, 1)
        plt.imshow(frame1)
        # sampled_points 是 (x, y) 格式
        plt.scatter(sampled_points[:, 0], sampled_points[:, 1], 
                    c='red', s=30, edgecolors='white', label='Handle')
        plt.title("Frame 1 (Source) & Handle Points")
        plt.axis('off')
        plt.legend(loc='upper right')

        # --- 2. Motion Magnitude ---
        plt.subplot(1, 3, 2)
        mag_np = magnitude.detach().cpu().numpy()
        plt.imshow(mag_np, cmap='magma')
        plt.title("Motion Magnitude (Flow)")
        plt.axis('off')

        # --- 3. Target Frame & Projected Points ---
        plt.subplot(1, 3, 3)
        plt.imshow(frame_target)
        
        # 通过光流计算这些点在第二帧的理论位置
        # flow 是 [2, H, W]，我们需要索引 (y, x) 位置的 (du, dv)
        flow_np = flow.detach().cpu().numpy()
        
        # 提取对应位置的光流向量
        y_idx = sampled_points[:, 1].astype(int)
        x_idx = sampled_points[:, 0].astype(int)
        
        dx = flow_np[0, y_idx, x_idx]
        dy = flow_np[1, y_idx, x_idx]
        
        # 计算 Target 位置
        target_x = sampled_points[:, 0] + dx
        target_y = sampled_points[:, 1] + dy
        
        # 画出 Target 点
        plt.scatter(target_x, target_y, 
                    c='cyan', s=30, edgecolors='white', label='Target (Flow-based)')
        
        # 画箭头显示运动趋势
        for i in range(len(sampled_points)):
            plt.arrow(sampled_points[i, 0], sampled_points[i, 1], dx[i], dy[i], 
                      color='yellow', head_width=5, alpha=0.5)

        plt.title("Frame Target & Movement Vectors")
        plt.axis('off')
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"对比可视化已保存至: {save_path}")

# --- 使用示例 ---
def process_video_pair(video_path, stride=50, num_samples=15):
    sampler = MotionSampler(device="cuda")
    
    # 读取视频
    reader = imageio.get_reader(video_path)
    frames = [f for i, f in enumerate(reader) if i < stride + 1]
    
    frame1 = frames[0]
    frame_target = frames[stride] # 跨帧采样，效果更好
    
    # 1. 计算光流
    flow, mag = sampler.get_flow_magnitude(frame1, frame_target)
    
    # 2. 采样处理点 (Handle Points)
    handle_points = sampler.sample_significant_points(mag, num_samples=num_samples)
    
    print(f"在视频中采样了 {len(handle_points)} 个显著运动点。")
    sampler.visualize_comparison(frames[0], frames[stride], flow, mag, handle_points, save_path="pair_check.png")
    return handle_points

# 假设你已经解压了视频
# pts = process_video_pair("/home/yanzhang/Video-T1/final_results/A_close-up_shot_captures_a_kangaroo_in_its_natural_habitat,_its_fur_a_rich_blend_of_earthy_browns_an.mp4")
