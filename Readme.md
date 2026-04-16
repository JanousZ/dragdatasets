## 数据对形式
(src_image, tgt_image, src_points, tgt_points)
用于表示drag-style image edit，并基于Fluxkontext微调

## 从现在的数据集构建中发现问题
Q1：分布不均
超八成是人脸移动的分布，这种分布体现的是移动的一致性，但是对于其余非平移拖拽则是OOD的。

Q2：点对标注数量与用户输入存在分布差异
训练集固定了10个点对的标注，但是一般用户输入少于10个点对，甚至可能是1个点对

Q3：图像质量与多样性
训练分辨率虽然为512X512，但是清晰度上也存在区别，训练集的图像过于模糊
训练数据集最好有鲜明、多样化的背景，防止拖拽后背景变差，并且背景必须保持静态

Q4：点对标注与整体变化
点对标注无法准确反映整体变化，即在点对标注之外有未被描述的变化时，这类数据要筛掉

Q5：训练集建构规范
必须按照时间、类别等构建子数据集，子数据集随意添加、拆卸构成整体数据集，实现数据集的拆分。
数据集必须统一规范，尽量使用相对路径。

Q6：视频时长
不能出现场景切换，或者镜头切换的视频，这个也要过滤。

A1:目标分布
人
面向变化（Orientation）例如：头部旋转（左右、上下）、身体朝向、头部朝向
表情变化（Expression）例如：微笑（嘴角慢慢扬起）、皱眉（眉部变化）、张嘴过程、惊讶过程等
姿态（Pose/Body Movement）例如：四肢动作改变
手指/手部动作（Fingers/Hands）例如：手指动作改变
位置移动 假如不是以上四种变化，但是人的位置发生了明显移动

动物
面向（Orientation）例如：左右转头、抬头、低头、身体朝向改变
面部变化 例如：嘴部张开关闭变化、耳朵变化、眼睛张开闭合变化
姿态变化 例如：局部肢体移动、翅膀扇动、尾巴移动等
位置移动 假如不是以上三种变化，但是动物的位置发生了明显移动

植物
生长 
绽放 
摆动

日常用品 / 常见物品
平移（Translation）	整体移动、局部移动
旋转（Rotation）	整体旋转、局部旋转
缩放（Scaling/Size Change）	放大、缩小
形变（Deformation）	弯曲、拉伸、压缩、开合
开闭（Open/Close）	门、抽屉、盒子等
翻转（Flip/Invert）	上下翻转、左右翻转

A2：数据来源
A2.1:混合分布数据集筛选，从市面上的通用视频数据集，筛选出：
静态镜头（背景不变化、不移动）
运动/变化主体单一（避免drag标注不全，导致推理时的耦合编辑）
符合目标分布（目标分布之外的视频数据不考虑）

Panda-70M
OpenVid-1M
WebVid-10M

A2.2：专门数据集筛选：
直接寻找符合目标分布的数据集，因此只需要筛选：
静态镜头（背景不变化、不移动）
运动/变化主体单一（避免drag标注不全，导致推理时的耦合编辑）

A2.3: 网络爬虫：

无论是哪种来源，必须对每个视频都进行主体及动作形式（变化形式）的标注，以便于分布统计

## 数据优化
植物类别：
1.必须找单一主体可存在的植物，避开类似cherry、grass这类
2.必须找存在外力干扰或能自身快速生成，带有瞬时变化的植物

## 自动化pipeline
1.获取原始视频
```bash 
python pexels.py --save_dir /mnt/disk1/datasets/drag_data/rawvideo/pexels_tdv2
```

2.视频进行裁剪，长宽为8的倍数，并删除长或宽小于500的视频，切分为20帧或60帧的片段，切分后删除源文件
```bash (base)
cd dragdatasets 
python crop_and_split.py --root_dir /mnt/disk1/datasets/drag_data/rawvideo/pexels_tdv2

python crop_and_split.py --root_dir /mnt/disk1/datasets/drag_data/rawvideo/OpenVid-1M
```

3.视频进行基于raft的运动分数初筛，每个视频最终选取topk个切分片段
root_dir命名规范：总数据集/rawvideo/子数据集
output_jsonl命名规范：总数据集/rawvideo/子数据集/子数据集_ms.jsonl
```bash
cd dragdatasets
python motionscore_filter.py --root_dir /mnt/disk1/datasets/drag_data/rawvideo/pexels_tdv2 --gpu_ids 0 --output_jsonl /mnt/disk1/datasets/drag_data/rawvideo/pexels_tdv2/pexels_tdv2_ms.jsonl --num_workers 32  --top_k 3

python motionscore_filter.py --root_dir /mnt/disk1/datasets/drag_data/rawvideo/OpenVid-1M --gpu_ids 0 --output_jsonl /mnt/disk1/datasets/drag_data/rawvideo/OpenVid-1M/OpenVid-1M_ms.jsonl --num_workers 32  --top_k 3
```

4.使用co-track对3中过滤的视频进行点标注，同时过滤一部分动态相机视频
output_root_dir命名规范：总数据集/selectframes/子数据集
video_jsonl命名规范：总数据集/rawvideo/子数据集/子数据集_ms.jsonl
```bash
cd dragdatasets/co-tracker
python demo.py --offline --backward_tracking --gpu_id 0 --output_root_dir /mnt/disk1/datasets/drag_data/selectframe/pexels_tdv2 --video_jsonl /mnt/disk1/datasets/drag_data/rawvideo/pexels_tdv2/pexels_tdv2_ms.jsonl --dataset_dir /mnt/disk1/datasets/drag_data/rawvideo/pexels_tdv2 --grid_size 30

python demo.py --offline --backward_tracking --gpu_id 0 --output_root_dir /mnt/disk1/datasets/drag_data/selectframe/OpenVid-1M --video_jsonl /mnt/disk1/datasets/drag_data/rawvideo/OpenVid-1M/OpenVid-1M_ms.jsonl --dataset_dir /mnt/disk1/datasets/drag_data/rawvideo/OpenVid-1M --grid_size 30
```

4.1
使用BiRefNet分离前景、背景；去除点对过多落于背景的数据对
```bash (omini)
python BiRefNet_filter.py \
  --root_dir /mnt/disk1/datasets/drag_data/selectframe/pexels_tdv2  \
  --output_jsonl /mnt/disk1/datasets/drag_data/train_json/pexels_tdv2_all.jsonl \
  --device cuda:0 \
  --bg_ratio 0.5 

python BiRefNet_filter.py \
  --root_dir /mnt/disk1/datasets/drag_data/selectframe/OpenVid-1M  \
  --output_jsonl /mnt/disk1/datasets/drag_data/train_json/OpenVid-1M_all.jsonl \
  --device cuda:0 \
  --bg_ratio 0.5 
```

4.2: 位移筛（在同一个 jsonl 上原地操作）
```bash
python displacement_filter.py \
  --jsonl /mnt/disk1/datasets/drag_data/train_json/pexels_tdv2_all.jsonl \
  --root_dir /mnt/disk1/datasets/drag_data/selectframe/pexels_tdv2 \
  --min_mean_ratio 0.01 \
  --min_std_ratio 0.01

python displacement_filter.py \
  --jsonl /mnt/disk1/datasets/drag_data/train_json/OpenVid-1M_all.jsonl \
  --root_dir /mnt/disk1/datasets/drag_data/selectframe/OpenVid-1M \
  --min_mean_ratio 0.01 \
  --min_std_ratio 0.01
```

4.3可选去重
```bash
python clean_duplicates.py --jsonl /mnt/disk1/datasets/drag_data/train_json/pexels_tdv2_all.jsonl

python clean_duplicates.py --jsonl /mnt/disk1/datasets/drag_data/train_json/OpenVid-1M_all.jsonl
```

5.人工挑选合适的pair
root_dir命名规范：总数据集/selectframes/子数据集
output_jsonl命名规范：总数据集/train_json/子数据集.jsonl
```bash
cd dragdatasets
python manual_select.py --root_dir /mnt/disk1/datasets/drag_data/selectframe/pexels_tdv2 --output_jsonl /mnt/disk1/datasets/drag_data/train_json/pexels_tdv2_all.jsonl

python manual_select.py --root_dir /mnt/disk1/datasets/drag_data/selectframe/OpenVid-1M --output_jsonl /mnt/disk1/datasets/drag_data/train_json/OpenVid-1M_all.jsonl
```

## 视频点集配对标注
1.如何确定视频点集追踪点？
Q1:如何确定跟踪点的个数？
暂时固定为10个点。

Q2:如何确定跟踪点的位置？
先用grid采样，然后根据所有点的运动均值和方差确定阈值，进行第一波筛选；
然后对筛选出来的点进行radius内随机扩展，变为附近的多个点，继续进行第二波筛选；
第二波筛选兼顾点的位移（大于第二波内的位移均值），点的运动方向以及点之间的物理距离。

2.如何确定哪些帧抽出来进行训练？
Q1：在确定跟踪点的基础上，我们固定了stride=15,60，对于任意满足时间差为stride的两帧，我们只选择跟踪点总位移最大的一对。
也尝试过自适应抽帧，发现即使抽到40或80的stride，视觉差距跟60不大，即15,60，以及视频长度的stride足够进行较好的覆盖。

## 下载视频数据集

### OpenVid-1M
```bash
hf download nkp37/OpenVid-1M \
--repo-type dataset \
--include OpenVid_part0.zip \
--local-dir ./
```

一些视频数据集及其特点：
1.OpenVid-1M，数据量很大。但是关于镜头运动类别标注不算很准确，另外很多是自然风景类别的视频，不太符合需求，或者说符合需求的极少，筛选很麻烦。

```bash
wget -c --tries=0 --read-timeout=20 --waitretry=5 https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part0.zip
unzip -j OpenVid_part0.zip -d video_folder
```