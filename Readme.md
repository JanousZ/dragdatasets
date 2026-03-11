## 下载视频数据集
```bash
hf download nkp37/OpenVid-1M \
--repo-type dataset \
--include OpenVid_part0.zip \
--local-dir ./
```

```bash
wget -c --tries=0 --read-timeout=20 --waitretry=5 https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part0.zip
unzip -j OpenVid_part0.zip -d video_folder
```

## 视频点集配对标注
```bash
cd co-tracker
python demo.py --offline
```
未解决问题：
1.如何确定视频点集追踪点？
2.如何确定哪些帧抽出来进行训练？

## 视频筛选原则

### 方案一
OpenVid-1M的数据集还是太杂太乱，需要进行处理。
1.静态视角，结合csv文件筛选
初筛后仍有一部分实际上不是完全的静态视角。
后续可尝试方法：使用qwen2.5-vl-7b-cam-motion进行进一步筛选。

2.主要变化发生在单一主体
3.视频变化能归类到操作类型的任意一类
分类原则：
1.对主体进行分类（人、动物、日常生活用品、风景/背景/地形、其他）
2.对操作类型进行分类（平移、缩放、姿态、形变、旋转、其他）
暂时尝试方法：使用qwen2.5-lv-7b-instruct模型进行筛选。

### 方案二
分门别类寻找对应的数据集，例如包含人物姿态变化、表情变化、面向变化的视频数据集，动物姿态变化的视频数据集、物品平移、缩放、形变、旋转的视频数据集。