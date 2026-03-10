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

视频筛选原则：
1.静态视角，结合csv文件筛选
2.主要变化发生在单一主体
3.视频变化能归类到操作类型的任意一类

分类原则：
1.对主体进行分类（人、动物、日常生活用品、风景/背景/地形）
2.对操作类型进行分类（平移、缩放、姿态、形变、旋转等）

你首先帮我写一段代码，读入csv文件，然后寻找数据集根目录下的所有视频，获取视频名称，在csv中迅速找到对应的caption，提取caption进行分类和筛选的进一步判断