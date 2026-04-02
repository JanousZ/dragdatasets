"""
使用 LLM 自动生成 Pexels 搜索关键词，扩展 target_distribution.json

用法:
    python generate_keywords.py --api_key YOUR_API_KEY --output target_distribution_gen.json

可选参数:
    --base_json   基础 JSON 文件，在其基础上扩展（默认不使用，从目标分布定义生成）
    --num_keywords 每个叶子节点生成的关键词数量（默认 15）
    --model       使用的模型（默认 claude-sonnet-4-20250514）
"""

import json
import argparse
import time
from anthropic import Anthropic

# --- 目标分布定义 ---
# 从 Readme.md 中提取的完整目标分布结构
TARGET_DISTRIBUTION_SPEC = {
    "human": {
        "orientation": "头部旋转（左右、上下）、身体朝向改变、头部朝向改变",
        "expression": "微笑（嘴角扬起）、皱眉（眉部变化）、张嘴过程、惊讶过程、哭泣、眨眼等",
        "pose": "四肢动作改变，如伸展、弯腰、踢腿、跳跃、舞蹈、瑜伽等",
        "fingers": "手指动作改变，如握拳、张开、比划手势、抓取等",
        "position": "人的整体位置发生明显移动，如行走、跑步、侧移等"
    },
    "animal": {
        "orientation": {
            "subjects": ["cat", "dog", "bird", "horse", "lion", "deer", "rabbit", "bear", "fox", "monkey"],
            "description": "动物左右转头、抬头、低头、身体朝向改变"
        },
        "face": {
            "subjects": ["cat", "dog", "lion", "bird", "monkey", "rabbit", "hamster"],
            "description": "嘴部张开关闭、耳朵变化、眼睛张开闭合变化"
        },
        "pose": {
            "subjects": ["cat", "dog", "bird", "horse", "butterfly", "fish", "elephant", "kangaroo", "snake", "spider"],
            "description": "局部肢体移动、翅膀扇动、尾巴移动等"
        },
        "position": {
            "subjects": ["cat", "dog", "bird", "horse", "rabbit", "deer", "fish"],
            "description": "动物整体位置发生明显移动"
        }
    },
    "plant": {
        "grow": {
            "subjects": ["seed", "vine", "mushroom", "bamboo", "moss"],
            "description": "植物生长过程"
        },
        "bloom": {
            "subjects": ["rose", "lily", "sunflower", "cherry blossom", "tulip", "orchid", "daisy", "lotus"],
            "description": "花朵绽放过程"
        },
        "swing": {
            "subjects": ["tree", "grass", "flower", "fern", "palm tree", "willow"],
            "description": "植物在风中摆动"
        }
    },
    "daily_object": {
        "translation": {
            "subjects": ["toy car", "drawer", "ball", "train", "sliding door", "conveyor belt"],
            "description": "物体整体或局部平移运动"
        },
        "rotation": {
            "subjects": ["spinning top", "fan", "wheel", "turntable product", "windmill", "gear"],
            "description": "物体整体或局部旋转"
        },
        "scaling": {
            "subjects": ["balloon", "umbrella", "bread dough", "bubble", "pupil", "accordion"],
            "description": "物体放大或缩小"
        },
        "deformation": {
            "subjects": ["sponge", "clay", "rubber band", "slime", "paper", "spring"],
            "description": "物体弯曲、拉伸、压缩、形变"
        },
        "openclose": {
            "subjects": ["book", "laptop", "box", "door", "jar", "drawer", "suitcase"],
            "description": "物体的开合动作"
        },
        "flip": {
            "subjects": ["coin", "pancake", "card", "page", "burger", "phone"],
            "description": "物体上下或左右翻转"
        }
    }
}

SYSTEM_PROMPT = """You are a search keyword expert for Pexels (a stock video platform).
Your job is to generate SHORT, HIGH-RECALL English search keywords that will return many results on Pexels.

Rules:
1. Each keyword must be 2-4 English words, NO MORE
2. Use common stock video terminology: "close up", "slow motion", "timelapse", "portrait", "macro"
3. Keep keywords SIMPLE - avoid compound/complex phrases
4. Vary the subject (man/woman/child, different breeds, etc.)
5. Include synonyms and alternative phrasings for the same action
6. Output ONLY a JSON array of strings, no explanation
7. Every keyword must be likely to return results on Pexels (think like a stock video uploader)
8. Avoid overly specific or niche terms that stock platforms won't have
"""


def generate_keywords_for_leaf(client, model, category, subcategory, description, subject=None, num_keywords=15):
    """为一个叶子节点生成关键词"""

    if subject:
        user_prompt = (
            f"Generate {num_keywords} Pexels search keywords for:\n"
            f"Category: {category} > {subcategory} > {subject}\n"
            f"Motion type: {description}\n"
            f"Subject: {subject}\n\n"
            f"Return a JSON array of {num_keywords} short search keywords."
        )
    else:
        user_prompt = (
            f"Generate {num_keywords} Pexels search keywords for:\n"
            f"Category: {category} > {subcategory}\n"
            f"Motion type: {description}\n\n"
            f"Return a JSON array of {num_keywords} short search keywords. "
            f"Vary the subjects (e.g., man/woman/child for humans)."
        )

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            text = response.content[0].text.strip()
            # 提取 JSON 数组
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end > start:
                keywords = json.loads(text[start:end])
                return keywords
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    return []


def build_distribution(client, model, num_keywords):
    """根据目标分布定义，逐个叶子节点生成关键词"""
    result = {}

    for category, subcategories in TARGET_DISTRIBUTION_SPEC.items():
        print(f"\n{'='*50}")
        print(f"Category: {category}")
        result[category] = {}

        for subcategory, spec in subcategories.items():
            result[category][subcategory] = {}

            if isinstance(spec, str):
                # human 类: spec 直接是描述字符串，无 subjects 细分
                print(f"  [{subcategory}] generating {num_keywords} keywords...")
                keywords = generate_keywords_for_leaf(
                    client, model, category, subcategory, spec,
                    num_keywords=num_keywords
                )
                # human 类直接用 list 而非 dict
                result[category][subcategory] = keywords
                print(f"    -> got {len(keywords)} keywords")
                time.sleep(0.5)

            elif isinstance(spec, dict) and "subjects" in spec:
                # 有 subjects 细分
                description = spec["description"]
                for subject in spec["subjects"]:
                    print(f"  [{subcategory}/{subject}] generating {num_keywords} keywords...")
                    keywords = generate_keywords_for_leaf(
                        client, model, category, subcategory, description,
                        subject=subject, num_keywords=num_keywords
                    )
                    safe_subject = subject.replace(" ", "_")
                    result[category][subcategory][safe_subject] = keywords
                    print(f"    -> got {len(keywords)} keywords")
                    time.sleep(0.5)

    return result


def merge_with_base(base, generated):
    """将生成的关键词合并到基础 JSON 中（去重）"""
    merged = json.loads(json.dumps(base))  # deep copy

    for cat, subcats in generated.items():
        if cat not in merged:
            merged[cat] = subcats
            continue
        for subcat, value in subcats.items():
            if subcat not in merged[cat]:
                merged[cat][subcat] = value
                continue
            if isinstance(value, list) and isinstance(merged[cat][subcat], list):
                # 合并 list 并去重
                existing = set(merged[cat][subcat])
                for kw in value:
                    if kw not in existing:
                        merged[cat][subcat].append(kw)
            elif isinstance(value, dict) and isinstance(merged[cat][subcat], dict):
                for subject, kws in value.items():
                    if subject not in merged[cat][subcat]:
                        merged[cat][subcat][subject] = kws
                    else:
                        existing = set(merged[cat][subcat][subject])
                        for kw in kws:
                            if kw not in existing:
                                merged[cat][subcat][subject].append(kw)
    return merged


def main():
    parser = argparse.ArgumentParser(description="使用 LLM 生成 Pexels 搜索关键词")
    parser.add_argument("--api_key", type=str, required=True, help="Anthropic API Key")
    parser.add_argument("--output", type=str, default="target_distribution_gen.json", help="输出文件路径")
    parser.add_argument("--base_json", type=str, default=None, help="基础 JSON 文件路径（可选，用于合并）")
    parser.add_argument("--num_keywords", type=int, default=15, help="每个叶子节点生成的关键词数量")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="使用的模型")
    args = parser.parse_args()

    client = Anthropic(api_key=args.api_key)

    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Keywords per node: {args.num_keywords}")
    print("=" * 50)

    # 生成关键词
    generated = build_distribution(client, args.model, args.num_keywords)

    # 如果有基础 JSON，合并
    if args.base_json:
        print(f"\nMerging with base: {args.base_json}")
        with open(args.base_json, 'r', encoding='utf-8') as f:
            base = json.load(f)
        final = merge_with_base(base, generated)
    else:
        final = generated

    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=4, ensure_ascii=False)

    # 统计
    total_keywords = 0
    def count_keywords(data):
        nonlocal total_keywords
        if isinstance(data, list):
            total_keywords += len(data)
        elif isinstance(data, dict):
            for v in data.values():
                count_keywords(v)
    count_keywords(final)

    print(f"\nDone! Saved to: {args.output}")
    print(f"Total keywords: {total_keywords}")


if __name__ == "__main__":
    main()
