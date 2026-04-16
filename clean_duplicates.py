"""
清理 JSONL 中的重复记录：同一个 pair 只保留最后一条（后写入的覆盖先写入的）。
用法: python clean_duplicates.py --jsonl path/to/output.jsonl
"""
import json
import argparse


def clean(jsonl_path):
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # 同一个 pair 保留最后一条
    seen = {}
    for idx, r in enumerate(records):
        pair = r.get("pair", [])
        if len(pair) == 2:
            key = tuple(pair)
        else:
            key = (idx,)  # 没有 pair 字段的记录原样保留
        seen[key] = r

    deduped = list(seen.values())
    removed = len(records) - len(deduped)

    if removed == 0:
        print(f"无重复记录，共 {len(records)} 条。")
        return

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"清理完成: {len(records)} -> {len(deduped)} 条 (移除 {removed} 条重复)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理 JSONL 中同一 pair 的重复记录，保留最后一条")
    parser.add_argument("--jsonl", required=True, help="要清理的 JSONL 文件路径")
    args = parser.parse_args()
    clean(args.jsonl)
