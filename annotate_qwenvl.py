"""
Run local Qwen2.5-VL on every image under --root and emit a JSONL of
drag-style edit instructions, using prompts.py.

Quick test on 5 images:
    python annotate_qwenvl.py \
        --limit 10 \
        --out instructions.test.jsonl

Full run:
    python annotate_qwenvl.py \
        --out instructions.jsonl --resume
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from prompts import assign_operations, build_messages, infer_category, validate_output


# TODO(user): set this to the local Qwen2.5-VL checkpoint directory once downloaded.
DEFAULT_MODEL_PATH = "/mnt/disk1/models/Qwen/Qwen2.5-VL-7B-Instruct"


def parse_json_loose(text):
    """Try strict json.loads, then fall back to extracting the first {...} block."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def run_once(model, processor, image_path, category, target_operations,
             max_new_tokens, do_sample, temperature):
    messages = build_messages(image_path, category, target_operations=target_operations)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if do_sample:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return output_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/yanzhang/dragdatasets/pexels_tdv3")
    ap.add_argument("--out", default="/home/yanzhang/dragdatasets/instructions.jsonl")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                    help="Local Qwen2.5-VL-Instruct directory.")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all images; otherwise process the first N images (sorted).")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="auto",
                    help="auto / bfloat16 / float16")
    ap.add_argument("--resume", action="store_true",
                    help="Skip images whose path is already present in --out.")
    ap.add_argument("--retry_temperature", type=float, default=0.2,
                    help="If greedy output fails JSON validation, retry once with this temperature.")
    args = ap.parse_args()

    print(f"[load] {args.model_path}")
    dtype = {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"[load] done. device={model.device}")

    image_paths = sorted(Path(args.root).rglob("*.jpg"))
    if args.limit:
        image_paths = image_paths[: args.limit]
    print(f"[scan] {len(image_paths)} images under {args.root}")

    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("error") is None:
                        done.add(rec["path"])
                except Exception:
                    continue
        print(f"[resume] {len(done)} already done, skipping them")

    out_f = open(args.out, "a" if args.resume else "w", encoding="utf-8")
    n_ok = n_reject = n_fail = 0
    cat_counter = {}  # per-category counter for round-robin operation assignment
    t0 = time.time()

    try:
        for i, p in enumerate(image_paths):
            rel = str(p.relative_to(args.root))
            if rel in done:
                continue

            try:
                category = infer_category(p, args.root)
            except Exception as e:
                print(f"  [skip] {rel}: cannot infer category ({e})")
                continue

            idx = cat_counter.get(category, 0)
            cat_counter[category] = idx + 1
            target_ops = assign_operations(category, idx)

            # First pass: greedy.
            raw = run_once(model, processor, str(p), category, target_ops,
                           args.max_new_tokens, do_sample=False, temperature=0.3)
            parsed = parse_json_loose(raw)

            if parsed is None or not validate_output(parsed, category):
                # Retry once with temperature.
                raw2 = run_once(model, processor, str(p), category, target_ops,
                                args.max_new_tokens, do_sample=True,
                                temperature=args.retry_temperature)
                parsed2 = parse_json_loose(raw2)
                if parsed2 is not None and validate_output(parsed2, category):
                    parsed = parsed2
                    raw = raw2
                else:
                    record = {
                        "path": rel, "category": category,
                        "error": "json_parse_or_validate_failed",
                        "raw": raw, "raw_retry": raw2,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_fail += 1
                    print(f"[{i+1}/{len(image_paths)}] {rel}  FAIL")
                    continue

            record = {"path": rel, "category": category, **parsed}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            if parsed["suitable"]:
                n_ok += 1
                tag = "OK"
                pairs = [
                    f"{ins['operation']}/{ins['direction']}" if ins.get("direction")
                    else ins["operation"]
                    for ins in parsed["instructions"]
                ]
                detail = f"pairs={pairs}"
            else:
                n_reject += 1
                tag = "REJECT"
                detail = f"ops={target_ops}"
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(image_paths)}] {rel}  {tag}  {detail}  ({elapsed:.1f}s)")
    finally:
        out_f.close()

    total = n_ok + n_reject + n_fail
    print(f"\n[done] processed={total}  OK={n_ok}  REJECT={n_reject}  FAIL={n_fail}")
    print(f"[done] output: {args.out}")


if __name__ == "__main__":
    main()
