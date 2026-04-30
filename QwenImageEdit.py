"""
python QwenImageEdit.py --gpu_ids 0,2,3
"""
import argparse
import json
import os
import sys
import time


def _parse_gpu_ids(s):
    ids = [int(x) for x in s.split(",") if x.strip() != ""]
    if not ids:
        raise argparse.ArgumentTypeError("--gpu_ids must contain at least one id")
    if len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError(f"--gpu_ids has duplicates: {ids}")
    return ids


def _early_parse_gpu_ids(argv):
    """Pluck --gpu_ids out of argv before torch is imported, so we can pin
    CUDA_VISIBLE_DEVICES first. bitsandbytes' 4-bit caching-allocator warmup
    fires on the *current* CUDA device during from_pretrained, so if we don't
    hide unselected GPUs we'll OOM on whichever GPU is busy (typically GPU 0).
    """
    for i, arg in enumerate(argv):
        if arg == "--gpu_ids" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--gpu_ids="):
            return arg.split("=", 1)[1]
    return None


_user_gpu_ids_str = _early_parse_gpu_ids(sys.argv)
if _user_gpu_ids_str is not None:
    # validate now so we fail before importing torch on a bad value
    _user_gpu_ids = _parse_gpu_ids(_user_gpu_ids_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in _user_gpu_ids)
else:
    _user_gpu_ids = None

import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def color_fix(source_img, edited_img):
    src = np.array(source_img).astype(np.float32)
    edt = np.array(edited_img).astype(np.float32)
    for i in range(3):
        edt[:, :, i] = edt[:, :, i] - np.mean(edt[:, :, i]) + np.mean(src[:, :, i])
    return Image.fromarray(np.clip(edt, 0, 255).astype(np.uint8))
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel

model_id = "/mnt/disk1/models/Qwen-Image-Edit-2511"

def build_pipe(gpu_id):
    # transformer量化，注意必须skip这一层
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod"],  # 关键！
        ),
        torch_dtype=torch.bfloat16,
    ).to("cpu")

    # text_encoder单独量化
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        subfolder="text_encoder",
        quantization_config=TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
    ).to("cpu")

    pipe = QwenImageEditPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    return pipe

# If the user did not pass --gpu_ids, default to every GPU visible to torch
# (CUDA_VISIBLE_DEVICES was not touched, so these are the real ids).
if _user_gpu_ids is None:
    _user_gpu_ids = list(range(max(1, torch.cuda.device_count())))

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", default="/home/yanzhang/dragdatasets/instructions.jsonl")
ap.add_argument("--src_root", default="/home/yanzhang/dragdatasets/pexels_tdv3")
ap.add_argument("--out_root", default="/home/yanzhang/dragdatasets/pexels_tdv3_edited")
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--gpu_ids", type=_parse_gpu_ids, default=_user_gpu_ids,
                help="Comma-separated CUDA device ids to run on, e.g. '0,2,3'. "
                     "One pipeline is built per id and records are sharded across them.")
args = ap.parse_args()

# Two views of the GPU list:
#   gpu_ids       - what the user wrote, used in print/log lines
#   torch_indices - 0..N-1 indices torch sees AFTER CUDA_VISIBLE_DEVICES remap
gpu_ids = args.gpu_ids
torch_indices = list(range(len(gpu_ids)))
print(f"[init] building pipes on GPUs {gpu_ids}")
pipes = [build_pipe(t) for t in torch_indices]

# 推荐的非漂移分辨率桶（社区验证：方形会色偏/糊掉，issue #243）
# tgt 用 ~1MP 的非方形桶；src 用对应的一半分辨率，方向跟随原图
LANDSCAPE_TGT = (1024, 1024)
PORTRAIT_TGT = (1024, 1024)

def pick_bucket(w, h):
    tgt_w, tgt_h = LANDSCAPE_TGT if w >= h else PORTRAIT_TGT
    return (tgt_w // 2, tgt_h // 2), (tgt_w, tgt_h)

def fit_to_bucket(im, bucket_w, bucket_h):
    # 先按目标长宽比 center-crop，再 resize 到桶尺寸；不拉伸变形
    w, h = im.size
    tgt_ratio = bucket_w / bucket_h
    src_ratio = w / h
    if src_ratio > tgt_ratio:
        new_w = int(round(h * tgt_ratio))
        x0 = (w - new_w) // 2
        im = im.crop((x0, 0, x0 + new_w, h))
    elif src_ratio < tgt_ratio:
        new_h = int(round(w / tgt_ratio))
        y0 = (h - new_h) // 2
        im = im.crop((0, y0, w, y0 + new_h))
    return im.resize((bucket_w, bucket_h), Image.LANCZOS)

with open(args.jsonl, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

def worker(slot, my_records):
    gpu_id = gpu_ids[slot]
    pipe = pipes[slot]
    for rec in my_records:
        if rec.get("error") or not rec.get("suitable"):
            continue
        rel = rec["path"]
        src_path = os.path.join(args.src_root, rel)
        name, _ = os.path.splitext(os.path.basename(rel))
        out_dir = os.path.join(args.out_root, os.path.dirname(rel))
        os.makedirs(out_dir, exist_ok=True)

        raw = Image.open(src_path).convert("RGB")
        (src_w, src_h), (tgt_w, tgt_h) = pick_bucket(*raw.size)
        src_img = fit_to_bucket(raw, src_w, src_h)
        src_save = os.path.join(out_dir, f"{name}__src.png")
        if not os.path.exists(src_save):
            src_img.save(src_save)

        for i, ins in enumerate(rec["instructions"]):
            tgt_save = os.path.join(out_dir, f"{name}__tgt{i}.png")
            if os.path.exists(tgt_save):
                continue
            inputs = {
                "image": [src_img],
                "prompt": ins["instruction"],
                # "prompt": "make the pants shorter",
                "generator": torch.manual_seed(i + int(time.time())),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": args.steps,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
                "height": tgt_h ,
                "width": tgt_w ,
            }
            with torch.inference_mode():
                output = pipe(**inputs)
            edited = output.images[0]
            if edited.size != src_img.size:
                ref = src_img.resize(edited.size)
            else:
                ref = src_img
            color_fix(ref, edited).save(tgt_save)
            #edited.save(tgt_save)
            print(f"[ok gpu{gpu_id}] {rel}  #{i}  -> {tgt_save}")

n_workers = len(gpu_ids)
chunks = [records[i::n_workers] for i in range(n_workers)]
with ThreadPoolExecutor(max_workers=n_workers) as ex:
    futures = [ex.submit(worker, i, chunks[i]) for i in range(n_workers)]
    for f in futures:
        f.result()
