"""
Prompt templates and per-category configuration for Qwen-VL drag-edit
instruction annotation.

Public API:
- infer_category(path, root) -> "clothes/skirt"
- build_messages(image_path, category) -> list[dict]  (Qwen-VL chat format)
- validate_output(d, category) -> bool
"""

import json
from pathlib import Path


# ---------------- Operation taxonomies ----------------
# Keyed by full "parent/child", or just "parent" when shared across all children.

TAXONOMY = {
    "clothes/skirt": [
        "lengthen", "shorten",
        "flare_hem_outward", "taper_hem_inward",
    ],
    "clothes/pants": [
        "lengthen", "shorten",
    ],
    "clothes/shirt": [
        "lengthen", "shorten",
    ],
    "design_furniture": [
        "stretch_horizontal", "compress_vertical",
        "scale_up_uniform", "scale_down_uniform",
        "curve_part", "flare_edge_outward", "bend_edge",
        "boundary_spread", "boundary_contract",
    ],
    "landscape": [
        "shift_boundary_down", "shift_boundary_up",
        "boundary_spread", "boundary_contract",
        "curve_boundary", "scallop_boundary",
    ],
    "text_in_environment": [
        "translate", "rotate",
        "scale_up", "scale_down",
        "stretch_horizontal", "squeeze_vertical",
        "tilt", "curve_baseline",
        "bend_panel", "flip_horizontal",
    ],
}

N_PER_PARENT = {
    "clothes": 1,
    "design_furniture": 1,
    "landscape": 1,
    "text_in_environment": 1,
}


# ---------------- Few-shot examples ----------------
# Each entry: (hypothetical-image description, expected JSON dict).
# We keep them in Python so json.dumps emits valid JSON at prompt-build time.

_FEW_SHOT_SKIRT = [
    (
        "A woman wearing a knee-length pleated red skirt against a plain wall, full body visible.",
        {
            "subject": "pleated red skirt",
            "suitable": True,
            "reject_reason": "",
            "instructions": [
                {"operation": "lengthen", "instruction": "make the skirt longer"}
            ]
        }
    ),
    (
        "A flat-lay photo with three different skirts side by side on a wooden floor.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "Multiple competing skirts, no single subject to edit.",
            "instructions": []
        }
    ),
]

_FEW_SHOT_PANTS = [
    (
        "A person standing against a brick wall wearing straight-cut blue jeans, full lower body visible.",
        {
            "subject": "blue straight-cut jeans",
            "suitable": True,
            "reject_reason": "",
            "instructions": [
                {"operation": "taper_leg", "instruction": "make the pants narrower"}
            ]
        }
    ),
    (
        "Close-up of a leather belt buckle; pants are barely visible at the bottom of the frame.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "The pants are mostly out of frame; only the belt is the prominent subject.",
            "instructions": []
        }
    ),
]

_FEW_SHOT_SHIRT = [
    (
        "A man wearing a white short-sleeve polo shirt, photographed from the front against a neutral studio background.",
        {
            "subject": "white short-sleeve polo shirt",
            "suitable": True,
            "reject_reason": "",
            "instructions": [
                {"operation": "flare_sleeve", "instruction": "make the sleeves wider"}
            ]
        }
    ),
    (
        "A pile of folded shirts stacked on a shelf; no individual shirt is fully visible.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "Many partially visible folded shirts; no single shirt subject to edit.",
            "instructions": []
        }
    ),
]

_FEW_SHOT_FURNITURE = [
    (
        "A single curved velvet sofa centered in a minimalist living-room shot, full sofa visible against a plain wall.",
        {
            "subject": "curved velvet sofa",
            "suitable": True,
            "reject_reason": "",
            "instructions": [
                {"operation": "stretch_horizontal", "instruction": "make the sofa wider"}
            ]
        }
    ),
    (
        "A wide living room with a sofa, two armchairs, a coffee table, a rug and several decorations.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "Multiple competing furniture pieces; no single dominant subject for an isolated edit.",
            "instructions": []
        }
    ),
]

_FEW_SHOT_LANDSCAPE = [
    (
        "A mountain ridge photographed from a distance, with a clearly visible snow line separating the white snowy upper area and the bare rocky lower area.",
        {
            "subject": "mountain ridge with a visible snow line",
            "suitable": True,
            "reject_reason": "",
            "instructions": [
                {"operation": "shift_boundary_down", "instruction": "move the snow line down"}
            ]
        }
    ),
    (
        "A panoramic green forest with no visible boundary, treeline, snow line, cliff, ridge or shoreline.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "No clear geographic boundary visible to deform.",
            "instructions": []
        }
    ),
]

_FEW_SHOT_TEXT = [
    (
        "A single round red octagonal STOP traffic sign on a metal pole, centered against a clear blue sky.",
        {
            "subject": "red octagonal STOP traffic sign",
            "suitable": True,
            "reject_reason": "",
            "instructions": [
                {"operation": "rotate", "instruction": "rotate the stop sign"},
                {"operation": "scale_up", "instruction": "make the sign bigger"}
            ]
        }
    ),
    (
        "A busy commercial street with many overlapping shop signs, pedestrians, cars and reflections.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "Many overlapping signs and clutter; no single isolated text subject.",
            "instructions": []
        }
    ),
]

FEW_SHOT = {
    "clothes/skirt": _FEW_SHOT_SKIRT,
    "clothes/pants": _FEW_SHOT_PANTS,
    "clothes/shirt": _FEW_SHOT_SHIRT,
    "design_furniture": _FEW_SHOT_FURNITURE,
    "landscape": _FEW_SHOT_LANDSCAPE,
    "text_in_environment": _FEW_SHOT_TEXT,
}


# ---------------- Resolution helpers ----------------

def resolve_category(category):
    """Return (taxonomy, few_shot, n) for a full category like 'design_furniture/organic_series'."""
    parent = category.split("/", 1)[0]
    if parent not in N_PER_PARENT:
        raise KeyError(f"Unknown parent category: {parent}")
    tax = TAXONOMY.get(category) or TAXONOMY.get(parent)
    if tax is None:
        raise KeyError(f"No taxonomy for category: {category}")
    fs = FEW_SHOT.get(category) or FEW_SHOT.get(parent)
    if fs is None:
        raise KeyError(f"No few-shot for category: {category}")
    return tax, fs, N_PER_PARENT[parent]


def infer_category(image_path, root):
    """Map /<root>/clothes/skirt/12345.jpg -> 'clothes/skirt'."""
    p = Path(image_path).resolve()
    r = Path(root).resolve()
    rel = p.relative_to(r)
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(f"Path too shallow to infer category: {rel}")
    return f"{parts[0]}/{parts[1]}"


# ---------------- Prompt construction ----------------

SYSTEM_PROMPT = """You are an annotator generating drag-style image-edit instructions.
Each instruction will be sent to an image editor that performs ONE atomic
geometric edit on a SINGLE subject. The edit will later be matched with
manually labeled drag point pairs (src_pts -> tgt_pts).

# Hard rules
- Operate on exactly ONE subject (the most prominent object in the image).
- Choose exactly ONE operation per instruction from the allowed list for this category.
- The instruction MUST be a very short imperative sentence, 3 to 15 words,
  in plain everyday English. Examples: "make the pants longer",
  "make the skirt shorter", "rotate the sign", "make the sofa wider".
- Do NOT use numbers, percentages, multipliers, degrees, pixels, or any
  numeric magnitude. No "by 30%", "1.3x", "20 degrees".
- Do NOT chain clauses with "while", "and", "but". One short imperative only.
- Geometry-only edits. Do not change color, texture, identity, lighting or background.
- If the image has no clear single subject, the subject is severely occluded,
  multiple subjects compete, or no allowed operation reasonably fits,
  return suitable=false with a brief reject_reason and an empty instructions list.

# Output: STRICT JSON only. No prose, no markdown fences, no comments. Schema:
{
  "subject": "<short noun phrase>",
  "suitable": true | false,
  "reject_reason": "<empty string if suitable>",
  "instructions": [
    {
      "operation": "<one of allowed ops>",
      "instruction": "<short imperative, 3-8 words, no numbers>"
    }
  ]
}
"""


def _format_few_shot(few_shot):
    blocks = []
    for i, (desc, out) in enumerate(few_shot, 1):
        blocks.append(
            f'Example {i} - hypothetical image: "{desc}"\n'
            f'Expected output:\n'
            f'{json.dumps(out, ensure_ascii=False, indent=2)}'
        )
    return "\n\n".join(blocks)


def assign_operations(category, idx):
    """Deterministic round-robin assignment of operations within a category.
    idx is the per-category zero-based image index. Returns a list of length n.
    """
    taxonomy, _, n = resolve_category(category)
    L = len(taxonomy)
    if n == 1:
        return [taxonomy[idx % L]]
    # n == 2: pick two ops far apart in the taxonomy for variety
    return [taxonomy[idx % L], taxonomy[(idx + L // 2) % L]]


def build_messages(image_path, category, target_operations=None):
    taxonomy, few_shot, n = resolve_category(category)
    if target_operations is not None:
        assert len(target_operations) == n, \
            f"target_operations must have length {n} for {category}"
        for op in target_operations:
            assert op in taxonomy, f"{op!r} not in taxonomy for {category}"
        ops_block = (
            f"# Required operation(s) for this image (use exactly these, in order; do NOT substitute)\n"
            f"{json.dumps(target_operations)}\n\n"
            f"If a required operation is not physically plausible for this image, "
            f"set suitable=false and leave instructions empty.\n\n"
            f"The few-shot below shows the JSON format and short-imperative style; "
            f"for THIS image use the required operation(s) above, not the few-shot's.\n\n"
        )
    else:
        ops_block = (
            f"# Allowed operations (pick ONE per instruction; do not invent new ones)\n"
            f"{json.dumps(taxonomy)}\n\n"
            f"# Number of instructions to produce\nN = {n}  "
            f"(produce exactly {n} instruction(s) when suitable=true)\n\n"
        )
    user_text = (
        f"# Category\n{category}\n\n"
        f"{ops_block}"
        f"# Few-shot examples (text descriptions of hypothetical images, with expected JSON)\n"
        f"{_format_few_shot(few_shot)}\n\n"
        f"# Now produce the JSON for the actual image attached to this message. "
        f"Output JSON only, no extra text."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_text},
        ]},
    ]


# ---------------- Output validation ----------------

def validate_output(d, category):
    try:
        taxonomy, _, n = resolve_category(category)
    except Exception:
        return False
    if not isinstance(d, dict):
        return False
    for k in ("subject", "suitable", "reject_reason", "instructions"):
        if k not in d:
            return False
    if not isinstance(d["suitable"], bool):
        return False
    if not isinstance(d["instructions"], list):
        return False
    if d["suitable"]:
        if len(d["instructions"]) != n:
            return False
        for ins in d["instructions"]:
            if not isinstance(ins, dict):
                return False
            for k in ("operation", "instruction"):
                if k not in ins or not isinstance(ins[k], str) or not ins[k].strip():
                    return False
            if ins["operation"] not in taxonomy:
                return False
            # length cap: short imperative only
            if len(ins["instruction"].split()) > 8:
                return False
    else:
        if len(d["instructions"]) != 0:
            return False
    return True
