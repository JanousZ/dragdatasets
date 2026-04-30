"""
Prompt templates and per-category configuration for Qwen-VL drag-edit
instruction annotation.

Public API:
- infer_category(path, root) -> "clothes/skirt"
- assign_operations(category, idx) -> list[str]  (length == N_PER_PARENT[parent])
- build_messages(image_path, category, target_operations=None) -> list[dict]
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
        "taper_leg", "flare_leg",
    ],
    "clothes/shirt": [
        "lengthen", "shorten",
        "flare_sleeve", "taper_sleeve",
    ],
    "design_furniture": [
        "curve_part", "flare_edge_outward",
        "taller", "wider", "shorter", "narrower",
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

# Number of distinct edit instructions to produce per image.
N_PER_PARENT = {
    "clothes": 3,
    "design_furniture": 3,
    "landscape": 3,
    "text_in_environment": 3,
}


# ---------------- Direction taxonomies ----------------
# Per-operation list of allowed `direction` values.
#
# An EMPTY list means the operation is non-directional: the JSON instruction
# for it MUST omit the `direction` field entirely (the direction is either
# meaningless, e.g. a uniform scale / horizontal flip, or already encoded
# in the op name, e.g. shift_boundary_down).
#
# Vocabulary used for directional ops:
#   "left" / "right" / "up" / "down" / "top" / "bottom"   - single side
#   "both_sides"                                          - symmetric L+R
#   "both_ends"                                           - symmetric T+B
#   "all_sides"                                           - all four sides
#   "clockwise" / "counterclockwise"                      - rotation only

OP_DIRECTIONS = {
    # clothes
    "lengthen": ["down", "up"],
    "shorten": ["up", "down"],
    "flare_hem_outward": ["left", "right", "both_sides"],
    "taper_hem_inward": ["left", "right", "both_sides"],
    "taper_leg": ["left", "right", "both_sides"],
    "flare_leg": ["left", "right", "both_sides"],
    "flare_sleeve": ["left", "right", "both_sides"],
    "taper_sleeve": ["left", "right", "both_sides"],

    # furniture
    "curve_part": ["left", "right", "top", "bottom"],
    "flare_edge_outward": ["left", "right", "top", "bottom"],
    "taller":   ["up", "down", "both_ends"],
    "wider":    ["left", "right", "both_sides"],
    "shorter":  ["top", "bottom", "both_ends"],
    "narrower": ["left", "right", "both_sides"],

    # landscape (shift_boundary_* have the direction baked into the op name)
    "shift_boundary_down": [],
    "shift_boundary_up": [],
    "boundary_spread":   ["left", "right", "top", "bottom", "all_sides"],
    "boundary_contract": ["left", "right", "top", "bottom", "all_sides"],
    "curve_boundary":   ["left", "right", "up", "down"],
    "scallop_boundary": ["left", "right", "up", "down"],

    # text in environment
    "translate": ["left", "right", "up", "down"],
    "rotate": ["clockwise", "counterclockwise"],
    "scale_up":   ["left", "right", "up", "down"],
    "scale_down": ["left", "right", "up", "down"],
    "stretch_horizontal": ["left", "right", "both_sides"],
    "squeeze_vertical":   ["top", "bottom", "both_ends"],
    "tilt": ["left", "right"],
    "curve_baseline": ["up", "down"],
    "bend_panel": ["left", "right"],
    "flip_horizontal": [],
}

# Sanity: every op in any taxonomy must have an entry in OP_DIRECTIONS.
for _cat, _ops in TAXONOMY.items():
    for _op in _ops:
        assert _op in OP_DIRECTIONS, f"{_op!r} (from {_cat}) missing in OP_DIRECTIONS"


# ---------------- Few-shot examples ----------------
# Each entry: (hypothetical-image description, expected JSON dict).
# Directional ops include a `direction` field; non-directional ops omit it.

_FEW_SHOT_SKIRT = [
    (
        "A woman wearing a knee-length pleated red skirt against a plain wall, full body visible.",
        {
            "subject": "pleated red skirt",
            "suitable": True,
            "reject_reason": "",
            "instructions": [
                {"operation": "lengthen", "direction": "down",
                 "instruction": "lengthen the skirt downward"},
                {"operation": "flare_hem_outward", "direction": "right",
                 "instruction": "flare the hem outward to the right"},
                {"operation": "taper_hem_inward", "direction": "both_sides",
                 "instruction": "taper the hem inward on both sides"},
            ],
        },
    ),
    (
        "A flat-lay photo with three different skirts side by side on a wooden floor.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "Multiple competing skirts, no single subject to edit.",
            "instructions": [],
        },
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
                {"operation": "taper_leg", "direction": "both_sides",
                 "instruction": "taper both legs inward"},
                {"operation": "lengthen", "direction": "down",
                 "instruction": "lengthen the pants downward"},
                {"operation": "flare_leg", "direction": "right",
                 "instruction": "flare the right leg outward"},
            ],
        },
    ),
    (
        "Close-up of a leather belt buckle; pants are barely visible at the bottom of the frame.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "The pants are mostly out of frame; only the belt is the prominent subject.",
            "instructions": [],
        },
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
                {"operation": "flare_sleeve", "direction": "both_sides",
                 "instruction": "flare both sleeves outward"},
                {"operation": "lengthen", "direction": "down",
                 "instruction": "lengthen the shirt downward"},
                {"operation": "taper_sleeve", "direction": "left",
                 "instruction": "taper the left sleeve inward"},
            ],
        },
    ),
    (
        "A pile of folded shirts stacked on a shelf; no individual shirt is fully visible.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "Many partially visible folded shirts; no single shirt subject to edit.",
            "instructions": [],
        },
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
                {"operation": "wider", "direction": "right",
                 "instruction": "make the sofa wider toward the right"},
                {"operation": "taller", "direction": "up",
                 "instruction": "make the sofa taller upward"},
                {"operation": "flare_edge_outward", "direction": "top",
                 "instruction": "flare the top edge outward"},
            ],
        },
    ),
    (
        "A wide living room with a sofa, two armchairs, a coffee table, a rug and several decorations.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "Multiple competing furniture pieces; no single dominant subject for an isolated edit.",
            "instructions": [],
        },
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
                {"operation": "shift_boundary_down",
                 "instruction": "move the snow line down"},
                {"operation": "curve_boundary", "direction": "up",
                 "instruction": "curve the snow line upward"},
                {"operation": "scallop_boundary", "direction": "down",
                 "instruction": "scallop the snow line downward"},
            ],
        },
    ),
    (
        "A panoramic green forest with no visible boundary, treeline, snow line, cliff, ridge or shoreline.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "No clear geographic boundary visible to deform.",
            "instructions": [],
        },
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
                {"operation": "rotate", "direction": "clockwise",
                 "instruction": "rotate the sign clockwise"},
                {"operation": "translate", "direction": "right",
                 "instruction": "move the sign to the right"},
                {"operation": "flip_horizontal",
                 "instruction": "flip the sign horizontally"},
            ],
        },
    ),
    (
        "A busy commercial street with many overlapping shop signs, pedestrians, cars and reflections.",
        {
            "subject": "",
            "suitable": False,
            "reject_reason": "Many overlapping signs and clutter; no single isolated text subject.",
            "instructions": [],
        },
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
- Each instruction picks exactly ONE operation from the allowed list.
- Operations are split into two kinds:
  * DIRECTIONAL ops have a non-empty allowed-direction set. For these, the
    JSON MUST include a `direction` field whose value is one of the allowed
    directions, AND the natural-language `instruction` MUST mention that
    direction (e.g. "make the sofa wider toward the right",
    "lengthen the skirt downward", "rotate the sign clockwise").
  * NON-DIRECTIONAL ops have an empty allowed-direction set (e.g. horizontal
    flip, or ops whose direction is already in the name like
    shift_boundary_down). For these, the JSON MUST OMIT the `direction`
    field entirely, and the instruction text describes the edit without
    inventing a side ("move the snow line down", "flip the sign horizontally").
- The instruction MUST be a short imperative sentence, 3 to 12 words, in
  plain everyday English. No numbers, percentages, multipliers, degrees,
  pixels, or any numeric magnitude. No "by 30%", "1.3x", "20 degrees".
- Do NOT chain clauses with "while", "and", "but". One short imperative only.
- Geometry-only edits. Do not change color, texture, identity, lighting or background.
- Produce N distinct instructions per image (see "Number of instructions").
  Distinct means the (operation, direction) pairs must all differ; for a
  non-directional op the pair counts as (operation, "").
- Each instruction must be physically plausible for THIS image (e.g. don't
  say "lengthen down" if the hem is already at the frame bottom, don't say
  "wider to the right" if the subject already touches the right edge). If
  you cannot produce N plausible distinct instructions, set suitable=false.
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
      "direction": "<one of allowed directions; OMIT this field for non-directional ops>",
      "instruction": "<short imperative, 3-12 words, mentions the direction when present>"
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


def _format_op_directions(ops):
    """Render the allowed-direction map for the given operations as a JSON block.
    Empty list means non-directional (omit `direction` in the output)."""
    return json.dumps({op: OP_DIRECTIONS[op] for op in ops}, ensure_ascii=False, indent=2)


def assign_operations(category, idx):
    """Deterministic round-robin assignment of operations within a category.
    idx is the per-category zero-based image index. Returns a list of length n.

    For thin taxonomies (len(taxonomy) < n) the same operation may repeat,
    in which case the model is expected to use different `direction` values
    to keep the instructions distinct (only meaningful for directional ops).
    """
    taxonomy, _, n = resolve_category(category)
    L = len(taxonomy)
    return [taxonomy[(idx + (k * L) // n) % L] for k in range(n)]


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
            f"# Allowed direction values per required operation\n"
            f"# (empty list = non-directional: OMIT the `direction` field for that instruction)\n"
            f"{_format_op_directions(target_operations)}\n\n"
            f"If a required operation is not physically plausible for this image, "
            f"or no plausible direction exists for it, set suitable=false and "
            f"leave instructions empty.\n\n"
            f"When the same directional operation appears more than once in the "
            f"required list, the `direction` values for those instructions MUST "
            f"differ so the resulting instructions are distinct. A non-directional "
            f"operation must not appear more than once.\n\n"
            f"The few-shot below shows the JSON format and short-imperative style; "
            f"for THIS image use the required operation(s) above, not the few-shot's.\n\n"
        )
    else:
        ops_block = (
            f"# Allowed operations (pick ONE per instruction; do not invent new ones)\n"
            f"{json.dumps(taxonomy)}\n\n"
            f"# Allowed direction values per operation\n"
            f"# (empty list = non-directional: OMIT the `direction` field for that instruction)\n"
            f"{_format_op_directions(taxonomy)}\n\n"
            f"# Number of instructions to produce\nN = {n}  "
            f"(produce exactly {n} distinct instructions when suitable=true)\n\n"
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

MAX_INSTRUCTION_WORDS = 12


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
        seen_pairs = set()
        for ins in d["instructions"]:
            if not isinstance(ins, dict):
                return False
            for k in ("operation", "instruction"):
                if k not in ins or not isinstance(ins[k], str) or not ins[k].strip():
                    return False
            if ins["operation"] not in taxonomy:
                return False
            allowed_dirs = OP_DIRECTIONS.get(ins["operation"], [])
            if allowed_dirs:
                direction = ins.get("direction")
                if not isinstance(direction, str) or direction not in allowed_dirs:
                    return False
            else:
                # non-directional op: direction field must be absent or empty
                if ins.get("direction"):
                    return False
                direction = ""
            pair = (ins["operation"], direction)
            if pair in seen_pairs:
                return False
            seen_pairs.add(pair)
            if len(ins["instruction"].split()) > MAX_INSTRUCTION_WORDS:
                return False
    else:
        if len(d["instructions"]) != 0:
            return False
    return True
