#!/usr/bin/env python3
"""
YouTube downloader for the drag-edit dataset.

Pipeline-aligned constraints:
  - Source short side ≥ 1080 (so post-crop 1024 training res is always satisfiable)
  - Per-clip duration window [min_dur, max_dur]; final ≤10s segments produced
    by the downstream crop_and_split.py
  - Static-camera bias via keyword engineering + title negative filter + optional
    pyscenedetect scene-cut check
  - Cross-keyword/run dedup via yt-dlp download archive
  - JSON keyword tree (mirrors pexels.py convention) → recursive dir layout

python yt-dlp.py \
  --save_dir /mnt/disk1/datasets/drag_data/rawvideo/youtube_v1 \
  --scene_filter \
  --remux \
  --max_workers 12
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, MaxDownloadsReached

# --- Title tokens that almost always mean "violates Q1/Q6 (cuts, handheld, OOD)" ---
BAD_TITLE_TOKENS = [
    "compilation", "reaction", "react ", "vlog", "gameplay", "gaming",
    "review", "tutorial", "trailer", "music video", "official video",
    "podcast", "interview", "talk show", "lecture", "live stream",
    "fails", "funny moments", "best of", "tiktok comp", "shorts compilation",
    "unboxing", "haul", "challenge",
]

# Negative search operators appended to every ytsearch query.
NEG_QUERY = " -vlog -reaction -gameplay -compilation -tutorial -review -podcast -trailer"

stats = {
    "downloaded": 0,
    "skipped_archive": 0,
    "filtered_title": 0,
    "filtered_duration": 0,
    "filtered_live": 0,
    "filtered_resolution": 0,
    "errors": 0,
}


def make_match_filter(min_dur, max_dur, min_side, max_side, min_tbr_kbps):
    """yt-dlp match_filter callable. Pre-checks duration, title, AND that at least
    one H.264 format meets resolution + bitrate, so we never download AV1-only
    uploads or visually-degraded low-bitrate streams."""
    def _f(info, *, incomplete=False):
        if incomplete:
            return None
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
            stats["filtered_live"] += 1
            return "skip: live"
        d = info.get("duration")
        if d is None or d < min_dur or d > max_dur:
            stats["filtered_duration"] += 1
            return f"skip: duration={d}"
        title = (info.get("title") or "").lower()
        for bad in BAD_TITLE_TOKENS:
            if bad in title:
                stats["filtered_title"] += 1
                return f"skip: title token '{bad}'"
        # Format-availability pre-check: must have at least one H.264 (avc1)
        # video-only format that meets resolution AND bitrate.
        formats = info.get("formats") or []
        if formats:
            def _ok(f):
                vc = (f.get("vcodec") or "").lower()
                if vc in ("", "none"):
                    return False
                if not vc.startswith(("avc1", "h264")):  # exclude av01, vp09
                    return False
                w, h = f.get("width") or 0, f.get("height") or 0
                if min(w, h) < min_side:
                    return False
                if max_side and max(w, h) > max_side:
                    return False
                tbr = f.get("tbr") or 0  # kbps
                if tbr and tbr < min_tbr_kbps:
                    return False
                return True
            if not any(_ok(f) for f in formats):
                stats["filtered_resolution"] += 1
                return "skip: no H.264 format meets resolution/bitrate"
        return None
    return _f


def progress_hook(d):
    if d.get("status") == "finished":
        stats["downloaded"] += 1


def build_opts(out_dir, archive_path, min_dur, max_dur, min_side, max_side,
               min_tbr_kbps, per_kw, cookiefile):
    # Format selector enforces: short side ∈ [min_side, max_side], H.264 codec,
    # and total bitrate ≥ min_tbr_kbps. AV1 (av01) and VP9 (vp09) are excluded
    # so files open in standard players; bitrate gate kills visually-trash uploads.
    side_pred = f"[width>={min_side}][height>={min_side}]"
    if max_side:
        side_pred += f"[width<={max_side}][height<={max_side}]"
    quality_pred = f"[vcodec~='^(avc1|h264)'][tbr>={min_tbr_kbps}]"
    pred = side_pred + quality_pred
    fmt = (
        f"bv*{pred}[ext=mp4]/"
        f"bv*{pred}/"
        f"b{pred}[ext=mp4]/"
        f"b{pred}"
    )

    opts = {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "format": fmt,
        "merge_output_format": "mp4",
        "match_filter": make_match_filter(min_dur, max_dur, min_side, max_side, min_tbr_kbps),
        "max_downloads": per_kw,
        "download_archive": archive_path,
        "writeinfojson": True,
        "allow_playlist_files": False,
        "ignoreerrors": True,
        # videos without any 1080p+ format are expected misses, not errors
        "ignore_no_formats_error": True,
        "no_warnings": True,
        "quiet": False,
        "noprogress": True,
        "progress_hooks": [progress_hook],
        "concurrent_fragment_downloads": 4,
        "retries": 3,
        "fragment_retries": 3,
        "source_address": "0.0.0.0",  # force ipv4
        "sleep_interval_requests": 1,
        "sleep_interval": 2,
        "max_sleep_interval": 5,
        # Enable Node.js as a JS runtime for nsig deobfuscation.
        # Without this, YouTube only serves ≤720p formats and our 1080p
        # format selector triggers "Requested format is not available".
        "js_runtimes": {"node": {}},
    }
    if cookiefile and os.path.exists(cookiefile):
        opts["cookiefile"] = cookiefile
    return opts


def search_and_download(keyword, out_dir, archive_path, search_pool, per_kw,
                       min_dur, max_dur, min_side, max_side, min_tbr_kbps, cookiefile):
    os.makedirs(out_dir, exist_ok=True)
    query = f"ytsearch{search_pool}:{keyword}{NEG_QUERY}"
    opts = build_opts(out_dir, archive_path, min_dur, max_dur,
                      min_side, max_side, min_tbr_kbps, per_kw, cookiefile)
    print(f"\n>>> [{keyword}] → {out_dir}")
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([query])
    except MaxDownloadsReached:
        # hitting per_kw is the success path, not an error
        return
    except DownloadError as e:
        print(f"  [!] {keyword}: {e}")
        stats["errors"] += 1
    except Exception as e:
        print(f"  [!] {keyword}: {e}")
        stats["errors"] += 1


def collect_tasks(tree, current_path, tasks):
    """Recursive walker matching pexels.py — dict keys are dirs, list leaves are keywords."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            safe = k.replace(" ", "_")
            collect_tasks(v, os.path.join(current_path, safe), tasks)
    elif isinstance(tree, list):
        for kw in tree:
            tasks.append((kw, current_path))


def maybe_scene_filter(save_dir):
    """Optional post-pass: drop any downloaded video that contains a scene cut."""
    try:
        from scenedetect import detect, ContentDetector
    except ImportError:
        print("[!] pyscenedetect not installed; skipping scene-cut filter")
        return
    removed = 0
    for root, _, files in os.walk(save_dir):
        for fn in files:
            if not fn.endswith(".mp4"):
                continue
            path = os.path.join(root, fn)
            try:
                scenes = detect(path, ContentDetector(threshold=27.0))
            except Exception as e:
                print(f"  [!] scene detect failed on {fn}: {e}")
                continue
            if len(scenes) > 1:
                os.remove(path)
                info_json = path.replace(".mp4", ".info.json")
                if os.path.exists(info_json):
                    os.remove(info_json)
                removed += 1
    print(f"[scene-filter] removed {removed} multi-cut videos")


# Standard ISO BMFF brands that mainstream players accept. YouTube's DASH-fragmented
# 1080p+ H.264 has ftyp brand 'dash', which QuickTime / Windows / browser <video>
# tags refuse, so we remux those in place via stream-copy (zero quality loss).
_STANDARD_FTYP_BRANDS = {b"isom", b"iso2", b"iso5", b"iso6", b"mp41", b"mp42"}


def _ftyp_brand(path):
    try:
        with open(path, "rb") as f:
            f.seek(8)
            return f.read(4)
    except OSError:
        return None


def remux_dash_to_standard(save_dir):
    """Scan all .mp4 under save_dir; remux any with non-standard ftyp brand to isom mp4.
    Stream-copy only — no re-encoding, picture/audio bit-exact preserved."""
    fixed = kept = failed = 0
    for root, _, files in os.walk(save_dir):
        for fn in files:
            if not fn.endswith(".mp4"):
                continue
            path = os.path.join(root, fn)
            brand = _ftyp_brand(path)
            if brand in _STANDARD_FTYP_BRANDS:
                kept += 1
                continue
            tmp = path + ".remux.tmp.mp4"
            r = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", path,
                 "-c", "copy", "-movflags", "+faststart", tmp],
                capture_output=True,
            )
            if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 1024:
                os.replace(tmp, path)
                fixed += 1
            else:
                if os.path.exists(tmp):
                    os.remove(tmp)
                err = (r.stderr or b"").decode(errors="replace").strip().splitlines()[-1:]
                print(f"  [!] remux failed: {path} :: {err}")
                failed += 1
    print(f"[remux] fixed={fixed} kept={kept} failed={failed}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json_file", default="target_distribution_yt.json",
                   help="Hierarchical keyword tree.")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--search_pool", type=int, default=120,
                   help="ytsearchN — candidates fetched per keyword before filtering.")
    p.add_argument("--per_kw", type=int, default=6,
                   help="Max successful downloads per keyword.")
    p.add_argument("--min_dur", type=int, default=3,
                   help="Lower bound on source duration (seconds).")
    p.add_argument("--max_dur", type=int, default=45,
                   help="Upper bound on source duration. Shorter source = higher static-cam ratio.")
    p.add_argument("--min_side", type=int, default=1080,
                   help="Reject if min(width,height) < this. 1080 = your 1024 training res margin.")
    p.add_argument("--max_side", type=int, default=2160,
                   help="Cap to avoid huge 4K/8K downloads. 0 = no cap.")
    p.add_argument("--min_tbr", type=int, default=500,
                   help="Minimum total bitrate in kbps. <500 at 1080p = visually trash.")
    p.add_argument("--max_workers", type=int, default=3,
                   help="Parallel keyword tasks. YT throttles aggressively; >6 risks rate-limit.")
    p.add_argument("--cookiefile", default=None,
                   help="Optional cookies.txt for age-gated content.")
    p.add_argument("--scene_filter", action="store_true",
                   help="After download, drop videos containing scene cuts (slow; needs pyscenedetect).")
    p.add_argument("--remux", action="store_true",
                   help="After download, remux DASH-fragmented mp4 → standard isom mp4 "
                        "(in place, stream-copy, zero quality loss). Fixes 'won't open' in "
                        "QuickTime / Windows / browser preview.")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        with open(args.json_file, encoding="utf-8") as f:
            tree = json.load(f)
    except FileNotFoundError:
        print(f"[!] Keyword file missing: {args.json_file}")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    archive = os.path.join(args.save_dir, ".dl_archive.txt")

    tasks = []
    collect_tasks(tree, args.save_dir, tasks)
    print(f"[info] {len(tasks)} keyword tasks queued under {args.save_dir}")
    print(f"[info] filters: dur∈[{args.min_dur},{args.max_dur}]s  "
          f"side∈[{args.min_side},{args.max_side or '∞'}]  "
          f"tbr≥{args.min_tbr}kbps  codec=H.264  "
          f"per_kw={args.per_kw}  pool={args.search_pool}")

    max_side = args.max_side if args.max_side > 0 else None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [
            ex.submit(search_and_download, kw, out_dir, archive,
                      args.search_pool, args.per_kw,
                      args.min_dur, args.max_dur, args.min_side, max_side,
                      args.min_tbr, args.cookiefile)
            for kw, out_dir in tasks
        ]
        for _ in as_completed(futs):
            pass

    if args.remux:
        print("\n[remux] scanning for DASH-fragmented mp4 to repackage...")
        remux_dash_to_standard(args.save_dir)

    if args.scene_filter:
        print("\n[scene-filter] scanning for multi-cut videos...")
        maybe_scene_filter(args.save_dir)

    print("\n" + "=" * 50)
    print(f"[done] {(time.time()-t0)/60:.1f} min")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
