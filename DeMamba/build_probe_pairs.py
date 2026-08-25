"""Build train-only fake/normal pairs for frozen-XCLIP neuron probing.

The normal counterpart can be supplied in the annotation record itself or in a
separate JSON/JSONL map.  For Tasle-CoT-10K it is normally inferred directly:
``same/directory/fake_name.mp4`` maps to
``same/directory/fake_name_real.mp4``. Accepted aliases are ``fake_video``,
``fake_video_path`` or ``video_path`` for the fake path and ``normal_video``,
``normal_video_path`` or ``original_video_path`` for its counterpart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: Path):
    text = path.read_text(encoding="utf-8")
    value = json.loads(text)
    return value if isinstance(value, list) else value.get("pairs", [])


def load_pair_map(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    records = json.loads(text) if text.startswith("[") or text.startswith("{") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(records, dict):
        if "pairs" in records or "records" in records:
            records = records.get("pairs", records.get("records", []))
        else:
            # Also accept the compact {"fake/path.mp4": "normal/path.mp4"}
            # mapping, which is convenient for manually curated counterparts.
            records = [{"fake_video": fake, "normal_video": normal} for fake, normal in records.items()]
    mapping = {}
    for record in records:
        fake = record.get("fake_video") or record.get("fake_video_path") or record.get("video_path")
        normal = record.get("normal_video") or record.get("normal_video_path") or record.get("original_video_path")
        if fake and normal:
            mapping[str(fake)] = str(normal)
    if not mapping:
        raise ValueError(f"No fake/normal mapping records found in {path}")
    return mapping


def append_pair(result, fake_video, normal_video, target, start, end, boundary=None):
    result.append({
        "pair_id": f"{Path(fake_video).stem}_{target}_{len(result):06d}",
        "target": target,
        "fake_video": fake_video,
        "normal_video": normal_video,
        "window": [round(float(start), 6), round(float(end), 6)],
        "boundary_time": None if boundary is None else round(float(boundary), 6),
    })


def normal_path_from_record(record):
    return (record.get("normal_video") or record.get("normal_video_path") or
            record.get("original_video") or record.get("original_video_path"))


def infer_real_counterpart(fake_video):
    """Tasle-CoT-10K convention: ``clip.mp4`` -> ``clip_real.mp4``."""
    directory, filename = str(fake_video).rsplit("/", 1) if "/" in str(fake_video) else ("", str(fake_video))
    stem, suffix = filename.rsplit(".", 1) if "." in filename else (filename, "")
    if stem.endswith("_real"):
        return fake_video
    real_name = f"{stem}_real.{suffix}" if suffix else f"{stem}_real"
    return f"{directory}/{real_name}" if directory else real_name


def centred_boundary_window(boundary, duration, window_length):
    """Return a full window containing a boundary, shifted at video ends."""
    if duration < window_length:
        return None
    start = min(max(boundary - window_length / 2.0, 0.0), duration - window_length)
    end = start + window_length
    return start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Train annotation JSON")
    parser.add_argument("--pair-map", type=Path,
                        help="Optional JSON/JSONL fake-to-normal mapping; required unless annotations contain normal paths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-length", type=float, default=2.0)
    parser.add_argument("--interior-margin", type=float, default=0.0,
                        help="Minimum distance from a pure-fake window to either boundary")
    parser.add_argument("--interior-stride", type=float, default=2.0)
    parser.add_argument("--boundary-min-context", type=float, default=0.50,
                        help="Minimum true/fake context required on both sides of a boundary (seconds)")
    args = parser.parse_args()
    if (args.window_length <= 0 or args.interior_margin < 0 or args.interior_stride <= 0
            or args.boundary_min_context < 0):
        parser.error("window length/stride must be positive and margins must be non-negative")

    mapping = load_pair_map(args.pair_map) if args.pair_map else {}
    pairs = []
    unresolved = []
    for item in load_records(args.annotations):
        if item.get("type") != "fake":
            continue
        fake_video = item["video_path"]
        normal_video = mapping.get(fake_video) or normal_path_from_record(item) or infer_real_counterpart(fake_video)
        if not normal_video:
            unresolved.append(fake_video)
            continue
        duration = float(item["duration"])
        for annotation in item.get("annotations", []):
            if annotation.get("segment_label", "fake") != "fake" or "segment" not in annotation:
                continue
            start, end = map(float, annotation["segment"])
            if end <= start:
                continue
            # Prefer a centred boundary window.  If a segment begins/ends
            # close to a video edge, shift the full window into the video and
            # keep the true boundary time in the pair manifest.
            # A boundary needs real sampled frames on both sides.  At the
            # very beginning/end of a video an annotated boundary can have
            # only a few milliseconds of context, which cannot yield a valid
            # R2F/F2R contrast at 8 fps.  Exclude it here rather than failing
            # the strict probe later.
            r2f_window = (centred_boundary_window(start, duration, args.window_length)
                          if start >= args.boundary_min_context else None)
            f2r_window = (centred_boundary_window(end, duration, args.window_length)
                          if duration - end >= args.boundary_min_context else None)
            if r2f_window:
                append_pair(pairs, fake_video, normal_video, "r2f", *r2f_window, start)
            if f2r_window:
                append_pair(pairs, fake_video, normal_video, "f2r", *f2r_window, end)

            # Pure fake windows are useful for selecting artifact-sensitive
            # channels and are deliberately kept away from both transitions.
            left = start + args.interior_margin
            right = end - args.interior_margin - args.window_length
            cursor = left
            while cursor <= right + 1e-8:
                append_pair(pairs, fake_video, normal_video, "fake", cursor, cursor + args.window_length)
                cursor += args.interior_stride

    if not pairs:
        raise RuntimeError("No valid probe pairs. Check the train split, pair map, and temporal alignment.")
    counts = {target: sum(pair["target"] == target for pair in pairs) for target in ("fake", "r2f", "f2r")}
    missing_targets = [target for target, count in counts.items() if count == 0]
    if missing_targets:
        raise RuntimeError(
            f"Missing required probe target(s): {missing_targets}. "
            "Check segment durations and fake-to-normal correspondences."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8")
    print(f"Wrote {len(pairs)} train-only pairs to {args.output}: {counts}")
    if unresolved:
        unresolved_path = args.output.with_suffix(".unmapped_fake_videos.txt")
        unresolved_path.write_text("\n".join(sorted(set(unresolved))) + "\n", encoding="utf-8")
        print(f"Skipped {len(set(unresolved))} fake videos without normal counterparts; see {unresolved_path}")


if __name__ == "__main__":
    main()
