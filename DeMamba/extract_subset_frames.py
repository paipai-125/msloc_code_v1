"""Extract 8-fps frames only for videos listed in one or more annotation JSON files."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
from pathlib import Path


def extract(task):
    video_root, frame_root, relative_path, fps = task
    source = video_root / relative_path
    destination = frame_root / relative_path.with_suffix("")
    if list(destination.glob("frame_*.jpg")):
        return "skip", str(relative_path)
    if not source.is_file():
        return "missing", str(source)
    destination.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
         "-r", str(fps), str(destination / "frame_%d.jpg")],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode:
        return "failed", f"{source}: {result.stderr.strip()}"
    return "done", str(relative_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if args.fps <= 0 or args.num_workers <= 0:
        parser.error("--fps and --num-workers must be positive")

    videos = set()
    for annotation_path in args.annotations:
        records = json.loads(annotation_path.read_text(encoding="utf-8"))
        videos.update(record["video_path"] for record in records if record.get("video_path"))
    video_root, output_root = args.video_root.resolve(), args.output_root.resolve()
    tasks = [(video_root, output_root, Path(video), args.fps) for video in sorted(videos)]
    print(f"Extracting frames for {len(tasks)} unique videos")
    counts = {"done": 0, "skip": 0, "missing": 0, "failed": 0}
    with mp.Pool(args.num_workers) as pool:
        for status, message in pool.imap_unordered(extract, tasks):
            counts[status] += 1
            if status in {"missing", "failed"}:
                print(f"{status}: {message}")
    print(counts)
    if counts["missing"] or counts["failed"]:
        raise RuntimeError("Some subset videos could not be extracted")


if __name__ == "__main__":
    main()
