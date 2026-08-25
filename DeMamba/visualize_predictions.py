"""Render ground-truth and predicted temporal segments as PNG timelines."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def read_segments(record, key):
    if key == "ground_truth":
        items = record.get("annotations", [])
        return [tuple(map(float, item["segment"])) for item in items if item.get("segment") and len(item["segment"]) == 2]
    inference = record.get("model_inference", {})
    return [tuple(map(float, segment)) for segment in inference.get("segment", []) if len(segment) == 2]


def safe_name(video_path, index):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(video_path)).strip("_")
    return f"{index:04d}_{stem}.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="predictions.json produced by DeMamba/eval.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--only-fake", action="store_true",
                        help="Render only records whose ground truth has a fake segment")
    args = parser.parse_args()

    records = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("predictions.json must contain a JSON list")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for record in records:
        ground_truth = read_segments(record, "ground_truth")
        if args.only_fake and not ground_truth:
            continue
        if args.max_videos and len(index_rows) >= args.max_videos:
            break
        predicted = read_segments(record, "prediction")
        duration = float(record.get("duration", 0.0))
        if duration <= 0:
            endpoints = [end for _, end in ground_truth + predicted]
            duration = max(endpoints, default=1.0)

        figure, axis = plt.subplots(figsize=(12, 2.6))
        for y, label, segments, color in ((1, "ground truth", ground_truth, "#d62728"),
                                          (0, "prediction", predicted, "#1f77b4")):
            axis.barh(y, duration, left=0, height=0.42, color="#eeeeee", edgecolor="none")
            for start, end in segments:
                axis.barh(y, max(0.0, end - start), left=start, height=0.42, color=color)
        axis.set_xlim(0, duration)
        axis.set_yticks([0, 1], ["prediction", "ground truth"])
        axis.set_xlabel("time (seconds)")
        axis.set_title(str(record.get("video_path", "unknown video")), fontsize=9)
        axis.grid(axis="x", alpha=0.25)
        axis.legend(handles=[Patch(color="#d62728", label="ground-truth fake"),
                             Patch(color="#1f77b4", label="predicted fake")],
                    loc="upper right", fontsize=8)
        figure.tight_layout()

        filename = safe_name(record.get("video_path", "unknown"), len(index_rows))
        figure.savefig(args.output_dir / filename, dpi=160)
        plt.close(figure)
        index_rows.append({"video_path": record.get("video_path", ""), "image": filename,
                           "ground_truth_segments": ground_truth, "predicted_segments": predicted})

    with (args.output_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "image", "ground_truth_segments", "predicted_segments"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Wrote {len(index_rows)} timeline image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
