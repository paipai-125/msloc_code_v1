"""Create a deterministic fake/real paired subset from Tasle-CoT-10K annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def infer_real_path(fake_path: str) -> str:
    stem, suffix = fake_path.rsplit(".", 1)
    return fake_path if stem.endswith("_real") else f"{stem}_real.{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fake-count", type=int, required=True)
    args = parser.parse_args()
    if args.fake_count <= 0:
        parser.error("--fake-count must be positive")

    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    real_by_path = {record.get("video_path"): record for record in records if record.get("type") == "real"}
    subset = []
    for record in records:
        if record.get("type") != "fake" or not record.get("video_path"):
            continue
        real = real_by_path.get(infer_real_path(record["video_path"]))
        if real is None:
            continue
        subset.extend((record, real))
        if len(subset) // 2 >= args.fake_count:
            break
    if len(subset) // 2 != args.fake_count:
        raise RuntimeError(f"Found only {len(subset)//2}/{args.fake_count} fake records with an annotated _real counterpart")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}: {args.fake_count} fake + {args.fake_count} real records")


if __name__ == "__main__":
    main()
