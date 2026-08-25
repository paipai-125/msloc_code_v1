#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect every unique class name from a training-split JSON.

Rules:
    bnd_cot_st / bnd_cot_ed -> 'bnd_class'
    obj_cot                 -> 'bnd_sub_class'

Usage:
    python extract_unique_classes.py \
        --data_path /path/to/train_all_1209.json \
        --output_path /path/to/class_names.txt
"""

import os
import json
import argparse
from collections import defaultdict


def safe_get_bnd_class(container):
    if container is None:
        return ''
    if isinstance(container, dict):
        return container.get('bnd_class', '')
    if isinstance(container, list) and len(container) > 0:
        return container[0].get('bnd_class', '') if isinstance(container[0], dict) else ''
    return ''


def safe_get_obj_class(container):
    if container is None:
        return ''
    if isinstance(container, dict):
        return container.get('bnd_sub_class', '')
    if isinstance(container, list) and len(container) > 0:
        return container[0].get('bnd_sub_class', '') if isinstance(container[0], dict) else ''
    return ''


def extract_classes_from_annotation(ann):
    classes = []
    c1 = safe_get_bnd_class(ann.get('bnd_cot_st'))
    if c1:
        classes.append(('bnd_cot_st', c1))
    c2 = safe_get_obj_class(ann.get('obj_cot'))
    if c2:
        classes.append(('obj_cot', c2))
    c3 = safe_get_bnd_class(ann.get('bnd_cot_ed'))
    if c3:
        classes.append(('bnd_cot_ed', c3))
    return classes


def extract_unique_classes(data_path: str, output_path: str = None, verbose: bool = True):
    """Return a sorted list of every unique class name found in ``data_path``."""
    print("=" * 60)
    print("Extracting Unique Classes from Training Data")
    print("=" * 60)

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Total items in data: {len(data)}")

    all_classes = set()
    class_sources = defaultdict(set)
    class_counts = defaultdict(int)

    for item in data:
        annotations = item.get('annotations', [])
        if not isinstance(annotations, list):
            annotations = [annotations]
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            for source, class_name in extract_classes_from_annotation(ann):
                if class_name:
                    all_classes.add(class_name)
                    class_sources[class_name].add(source)
                    class_counts[class_name] += 1

    unique_classes = sorted(all_classes)

    print(f"\n{'=' * 60}")
    print(f"Found {len(unique_classes)} unique classes:")
    print(f"{'=' * 60}")
    if verbose:
        for i, cls in enumerate(unique_classes):
            sources = ', '.join(sorted(class_sources[cls]))
            print(f"  [{i + 1:2d}] '{cls}' (count: {class_counts[cls]}, from: {sources})")

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for cls in unique_classes:
                f.write(cls + '\n')
        print(f"\nClasses saved to: {output_path}")

    python_list_str = "[\n" + ",\n".join([f'    "{cls}"' for cls in unique_classes]) + "\n]"
    comma_separated = ",".join(unique_classes)

    print(f"\n{'=' * 60}")
    print("Python list format (for extract_class_features.py):")
    print(f"{'=' * 60}")
    print(python_list_str)

    print(f"\n{'=' * 60}")
    print("Comma-separated format (for --class_names argument):")
    print(f"{'=' * 60}")
    print(comma_separated)

    print(f"\n{'=' * 60}")
    print("Usage:")
    print(f"{'=' * 60}")
    print("python trace/scripts/extract_class_features.py \\")
    print("    --bge_model_path /path/to/bge-large-en-v1.5 \\")
    print("    --output_path /path/to/class_features_bge.pt \\")
    print(f'    --class_names "{comma_separated}"')

    return unique_classes


def main():
    parser = argparse.ArgumentParser(description="Extract unique classes from training data")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/Tasle-CoT-10K/annos/train_all_1209.json",
        help="Path to training data JSON file (relative to MSLoc/Trace by default).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for class names (one per line)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    extract_unique_classes(
        data_path=args.data_path,
        output_path=args.output_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
