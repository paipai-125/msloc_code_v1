"""Visualize XCLIP paired-neuron scores and the final 768-neuron union.

The input ``xclip_neuron_scores.npz`` is written by probe_xclip_neurons.py.
It contains a 768-channel score vector for each target (fake/R2F/F2R) and
XCLIP layer.  This script follows WAFL's ranked-per-layer heatmap convention,
then adds plots for the final fixed-width selector used by DeMamba.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCORE_KEY = re.compile(r"^(fake|r2f|f2r)_layer_(\d+)_score$")
TARGETS = ("fake", "r2f", "f2r")


def load_scores(path: Path):
    scores = {target: {} for target in TARGETS}
    means = {target: {} for target in TARGETS}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            match = SCORE_KEY.match(key)
            if not match:
                continue
            target, layer = match.group(1), int(match.group(2))
            scores[target][layer] = np.asarray(data[key], dtype=np.float64)
            mean_key = f"{target}_layer_{layer:02d}_mean_delta"
            means[target][layer] = np.asarray(data[mean_key], dtype=np.float64) if mean_key in data.files else np.full_like(scores[target][layer], np.nan)
    if not any(scores.values()):
        raise ValueError(f"No XCLIP target/layer score vectors found in {path}")
    return scores, means


def render_target(target, score_by_layer, mean_by_layer, ratio, output_dir):
    layers = sorted(score_by_layer)
    orders = {}
    width = 0
    for layer in layers:
        score = score_by_layer[layer]
        count = max(1, math.ceil(ratio * score.size))
        orders[layer] = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1][:count]
        width = max(width, count)
    matrix = np.full((len(layers), width), np.nan)
    csv_path = output_dir / f"{target}_top_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "layer", "rank", "channel_index", "score", "mean_delta"])
        for row, layer in enumerate(layers):
            score, mean, order = score_by_layer[layer], mean_by_layer[layer], orders[layer]
            matrix[row, :len(order)] = score[order]
            for rank, channel in enumerate(order, 1):
                writer.writerow([target, layer, rank, int(channel), float(score[channel]), float(mean[channel])])
    figure, axis = plt.subplots(figsize=(max(9, width * .10), max(4, len(layers) * .48)), dpi=180)
    image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="magma", vmin=0.0)
    axis.set_yticks(np.arange(len(layers)), [f"layer {layer:02d}" for layer in layers])
    axis.set_xlabel("candidate rank within layer")
    axis.set_title(f"XCLIP {target}: top {ratio:.0%} paired-neuron scores")
    figure.colorbar(image, ax=axis, label="|mean delta| / delta variability")
    figure.tight_layout()
    figure.savefig(output_dir / f"{target}_score_heatmap.png", bbox_inches="tight")
    plt.close(figure)


def selector_layers(selector):
    layers = selector.get("layers", selector.get("selected_indices", selector))
    return {int(layer.removeprefix("layer_")): sorted(map(int, indices)) for layer, indices in layers.items()}


def final_membership(selector):
    source = selector.get("final_by_target", selector.get("by_target", {}))
    membership = {target: set() for target in TARGETS}
    for target, layer_map in source.items():
        if target not in membership:
            continue
        for layer, indices in layer_map.items():
            layer = int(str(layer).removeprefix("layer_"))
            membership[target].update((layer, int(index)) for index in indices)
    return membership


def render_final_union(selector, scores, output_dir):
    layers = selector_layers(selector)
    membership = final_membership(selector)
    total = sum(len(indices) for indices in layers.values())
    if total != 768:
        raise ValueError(f"Expected exactly 768 final neurons, found {total} in selector")

    sorted_indices, width = {}, 0
    for layer, indices in sorted(layers.items()):
        def combined_score(index):
            return max(scores[target].get(layer, np.full(768, np.nan))[index] for target in TARGETS)
        sorted_indices[layer] = sorted(indices, key=combined_score, reverse=True)
        width = max(width, len(indices))

    matrix = np.full((len(layers), width), np.nan)
    with (output_dir / "final_768_neurons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "rank_within_layer", "channel_index", "combined_score", "fake_score", "r2f_score", "f2r_score", "selected_for_fake", "selected_for_r2f", "selected_for_f2r"])
        for row, (layer, indices) in enumerate(sorted(sorted_indices.items())):
            for rank, index in enumerate(indices, 1):
                target_scores = [float(scores[target][layer][index]) for target in TARGETS]
                combined = max(target_scores)
                matrix[row, rank - 1] = combined
                writer.writerow([layer, rank, index, combined, *target_scores,
                                 *(int((layer, index) in membership[target]) for target in TARGETS)])

    figure, axis = plt.subplots(figsize=(max(9, width * .10), max(4, len(layers) * .48)), dpi=180)
    image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis", vmin=0.0)
    ordered_layers = sorted(layers)
    axis.set_yticks(np.arange(len(ordered_layers)), [f"layer {layer:02d}" for layer in ordered_layers])
    axis.set_xlabel("selected neuron rank within layer")
    axis.set_title("Final 768 XCLIP neurons: maximum task score")
    figure.colorbar(image, ax=axis, label="maximum fake / R2F / F2R score")
    figure.tight_layout()
    figure.savefig(output_dir / "final_768_score_heatmap.png", bbox_inches="tight")
    plt.close(figure)

    counts = np.asarray([len(layers[layer]) for layer in ordered_layers])
    figure, axis = plt.subplots(figsize=(10, 4), dpi=180)
    axis.bar(ordered_layers, counts, color="#4c78a8")
    axis.axhline(64, color="#d62728", linestyle="--", linewidth=1, label="uniform 768 / 12 = 64")
    axis.set_xlabel("XCLIP layer")
    axis.set_ylabel("final selected neurons")
    axis.set_title("Distribution of the final 768 selected neurons")
    axis.set_xticks(ordered_layers)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "final_768_layer_distribution.png", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-ratio", type=float, default=None,
                        help="Heatmap candidate ratio; defaults to the selector's recorded ratio")
    args = parser.parse_args()

    selector = json.loads(args.selector.read_text(encoding="utf-8"))
    ratio = args.top_ratio if args.top_ratio is not None else float(selector.get("top_ratio_per_target", .10))
    if not 0 < ratio <= 1:
        parser.error("--top-ratio must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores, means = load_scores(args.scores)
    for target in TARGETS:
        if not scores[target]:
            raise ValueError(f"No score vectors for target {target}")
        render_target(target, scores[target], means[target], ratio, args.output_dir)
    render_final_union(selector, scores, args.output_dir)
    print(f"Wrote XCLIP neuron heatmaps and CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()
