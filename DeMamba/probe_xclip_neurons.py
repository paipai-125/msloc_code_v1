"""Probe discriminative neurons in a frozen microsoft/xclip-base-patch16.

For every time-aligned fake/normal pair, the script accumulates only channel
statistics.  It never serialises activation tensors.  ``fake`` uses the mean
fake-minus-normal response; boundary targets use the change of that response
between the first and second half of the window, which preserves direction.

Multi-GPU: pairs are split evenly across all visible GPUs; each GPU runs in a
separate subprocess and the per-layer RunningVectorStats are merged in the
main process before neuron selection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import XCLIPVisionModel


CLIP_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
CLIP_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class RunningVectorStats:
    def __init__(self, width):
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, value):
        value = np.asarray(value, dtype=np.float64)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def score(self, eps=1e-4):
        if self.count < 2:
            return np.zeros_like(self.mean)
        std = np.sqrt(self.m2 / (self.count - 1))
        valid = std[std > eps]
        floor = max(eps, float(np.median(valid)) * 0.05) if valid.size else 1.0
        return np.abs(self.mean) / np.sqrt(std * std + floor * floor)


def read_pairs(path):
    pairs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not pairs:
        raise ValueError(f"No probe pairs in {path}")
    invalid = [pair.get("target") for pair in pairs if pair.get("target") not in {"fake", "r2f", "f2r"}]
    if invalid:
        raise ValueError(f"Unsupported pair target(s): {sorted(set(invalid))}")
    return pairs


def frame_directory(frame_root, video_path):
    relative = os.path.splitext(str(video_path))[0]
    return Path(frame_root) / relative


def load_window(frame_root, video_path, start, end, frames_per_window, image_size, crop_youku,
                reference_times=None):
    directory = frame_directory(frame_root, video_path)
    files = sorted(directory.glob("frame_*.jpg"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not files:
        raise FileNotFoundError(f"No extracted frames found in {directory}")
    fps = 8.0
    first = max(0, int(start * fps))
    last = min(len(files) - 1, max(first, int(end * fps)))
    if reference_times is not None:
        times = np.asarray(reference_times, dtype=np.float32)
        if times.shape != (frames_per_window,):
            raise ValueError(f"Expected {frames_per_window} reference times, got shape {times.shape}")
        indices = np.clip(np.rint(times * fps).astype(np.int64), 0, len(files) - 1).tolist()
    elif last - first + 1 >= frames_per_window:
        step = max(1, (last - first) // frames_per_window)
        indices = list(range(first, last + 1, step))[:frames_per_window]
        times = np.asarray(indices, dtype=np.float32) / fps
    else:
        indices = list(range(first, last + 1))
        indices.extend([indices[-1]] * (frames_per_window - len(indices)))
        times = np.asarray(indices, dtype=np.float32) / fps
    images = []
    for index in indices:
        image = cv2.imread(str(files[int(index)]))
        if image is None:
            raise ValueError(f"Unreadable frame: {files[int(index)]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if crop_youku and "youku" in str(files[int(index)]).lower():
            height, width = image.shape[:2]
            if width > height:
                image = image[:, int(width * .15):int(width * .85)]
            else:
                image = image[int(height * .15):int(height * .85), :]
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        images.append((image - CLIP_MEAN) / CLIP_STD)
    tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    return tensor, times


def paired_delta(hidden, sample_times, target, boundary_time=None):
    # hidden: [2*T, tokens, channels], with fake frames preceding normal ones.
    frames_per_window = len(sample_times)
    patch_mean = hidden[:, 1:, :].float().mean(dim=1).reshape(2, frames_per_window, -1)
    difference = patch_mean[0] - patch_mean[1]
    if target == "fake":
        return difference.mean(dim=0)
    if boundary_time is None:
        raise ValueError(f"Boundary time is required for target {target}")
    pre_mask = torch.from_numpy(sample_times < float(boundary_time)).to(difference.device)
    post_mask = ~pre_mask
    if not bool(pre_mask.any()) or not bool(post_mask.any()):
        raise ValueError(
            f"No samples on both sides of boundary {boundary_time} in window times "
            f"[{sample_times[0]}, {sample_times[-1]}]"
        )
    pre, post = difference[pre_mask].mean(dim=0), difference[post_mask].mean(dim=0)
    return post - pre if target == "r2f" else pre - post


def _worker(rank: int, world_size: int, args, pairs_chunk: list, result_queue):
    """Run on a single GPU (rank).  Puts (stats_dict, failures, completed) into result_queue."""
    device = torch.device(f"cuda:{rank}")
    model = XCLIPVisionModel.from_pretrained(
        args.model_path, local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    config = model.config
    image_size = int(config.image_size)

    stats = {
        target: {
            layer: RunningVectorStats(int(config.hidden_size))
            for layer in range(1, int(config.num_hidden_layers) + 1)
        }
        for target in ("fake", "r2f", "f2r")
    }
    failures, completed = [], 0

    desc = f"GPU {rank}"
    pbar = tqdm(pairs_chunk, desc=desc, unit="pair", dynamic_ncols=True, position=rank, leave=True)
    for pair in pbar:
        try:
            start, end = map(float, pair["window"])
            fake, sample_times = load_window(
                args.frame_root, pair["fake_video"], start, end,
                args.frames_per_window, image_size, args.crop_youku,
            )
            normal, _ = load_window(
                args.frame_root, pair["normal_video"], start, end,
                args.frames_per_window, image_size, args.crop_youku,
                reference_times=sample_times,
            )
            batch = torch.cat((fake, normal), dim=0).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=args.amp,
            ):
                outputs = model(batch, output_hidden_states=True)
            for layer, hidden in enumerate(outputs.hidden_states[1:], 1):
                delta = paired_delta(hidden, sample_times, pair["target"], pair.get("boundary_time"))
                stats[pair["target"]][layer].update(delta.cpu().numpy())
            completed += 1
            pbar.set_postfix(done=completed, fail=len(failures))
        except FileNotFoundError as error:
            # 帧目录缺失属于数据问题，始终跳过，不受 --strict 影响
            message = f"{pair.get('pair_id', '<unknown>')}: {type(error).__name__}: {error}"
            failures.append(message)
        except Exception as error:
            message = f"{pair.get('pair_id', '<unknown>')}: {type(error).__name__}: {error}"
            if args.strict:
                raise RuntimeError(message) from error
            failures.append(message)

    result_queue.put((stats, failures, completed))


def _merge_stats(base: RunningVectorStats, other: RunningVectorStats) -> RunningVectorStats:
    """Merge two RunningVectorStats using Chan's parallel algorithm."""
    if other.count == 0:
        return base
    if base.count == 0:
        base.count = other.count
        base.mean = other.mean.copy()
        base.m2 = other.m2.copy()
        return base
    combined_count = base.count + other.count
    delta = other.mean - base.mean
    base.m2 = base.m2 + other.m2 + delta ** 2 * base.count * other.count / combined_count
    base.mean = (base.mean * base.count + other.mean * other.count) / combined_count
    base.count = combined_count
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Local microsoft/xclip-base-patch16 directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-ratio", type=float, default=.10,
                        help="Per-layer candidate ratio for each of fake/R2F/F2R")
    parser.add_argument("--final-neuron-count", type=int, default=768,
                        help="Exact number of union neurons written for direct Mamba input")
    parser.add_argument("--min-neurons-per-target", type=int, default=256,
                        help="At least this many high-score candidates from each target are protected before filling the final union")
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--crop-youku", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop instead of skipping unreadable pairs")
    args = parser.parse_args()
    if (not 0 < args.top_ratio <= 1 or args.frames_per_window < 2 or args.frames_per_window % 2
            or args.final_neuron_count <= 0 or args.min_neurons_per_target < 0):
        parser.error("top ratio must be in (0, 1] and frames per window must be a positive even number")
    if 3 * args.min_neurons_per_target > args.final_neuron_count:
        parser.error("3 * --min-neurons-per-target cannot exceed --final-neuron-count")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"XCLIP model directory not found: {args.model_path}")

    pairs = read_pairs(args.pairs)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    # ── 多卡并行 ──────────────────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count() if args.device == "cuda" else 0
    if n_gpus > 1:
        print(f"Found {n_gpus} GPUs. Loading model on GPU 0 to verify config ...")
        _model_check = XCLIPVisionModel.from_pretrained(
            args.model_path, local_files_only=True
        ).to("cuda:0")
        config = _model_check.config
        del _model_check
        torch.cuda.empty_cache()

        if int(config.hidden_size) != 768 or int(config.patch_size) != 16:
            raise ValueError(
                "This probe targets microsoft/xclip-base-patch16 (hidden_size=768, patch_size=16); "
                f"loaded hidden_size={config.hidden_size}, patch_size={config.patch_size}."
            )

        # 均匀切分 pairs
        chunks = [pairs[i::n_gpus] for i in range(n_gpus)]
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        print(f"Spawning {n_gpus} worker processes ({len(pairs)} pairs total) ...")
        processes = []
        for rank in range(n_gpus):
            p = ctx.Process(
                target=_worker,
                args=(rank, n_gpus, args, chunks[rank], result_queue),
                daemon=False,
            )
            p.start()
            processes.append(p)

        # 必须先 get() 再 join()，否则 pipe 缓冲区满会导致 worker 阻塞，主进程死锁
        all_stats = None
        all_failures, total_completed = [], 0
        for _ in range(n_gpus):
            worker_stats, worker_failures, worker_completed = result_queue.get()
            all_failures.extend(worker_failures)
            total_completed += worker_completed
            if all_stats is None:
                all_stats = worker_stats
            else:
                for target in ("fake", "r2f", "f2r"):
                    for layer in worker_stats[target]:
                        all_stats[target][layer] = _merge_stats(
                            all_stats[target][layer], worker_stats[target][layer]
                        )

        for p in processes:
            p.join()
        stats = all_stats
        failures = all_failures
        completed = total_completed
        image_size = int(config.image_size)

    else:
        # ── 单卡 / CPU 路径（保持原逻辑）────────────────────────────────────
        device = torch.device(args.device)
        print(f"Loading XCLIP model from {args.model_path} ...")
        model = XCLIPVisionModel.from_pretrained(
            args.model_path, local_files_only=True
        ).to(device).eval()
        print("Model loaded.")
        model.requires_grad_(False)
        config = model.config
        if int(config.hidden_size) != 768 or int(config.patch_size) != 16:
            raise ValueError(
                "This probe targets microsoft/xclip-base-patch16 (hidden_size=768, patch_size=16); "
                f"loaded hidden_size={config.hidden_size}, patch_size={config.patch_size}."
            )
        image_size = int(config.image_size)
        stats = {
            target: {
                layer: RunningVectorStats(int(config.hidden_size))
                for layer in range(1, int(config.num_hidden_layers) + 1)
            }
            for target in ("fake", "r2f", "f2r")
        }
        failures, completed = [], 0
        pbar = tqdm(pairs, desc="Probing pairs", unit="pair", dynamic_ncols=True)
        for pair in pbar:
            try:
                start, end = map(float, pair["window"])
                fake, sample_times = load_window(
                    args.frame_root, pair["fake_video"], start, end,
                    args.frames_per_window, image_size, args.crop_youku,
                )
                normal, _ = load_window(
                    args.frame_root, pair["normal_video"], start, end,
                    args.frames_per_window, image_size, args.crop_youku,
                    reference_times=sample_times,
                )
                batch = torch.cat((fake, normal), dim=0).to(device)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=args.amp and device.type == "cuda",
                ):
                    outputs = model(batch, output_hidden_states=True)
                for layer, hidden in enumerate(outputs.hidden_states[1:], 1):
                    delta = paired_delta(hidden, sample_times, pair["target"], pair.get("boundary_time"))
                    stats[pair["target"]][layer].update(delta.cpu().numpy())
                completed += 1
                pbar.set_postfix(done=completed, fail=len(failures))
            except FileNotFoundError as error:
                # 帧目录缺失属于数据问题，始终跳过，不受 --strict 影响
                message = f"{pair.get('pair_id', '<unknown>')}: {type(error).__name__}: {error}"
                failures.append(message)
            except Exception as error:
                message = f"{pair.get('pair_id', '<unknown>')}: {type(error).__name__}: {error}"
                if args.strict:
                    raise RuntimeError(message) from error
                failures.append(message)

    # ── 以下神经元选择逻辑不变 ────────────────────────────────────────────────
    arrays, selected_by_target, candidate_scores = {}, {}, {}
    for target, layer_stats in stats.items():
        selected_by_target[target] = {}
        for layer, state in layer_stats.items():
            score = state.score()
            count = max(1, math.ceil(args.top_ratio * score.size)) if state.count else 0
            indices = np.argsort(score)[::-1][:count].astype(np.int32)
            arrays[f"{target}_layer_{layer:02d}_score"] = score.astype(np.float32)
            arrays[f"{target}_layer_{layer:02d}_mean_delta"] = state.mean.astype(np.float32)
            arrays[f"{target}_layer_{layer:02d}_top_indices"] = indices
            if count:
                selected_by_target[target][str(layer)] = indices.tolist()
                for index in indices.tolist():
                    candidate_scores.setdefault((layer, index), {})[target] = float(score[index])
    if not candidate_scores:
        raise RuntimeError("No valid pairs were processed; no selector was written")

    # Each target first proposes its own per-layer top-ratio candidates.  We
    # protect its strongest candidates, then fill remaining slots with the
    # largest score of every candidate in their union.  The result contains
    # *exactly* final_neuron_count channels and can therefore enter the
    # original 768-wide Mamba with no learned projection layer.
    protected = set()
    for target in ("fake", "r2f", "f2r"):
        ranked_target = sorted(
            ((scores[target], coordinate) for coordinate, scores in candidate_scores.items() if target in scores),
            reverse=True,
        )
        if len(ranked_target) < args.min_neurons_per_target:
            raise RuntimeError(
                f"Target {target!r} produced only {len(ranked_target)} candidates; "
                f"cannot protect {args.min_neurons_per_target}."
            )
        protected.update(coordinate for _, coordinate in ranked_target[:args.min_neurons_per_target])
    if len(protected) > args.final_neuron_count:
        raise RuntimeError(
            f"Protected target candidates occupy {len(protected)} slots, exceeding "
            f"--final-neuron-count={args.final_neuron_count}. Reduce --min-neurons-per-target."
        )
    ranked_union = sorted(
        ((max(scores.values()), coordinate) for coordinate, scores in candidate_scores.items()),
        reverse=True,
    )
    final_selected = set(protected)
    for _, coordinate in ranked_union:
        if len(final_selected) >= args.final_neuron_count:
            break
        final_selected.add(coordinate)
    if len(final_selected) != args.final_neuron_count:
        raise RuntimeError(
            f"Candidate union contains only {len(final_selected)} neurons, but "
            f"--final-neuron-count={args.final_neuron_count} was requested. Increase --top-ratio."
        )
    final_layers = {}
    for layer, index in sorted(final_selected):
        final_layers.setdefault(str(layer), []).append(index)
    final_by_target = {
        target: {
            str(layer): sorted(index for selected_layer, index in final_selected
                               if selected_layer == layer and target in candidate_scores[(selected_layer, index)])
            for layer in range(1, int(config.num_hidden_layers) + 1)
            if any(selected_layer == layer and target in candidate_scores[(selected_layer, index)]
                   for selected_layer, index in final_selected)
        }
        for target in ("fake", "r2f", "f2r")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "xclip_neuron_scores.npz", **arrays)
    selector = {
        "schema_version": 2,
        "model": "microsoft/xclip-base-patch16",
        "hidden_size": int(config.hidden_size),
        "patch_size": int(config.patch_size),
        "image_size": image_size,
        "num_hidden_layers": int(config.num_hidden_layers),
        "top_ratio_per_target": args.top_ratio,
        "final_neuron_count": args.final_neuron_count,
        "min_neurons_per_target": args.min_neurons_per_target,
        "processed_pairs": completed,
        "layers": final_layers,
        "candidate_by_target": selected_by_target,
        "final_by_target": final_by_target,
    }
    (args.output_dir / "xclip_neuron_indices.json").write_text(json.dumps(selector, indent=2), encoding="utf-8")
    (args.output_dir / "probe_failures.txt").write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    print(f"Processed {completed}/{len(pairs)} pairs. Selector: {args.output_dir / 'xclip_neuron_indices.json'}")
    if failures:
        print(f"Skipped {len(failures)} invalid pair(s); see {args.output_dir / 'probe_failures.txt'}")


if __name__ == "__main__":
    main()
