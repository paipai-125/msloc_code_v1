#!/usr/bin/env bash
# Run this script from the MSLoc_code directory:
#   bash DeMamba/run_neuron_pipeline.sh
# Every path is deliberately relative to that working directory.
set -euo pipefail

command -v ffmpeg >/dev/null || { echo "ffmpeg is required for frame extraction"; exit 1; }

python DeMamba/Preprocess/video2frame.py \
  --input_root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --output_root ../MSLoc_data/DeMamba/video_frames \
  --num_workers 8

python DeMamba/build_probe_pairs.py \
  --annotations ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --output ../MSLoc_data/DeMamba/neuron_probe/train_pairs.jsonl \
  --window-length 2.0

python DeMamba/probe_xclip_neurons.py \
  --pairs ../MSLoc_data/DeMamba/neuron_probe/train_pairs.jsonl \
  --frame-root ../MSLoc_data/DeMamba/video_frames \
  --model-path ../MSLoc_data/DeMamba/pretrained_weights/xclip-base-patch16 \
  --output-dir ../MSLoc_data/DeMamba/neuron_probe \
  --top-ratio 0.10 \
  --final-neuron-count 768 \
  --min-neurons-per-target 256 \
  --frames-per-window 8 \
  --crop-youku \
  --amp \
  --strict

python DeMamba/train.py --config DeMamba/configs/XCLIP_Tasle_neurons.yaml

python DeMamba/eval.py \
  --config DeMamba/configs/XCLIP_Tasle_neurons.yaml \
  --model_path ../MSLoc_data/DeMamba/results/xclip_neurons_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/xclip_neurons_4/eval
