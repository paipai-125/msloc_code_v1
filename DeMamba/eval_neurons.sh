#!/usr/bin/env bash
# Run from MSLoc_code: bash DeMamba/eval_neurons.sh
set -euo pipefail
python DeMamba/eval.py \
  --config DeMamba/configs/XCLIP_Tasle_neurons.yaml \
  --model_path ../MSLoc_data/DeMamba/results/xclip_neurons_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/xclip_neurons_4/eval
