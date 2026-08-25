#!/bin/bash
# Train DeMamba (small XCLIP-based) classifier on Tasle-CoT-10K.
# Run from MSLoc/DeMamba directory.
set -e

cd "$(dirname "$0")"

# Point to the pre-trained XCLIP backbone in MSLoc_data
export XCLIP_WEIGHTS_PATH="../../MSLoc_data/DeMamba/pretrained_weights/xclip-base-patch16"

# four-class
python train.py --config ./configs/XCLIP_Tasle.yaml

# binary
# python train.py --config ./configs/03.yaml
