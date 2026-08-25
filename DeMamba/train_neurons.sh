#!/usr/bin/env bash
# Run from MSLoc_code: bash DeMamba/train_neurons.sh
set -euo pipefail
python DeMamba/train.py --config DeMamba/configs/XCLIP_Tasle_neurons.yaml
