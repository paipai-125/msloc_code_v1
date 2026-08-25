#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stand-alone script for extracting class-name features.

Uses the bge-large-en-v1.5 sentence encoder to pre-compute features for the
set of class names and stores them in a local `.pt` file that the training
script loads directly.

Usage 1 - extract classes automatically from training annotations:
    python extract_class_features.py \
        --bge_model_path /path/to/bge-large-en-v1.5 \
        --output_path /path/to/class_features.pt \
        --data_path /path/to/train_all_1209.json

Usage 2 - specify class names manually:
    python extract_class_features.py \
        --bge_model_path /path/to/bge-large-en-v1.5 \
        --output_path /path/to/class_features.pt \
        --class_names "Normal,temporal inconsistency,..."
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ========== Helpers for collecting classes from training annotations ==========

def safe_get_bnd_class(container):
    """Read 'bnd_class' from a bnd_cot_st / bnd_cot_ed container."""
    if container is None:
        return ''
    if isinstance(container, dict):
        return container.get('bnd_class', '')
    if isinstance(container, list) and len(container) > 0:
        return container[0].get('bnd_class', '') if isinstance(container[0], dict) else ''
    return ''


def safe_get_obj_class(container):
    """Read 'bnd_sub_class' from an obj_cot container."""
    if container is None:
        return ''
    if isinstance(container, dict):
        return container.get('bnd_sub_class', '')
    if isinstance(container, list) and len(container) > 0:
        return container[0].get('bnd_sub_class', '') if isinstance(container[0], dict) else ''
    return ''


def extract_classes_from_data(data_path: str):
    """
    Collect every unique class name appearing in the training annotations.

    Rules:
      - bnd_cot_st / bnd_cot_ed: read 'bnd_class'.
      - obj_cot:                 read 'bnd_sub_class'.

    Args:
        data_path: Path to a training-split JSON file.

    Returns:
        Sorted list of unique class names.
    """
    print(f"Extracting classes from training data: {data_path}")
    
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total items in data: {len(data)}")
    
    all_classes = set()
    
    for item in data:
        annotations = item.get('annotations', [])
        if not isinstance(annotations, list):
            annotations = [annotations]
        
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            
            # bnd_cot_st -> bnd_class
            c1 = safe_get_bnd_class(ann.get('bnd_cot_st'))
            if c1:
                all_classes.add(c1)
            
            # obj_cot -> bnd_sub_class
            c2 = safe_get_obj_class(ann.get('obj_cot'))
            if c2:
                all_classes.add(c2)
            
            # bnd_cot_ed -> bnd_class
            c3 = safe_get_bnd_class(ann.get('bnd_cot_ed'))
            if c3:
                all_classes.add(c3)
    
    unique_classes = sorted(list(all_classes))
    
    # Always include the "Normal" class (real / non-forgery samples).
    if "Normal" not in unique_classes:
        unique_classes.insert(0, "Normal")  # put Normal first
        print("[INFO] Added 'Normal' class for real/non-forgery samples.")
    
    print(f"Found {len(unique_classes)} unique classes:")
    for i, cls in enumerate(unique_classes):
        print(f"  [{i+1:2d}] '{cls}'")
    
    return unique_classes


# ========== Feature extraction ==========

def extract_features(
    bge_model_path: str,
    class_names: list,
    output_path: str,
    device: str = "cuda"
):
    """
    Extract sentence-level features for `class_names` with bge-large-en-v1.5
    and save them to a `.pt` file.

    Args:
        bge_model_path: Local path to a bge-large-en-v1.5 checkpoint.
        class_names:    List of class names.
        output_path:    Where the feature `.pt` file is written.
        device:         'cuda' or 'cpu'.
    """
    print("=" * 60)
    print("BGE Class Feature Extraction Script")
    print("=" * 60)
    
    # ========== 1. Validate the model path ==========
    if not os.path.isdir(bge_model_path):
        raise RuntimeError(f"BGE model path does not exist: {bge_model_path}")
    
    if 'bge-large-en-v1.5' not in bge_model_path.lower().replace('_', '-').replace(' ', '-'):
        raise RuntimeError(
            f"Only bge-large-en-v1.5 is supported. Got: {bge_model_path}"
        )
    
    # Validate that the required files are present.
    has_config = os.path.isfile(os.path.join(bge_model_path, 'config.json'))
    has_model = (
        os.path.isfile(os.path.join(bge_model_path, 'pytorch_model.bin')) or
        os.path.isfile(os.path.join(bge_model_path, 'model.safetensors'))
    )
    
    if not has_config or not has_model:
        raise RuntimeError(f"BGE model files are incomplete: {bge_model_path}")
    
    print(f"[ok] BGE Model Path: {bge_model_path}")
    print(f"[ok] Number of classes: {len(class_names)}")
    print(f"[ok] Output Path: {output_path}")
    print(f"[ok] Device: {device}")
    print()
    
    # ========== 2. Load the model ==========
    print("Loading BGE model (bge-large-en-v1.5)...")
    tokenizer = AutoTokenizer.from_pretrained(bge_model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        bge_model_path, 
        local_files_only=True,
        torch_dtype=torch.float32
    )
    
    # Move to the requested device.
    use_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(use_device)
    model.eval()
    print(f"[ok] Model loaded on: {use_device}")
    print()
    
    # ========== 3. Extract features ==========
    print("Extracting features using CLS pooling...")
    print(f"Class names: {class_names[:5]}... (showing first 5)")
    
    # Tokenize
    encoded_input = tokenizer(
        class_names,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    encoded_input = {k: v.to(use_device) for k, v in encoded_input.items()}
    
    # Forward pass
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # CLS Pooling (recommended by the BGE authors).
    sentence_embeddings = model_output[0][:, 0]  # (num_classes, hidden_size)
    
    # L2 normalization (recommended by the BGE authors).
    class_features = F.normalize(sentence_embeddings, p=2, dim=1)
    
    # Cast to float32 and move to CPU.
    class_features = class_features.cpu().float()
    
    feat_dim = class_features.shape[-1]
    print(f"[ok] Feature shape: {class_features.shape}")
    print(f"[ok] Feature dim: {feat_dim} (expected 1024 for bge-large-en-v1.5)")
    print()
    
    # ========== 4. Save to file ==========
    print("Saving features to file...")
    
    # Create the output directory if necessary.
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Payload written to disk.
    save_data = {
        'class_features': class_features,           # (num_classes, feat_dim)
        'class_names': class_names,                 # list of class names
        'feat_dim': feat_dim,                       # feature dimension
        'num_classes': len(class_names),            # number of classes
        'model_name': 'bge-large-en-v1.5',          # model identifier
        'pooling_method': 'CLS',                    # pooling strategy
        'normalized': True,                         # whether features are L2-normalized
    }
    
    torch.save(save_data, output_path)
    
    # Sanity-check the file.
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"[ok] Features saved to: {output_path}")
    print(f"[ok] File size: {file_size:.2f} KB")
    print()
    
    # ========== 5. Verify the saved file ==========
    print("Verifying saved file...")
    loaded_data = torch.load(output_path, map_location='cpu')
    
    assert loaded_data['class_features'].shape == class_features.shape
    assert loaded_data['class_names'] == class_names
    assert loaded_data['feat_dim'] == feat_dim
    
    print(f"[ok] Verification passed!")
    print()
    
    # ========== 6. Release resources ==========
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("=" * 60)
    print("Feature extraction completed successfully!")
    print("=" * 60)
    print()
    print("Usage in training:")
    print(f"  --class_feature_path {output_path}")
    print()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract class features using BGE model")
    
    parser.add_argument(
        "--bge_model_path",
        type=str,
        default="./bge-large-en-v1.5",
        help="Path to bge-large-en-v1.5 model (local only)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="./class_features_bge.pt",
        help="Output path for the extracted features (.pt file)"
    )
    
    parser.add_argument(
        "--class_names",
        type=str,
        default=None,
        help="Comma-separated class names. If not provided and --data_path not specified, will raise error."
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training data JSON file. If provided, will extract classes from data automatically."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for feature extraction"
    )
    
    args = parser.parse_args()
    
    # Decide where the class list comes from.
    if args.class_names:
        # Option 1: manually specified class names.
        class_names = [name.strip() for name in args.class_names.split(",")]
        # Always include the Normal class.
        if "Normal" not in class_names:
            class_names.insert(0, "Normal")
            print("[INFO] Added 'Normal' class for real/non-forgery samples.")
        print(f"Using {len(class_names)} manually specified class names.")
    elif args.data_path:
        # Option 2: extract classes from a training JSON file.
        class_names = extract_classes_from_data(args.data_path)
        print(f"Extracted {len(class_names)} class names from training data.")
    else:
        raise ValueError(
            "Either --class_names or --data_path must be provided.\n"
            "  1. --class_names 'class1,class2,...' (manual)\n"
            "  2. --data_path /path/to/train.json    (auto-extract)"
        )
    
    if len(class_names) == 0:
        raise ValueError("No class names were collected. Check the data file or --class_names.")
    
    # Run feature extraction.
    extract_features(
        bge_model_path=args.bge_model_path,
        class_names=class_names,
        output_path=args.output_path,
        device=args.device
    )


if __name__ == "__main__":
    main()
