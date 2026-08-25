import json
import argparse
import os
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate(data):
    y_true = []
    y_pred = []
    
    # Store results by domain
    domain_results = defaultdict(lambda: {'true': [], 'pred': []})
    # Store results by data_domain (if available) -> tool_domain
    # Structure: data_domain -> tool_domain -> {'true': [], 'pred': []}
    nested_results = defaultdict(lambda: defaultdict(lambda: {'true': [], 'pred': []}))
    
    total_segments = 0
    valid_predictions = 0
    
    for video in data:
        annotations = video.get('annotations', [])
        # Video level data domain
        video_data_domain = video.get('data_domain', 'unknown_data_domain')
        
        for ann in annotations:
            total_segments += 1
            
            gt = ann.get('segment_label')
            pred = ann.get('predict_label')
            
            # Use segment level domains if available, otherwise fallback or unknown
            tool_domain = ann.get('tool_domain', 'unknown_tool_domain')
            data_domain = ann.get('data_domain', video_data_domain)
            
            # Normalize labels
            if gt: gt = gt.lower()
            if pred: pred = pred.lower()
            
            # Robust mapping for BusterXpp output which might be 'A', 'B' or 'real', 'fake'
            # Assuming standard output is "real" or "fake" based on the provided script
            
            # Skip if prediction is missing
            if not pred:
                continue
                
            # If gt is missing, we can't evaluate
            if not gt:
                continue

            y_true.append(gt)
            y_pred.append(pred)
            
            domain_results[tool_domain]['true'].append(gt)
            domain_results[tool_domain]['pred'].append(pred)
            
            nested_results[data_domain][tool_domain]['true'].append(gt)
            nested_results[data_domain][tool_domain]['pred'].append(pred)
            
            valid_predictions += 1
            
    print(f"Total segments found: {total_segments}")
    print(f"Valid predictions used for eval: {valid_predictions}")
    
    if valid_predictions == 0:
        print("No valid predictions found to evaluate.")
        return

    print("\n" + "="*60)
    print("OVERALL SEGMENT-LEVEL RESULTS")
    print("="*60)
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    
    # Per Tool Domain Evaluation
    if len(domain_results) > 0:
        print("\n" + "="*60)
        print("RESULTS BY TOOL DOMAIN")
        print("="*60)
        
        sorted_domains = sorted(domain_results.keys())
        
        print(f"{'Tool Domain':<25} | {'Count':<8} | {'Accuracy':<10} | {'F1-Macro':<10}")
        print("-" * 65)
        
        for domain in sorted_domains:
            d_true = domain_results[domain]['true']
            d_pred = domain_results[domain]['pred']
            count = len(d_true)
            
            acc = accuracy_score(d_true, d_pred)
            rep = classification_report(d_true, d_pred, output_dict=True, zero_division=0)
            f1 = rep['macro avg']['f1-score']
            
            print(f"{domain:<25} | {count:<8} | {acc:.4f}     | {f1:.4f}")

    # Nested Evaluation (Data Domain -> Tool Domain)
    if len(nested_results) > 0:
        print("\n" + "="*60)
        print("RESULTS BY DATA DOMAIN -> TOOL DOMAIN")
        print("="*60)
        
        for d_domain in sorted(nested_results.keys()):
            print(f"\n[ Data Domain: {d_domain} ]")
            print(f"{'  Tool Domain':<25} | {'Count':<8} | {'Accuracy':<10}")
            print("-" * 55)
            
            t_domains = nested_results[d_domain]
            for t_domain in sorted(t_domains.keys()):
                d_true = t_domains[t_domain]['true']
                d_pred = t_domains[t_domain]['pred']
                count = len(d_true)
                acc = accuracy_score(d_true, d_pred)
                print(f"  {t_domain:<23} | {count:<8} | {acc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", type=str, required=True, help="Path to GT JSON file")
    parser.add_argument("--infer_file", type=str, required=True, help="Path to Inference JSON file")
    args = parser.parse_args()
    
    gt_path = args.gt_file
    infer_path = args.infer_file
    
    if not os.path.exists(gt_path):
        print(f"Error: GT file not found - {gt_path}")
        return
    if not os.path.exists(infer_path):
        print(f"Error: Infer file not found - {infer_path}")
        return
        
    print(f"Loading GT from: {gt_path}")
    print(f"Loading Infer from: {infer_path}")
    
    # Load both files
    try:
        gt_data = load_data(gt_path)
        infer_data = load_data(infer_path)
        
        # In evaluate_short for short videos, we typically iterate over the infer_data or gt_data.
        # However, the previous code structure for evaluate_short assumed one single file 'data' 
        # that contained BOTH 'segment_label' (GT) and 'predict_label' (Pred).
        # Typically short video evaluation output merges the prediction into the original json.
        # If infer_file is a MERGED file containing predictions, we just need to load that.
        # BUT, to be safe and consistent with typical evaluation pipelines (and 'evaluate_long.py' changes), 
        # we should map predictions from infer_data to gt_data based on video_path/id, OR assume infer_data contains everything.
        
        # Let's check the logic of 'evaluate(data)'. 
        # It iterates 'data' and looks for 'segment_label' AND 'predict_label'.
        # If 'infer_file' is the output of the inference script, it usually contains the original annotations + 'predict_label'.
        # So passing 'infer_data' to 'evaluate' should work IF the inference script preserved the GT structure.
        
        # However, if the user requested separate files (gt_js and infer_js), it implies they might be separate.
        # If they are separate, we need to merge them.
        
        # Strategy: 
        # 1. Index predictions by video path (or some ID).
        # 2. Iterate GT data, find matching prediction, inject 'predict_label' into annotations.
        # 3. Pass the merged structure to 'evaluate'.
        
        # Indexing infer_data
        # Typically short video json structure is a list of video items.
        
        pred_map = {} # video_path -> video_item
        for item in infer_data:
            v_path = item.get('video_path', '')
            if v_path:
                pred_map[v_path] = item
        
        merged_data = []
        for gt_item in gt_data:
            v_path = gt_item.get('video_path', '')
            pred_item = pred_map.get(v_path)
            
            # Deep copy to avoid mutating original GT
            import copy
            new_item = copy.deepcopy(gt_item)
            
            if pred_item:
                # Assuming simple structure where we have annotations list.
                # Use strict alignment by index? Or just assume same order?
                # Usually short video structure is 1 video -> 1 segment? Or multiple?
                # The evaluate code iterates annotations.
                
                gt_anns = new_item.get('annotations', [])
                pred_anns = pred_item.get('annotations', [])
                
                # If prediction overwrites annotations with 'predict_label', we can just take pred_anns?
                # But we need to ensure GT 'segment_label' is there.
                
                # Safer: Match annotations.
                # If there's 1 annotation per video (common in some datasets), simple.
                # If multiple, need to match.
                
                if len(gt_anns) == len(pred_anns):
                     for i, ann in enumerate(gt_anns):
                         p_ann = pred_anns[i]
                         if 'predict_label' in p_ann:
                             ann['predict_label'] = p_ann['predict_label']
            
            merged_data.append(new_item)
            
        evaluate(merged_data)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
