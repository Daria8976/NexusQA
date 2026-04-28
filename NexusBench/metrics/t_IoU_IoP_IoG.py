import os
import json
import re
import ast
import numpy as np
from pathlib import Path

# ==========================================
# 1. Core Mathematical Calculation Logic (Modified to Max-IoU Matching)
# ==========================================

def calculate_pair_metrics(pred_seg, gt_seg):
    """
    Calculate metrics between a [one-to-one] segment pair [start, end]
    """
    p_s, p_e = pred_seg
    g_s, g_e = gt_seg

    # Calculate intersection
    inter_s = max(p_s, g_s)
    inter_e = min(p_e, g_e)
    intersection = max(0.0, inter_e - inter_s)

    # Respective lengths
    len_p = p_e - p_s
    len_g = g_e - g_s

    # Union
    union = len_p + len_g - intersection

    epsilon = 1e-8
    iou = intersection / (union + epsilon)
    iop = intersection / (len_p + epsilon)  # Precision (IoP)
    iog = intersection / (len_g + epsilon)  # Recall (IoG)
    
    return iou, iop, iog

def calculate_metrics(pred_intervals, gt_intervals):
    """
    Calculation logic: Iterate through each predicted segment, find the match 
    with the maximum IoU in the GT, and finally take the average.
    """
    # Filter invalid intervals (start >= end)
    valid_preds = [p for p in pred_intervals if len(p) == 2 and p[0] < p[1]]
    valid_gts = [g for g in gt_intervals if len(g) == 2 and g[0] < g[1]]

    if not valid_preds or not valid_gts:
        return 0.0, 0.0, 0.0

    item_ious = []
    item_iops = []
    item_iogs = []

    print("gt:", gt_intervals, "pred:", pred_intervals)
    print()

    # Find the best matching GT segment G for each predicted segment P
    for p in valid_preds:
        max_iou = -1.0
        best_match_metrics = (0.0, 0.0, 0.0) # (iou, iop, iog)
        
        for g in valid_gts:
            iou, iop, iog = calculate_pair_metrics(p, g)
            if iou > max_iou:
                max_iou = iou
                best_match_metrics = (iou, iop, iog)
        
        item_ious.append(best_match_metrics[0])
        item_iops.append(best_match_metrics[1])
        item_iogs.append(best_match_metrics[2])

    # Average the best match scores of all predicted segments for this sample
    avg_iou = float(sum(item_ious)/max(len(pred_intervals), len(gt_intervals), 1))
    avg_iop = float(sum(item_iops)/max(len(pred_intervals), len(gt_intervals), 1))
    avg_iog = float(sum(item_iogs)/max(len(pred_intervals), len(gt_intervals), 1))

    return round(float(avg_iou), 4), round(float(avg_iop), 4), round(float(avg_iog), 4)

# ==========================================
# 2. Data Parsing and Conversion (Extraction logic remains unchanged)
# ==========================================

def time_str_to_seconds(time_str):
    """Convert 'MM:SS.ms' or 'SS.ms' to total seconds"""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:  # MM:SS.ms
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1: # SS.ms
            return float(parts[0])
        return 0.0
    except Exception:
        return 0.0

def extract_variables(model_answer, model_name):
    """Robustly extract answers and temporal evidence from model outputs (keeping original logic)"""
    answer = model_answer.strip()
    temporal_evidence = []
    spatial_evidence = {}

    try:
        temp_block_match = re.search(r"Temporal Evidence:\s*(\[\[.*?\]\])", model_answer, re.DOTALL)
        if temp_block_match:
            temp_str = temp_block_match.group(1)
            pairs = re.findall(r"\[\s*['\"]?([0-9:.]+)['\"]?\s*,\s*['\"]?([0-9:.]+)['\"]?\s*\]", temp_str)
            for p in pairs:
                start_sec = time_str_to_seconds(p[0])
                end_sec = time_str_to_seconds(p[1])
                temporal_evidence.append([start_sec, end_sec])

        if "Temporal Evidence:" in model_answer:
            answer = model_answer.split("Temporal Evidence:")[0].strip()
            if answer.startswith("Answer:"):
                answer = answer[7:].strip()

        spat_match = re.search(r"Spatial Evidence:\s*(\{.*?\})", model_answer, re.DOTALL)
        if spat_match:
            try:
                spatial_evidence = json.loads(spat_match.group(1).replace("'", '"'))
            except:
                pass
    except Exception as e:
        print(f"Parsing exception: {e}")

    return answer, temporal_evidence, spatial_evidence

# ==========================================
# 3. Evaluation Wrapper Function (Updated to call new matching calculation logic)
# ==========================================

def eval_temporal(pred_temporal, gt_temporal_dict):
    if not gt_temporal_dict:
        return {"iou": 0.0, "iop": 0.0, "iog": 0.0}

    # 1. Convert GT format
    gt_intervals = []
    for val in gt_temporal_dict.values():
        if isinstance(val, list) and len(val) == 2:
            gt_intervals.append([time_str_to_seconds(val[0]), time_str_to_seconds(val[1])])

    # 2. Normalize Pred format (ensure it's a list of lists)
    if not pred_temporal:
        return {"iou": 0.0, "iop": 0.0, "iog": 0.0}
    
    if len(pred_temporal) > 0 and not isinstance(pred_temporal[0], list):
        preds = [pred_temporal]
    else:
        preds = pred_temporal

    # 3. Calculate (using the new Max-IoU matching function)
    iou, iop, iog = calculate_metrics(preds, gt_intervals)
    return {"iou": iou, "iop": iop, "iog": iog}

# ==========================================
# 4. Main Program
# ==========================================

def process_json_file(json_path):
    print(f"\nProcessing file: {os.path.basename(json_path)}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model_name = os.path.basename(json_path)

    for item in data:
        # 1. Extract variables (using the original extraction function)
        raw_output = item.get("model_answer", "")
        ans, temp, spat = extract_variables(raw_output, model_name)
        
        # 2. Obtain GT data
        gt_temp_data = item.get("evidence", {}).get("temporal", {})
        
        # 3. Evaluation calculation
        if not temp: 
            temp_metrics = {"iou": 0.0, "iop": 0.0, "iog": 0.0}
        else:
            temp_metrics = eval_temporal(temp, gt_temp_data)
        
        # 4. Write back results
        item["iou"] = temp_metrics["iou"]
        item["iop"] = temp_metrics["iop"]
        item["iog"] = temp_metrics["iog"]

        # print(f"   ID: {item.get('sample_id')} | IoU: {item['iou']} | IoP: {item['iop']} | IoG: {item['iog']}")

    # 5. Print average IoU for the entire file
    all_ious = [i["iou"] for i in data if "iou" in i]
    if all_ious:
        print(f"Dataset average IoU (Max-IoU Match): {np.mean(all_ious):.4f}")

    # Save results
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    json_files = ["path/NexusBench/results/Assembly/InternVL3_5-38B/Assembly_Process_direct.json"]
    for file_path in json_files:
        process_json_file(str(file_path))