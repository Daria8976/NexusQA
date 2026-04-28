import os
import json
import numpy as np
from collections import defaultdict

def format_pct(value):
    """Format as a percentage string"""
    return f"{value * 100:.2f}%"

def calculate_dataset_metrics(file_path): # Changed to accept a specific file path
    # 1. Container initialization
    all_samples = []
    # For categorical statistics: added acc_at_iou_count
    category_metrics = defaultdict(lambda: {"iou": [], "iop": [], "iog": [], "judge_right": 0, "acc_at_iou_count": 0})

    # 2. Read JSON
    if not os.path.exists(file_path):
        print(f"Error: Path does not exist {file_path}")
        return

    print(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_samples.extend(data)
    except Exception as e:
        print(f"Failed to read file: {e}")

    if not all_samples:
        print("No valid data samples found.")
        return

    # 3. Compute core metrics
    THRESHOLD = 0.3
    
    total_iou = []
    total_iop = []
    total_iog = []
    total_judge_right = 0
    total_iou_03_count = 0
    total_acc_at_iou_count = 0  # Added: Global acc@IoU count

    for item in all_samples:
        iou = item.get("iou", 0.0)
        iop = item.get("iop", 0.0)
        iog = item.get("iog", 0.0)
        
        # Answer judgment
        judge = str(item.get("answer_llm_judge", "")).lower()
        is_answer_right = "right" in judge
        
        cat = item.get("type", "General")

        # Aggregate global metrics
        total_iou.append(iou)
        total_iop.append(iop)
        total_iog.append(iog)
        
        if iou > THRESHOLD:
            total_iou_03_count += 1
            
        if is_answer_right:
            total_judge_right += 1

        # --- Added: Core logic for acc@IoU ---
        # Counted in acc@IoU only if the answer is correct and IoU > 0
        if is_answer_right and iou > 0:
            total_acc_at_iou_count += 1
            category_metrics[cat]["acc_at_iou_count"] += 1
        # ----------------------------

        # Aggregate category metrics
        category_metrics[cat]["iou"].append(iou)
        category_metrics[cat]["iop"].append(iop)
        category_metrics[cat]["iog"].append(iog)
        if is_answer_right:
            category_metrics[cat]["judge_right"] += 1

    # 4. Calculate final metrics
    num_total = len(all_samples)
    mIoU = np.mean(total_iou)
    mIoP = np.mean(total_iop)
    mIoG = np.mean(total_iog)
    overall_iou_03 = total_iou_03_count / num_total
    qa_accuracy = total_judge_right / num_total
    acc_at_iou = total_acc_at_iou_count / num_total # Added

    # 5. Print results report
    print("="*85)
    print(f"{'Overall Dataset Evaluation Summary':^85}")
    print("="*85)
    print(f"{'Total Samples':<25}: {num_total}")
    print(f"{'QA Accuracy (LLM Judge)':<25}: {format_pct(qa_accuracy)}")
    print(f"{'mIoU':<25}: {format_pct(mIoU)}")
    print(f"{'acc@IoU (τ=0.3)':<25}: {format_pct(acc_at_iou)} (Answer Right & IoU > 0.3)")
    print(f"{'IoU@0.3 (Recall)':<25}: {format_pct(overall_iou_03)}")
    print(f"{'mIoP':<25}: {format_pct(mIoP)}")
    print(f"{'mIoG':<25}: {format_pct(mIoG)}")
    print("="*85)

    # 6. Print category breakdown
    print(f"\n{'Category-wise Breakdown':^85}")
    print("-" * 85)
    # Added Acc@IoU to header
    header = f"{'Question Type':<25} | {'Cnt':<5} | {'QA Acc':<8} | {'mIoU':<8} | {'Acc@IoU':<8}"
    print(header)
    print("-" * 85)

    for cat, val in category_metrics.items():
        cnt = len(val["iou"])
        c_qa = val["judge_right"] / cnt
        c_iou = np.mean(val["iou"])
        c_acc_iou = val["acc_at_iou_count"] / cnt
        
        print(f"{cat[:25]:<25} | {cnt:<5} | {format_pct(c_qa):<8} | {format_pct(c_iou):<8} | {format_pct(c_acc_iou):<8}")
    print("-" * 85)

if __name__ == "__main__":
    target_file = "path/NexusBench/results/Assembly/InternVL3_5-38B/Assembly_Process_direct.json"
    calculate_dataset_metrics(target_file)