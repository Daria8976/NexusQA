import json
import os
import cv2
import re
from tqdm import tqdm

def calculate_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    if (boxAArea + boxBArea - interArea) == 0: return 0
    return interArea / float(boxAArea + boxBArea - interArea)

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return None
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def time_to_seconds(time_str):
    """Convert HH:MM:SS to total seconds"""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return float(time_str)
    except:
        return 0.0

def parse_internvl_style_answer(answer_str):
    """
    Parse InternVL style text
    Extract: Time 00:00:05: [450, 250, 500, 300]
    """
    extracted_data = []
    # Match pattern: Time HH:MM:SS: [ymin, xmin, ymax, xmax]
    pattern = r"Time\s+(\d{2}:\d{2}:\d{2}):\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    matches = re.findall(pattern, answer_str)
    
    for m in matches:
        t_str, ymin, xmin, ymax, xmax = m
        timestamp = time_to_seconds(t_str)
        bbox = [int(ymin), int(xmin), int(ymax), int(xmax)]
        extracted_data.append({"timestamp": timestamp, "bbox": bbox})
    
    return extracted_data

# ================= Path Configuration =================
results_path = "path/NexusBench/results/Assembly/InternVL3_5-38B/Assembly_Process_direct.json"
gt_root = "path/NexusBench/key_frame/Assembly"
original_video_root = "path/NexusBench/video/assembly"

with open(results_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

final_ious = []
fps_cache = {}

print(f"Analyzing InternVL model results, {len(results)} items in total...")

for item in tqdm(results):
    video_id = item['video_id']
    
    # ================= Extract Prediction Information =================
    # Try reading InternVL specific fields first
    model_raw_answer = item.get('InternVL3_5-8B_answer', "")
    model_spatial_list = parse_internvl_style_answer(model_raw_answer)
    
    # If not found, try the previous structured fields
    if not model_spatial_list:
        model_spatial_list = item.get('model_spatial', [])
        
    if not model_spatial_list:
        continue

    # ================= Parse GT and Calculate =================
    target_ids = []
    spatial_mapping = item.get('evidence', {}).get('spatial', {})
    for name, id_str in spatial_mapping.items():
        match = re.search(r'\d+', id_str)
        if match:
            target_ids.append(int(match.group()))
            
    gt_json_path = os.path.join(gt_root, video_id, "annotations.json")
    if not os.path.exists(gt_json_path):
        continue
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    if video_id not in fps_cache:
        orig_v_path = os.path.join(original_video_root, f"{video_id}.mp4")
        fps_orig = get_video_fps(orig_v_path)
        fps_cache[video_id] = fps_orig
    else:
        fps_orig = fps_cache[video_id]
    fps_orig = fps_orig if fps_orig else 30.0

    max_iou_for_sample = 0.0

    for spatial_entry in model_spatial_list:
        pred_timestamp = spatial_entry.get("timestamp")
        pred_bbox_norm = spatial_entry.get("bbox") 
        
        if pred_timestamp is None or pred_bbox_norm is None:
            continue

        target_gt_frame = int(round(pred_timestamp * fps_orig))
        relevant_gt_anns = [ann for ann in gt_data if ann['image_id'] == target_gt_frame]
        
        p_ymin, p_xmin, p_ymax, p_xmax = pred_bbox_norm
        p_box = [p_xmin, p_ymin, p_xmax, p_ymax] 

        for gt_ann in relevant_gt_anns:
            if gt_ann['category_id'] in target_ids:
                g_x, g_y, g_w, g_h = gt_ann['bbox'] 
                g_box = [g_x, g_y, g_x + g_w, g_y + g_h]
                
                iou = calculate_iou(p_box, g_box)
                if iou > max_iou_for_sample:
                    max_iou_for_sample = iou
    
    final_ious.append(max_iou_for_sample)

if final_ious:
    print("\n" + "="*40)
    print(f"Evaluation completed! Average Max IoU: {sum(final_ious)/len(final_ious):.4f}")
    print("="*40)