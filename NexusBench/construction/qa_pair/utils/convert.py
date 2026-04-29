import os
import json
import cv2
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONSTRUCTION_ROOT = HERE.parent.parent
INPUT_JSON_FOLDER = str(CONSTRUCTION_ROOT / "qa_pair" / "generate_qa_piar" / "output_json")
VIDEO_ROOT = str(CONSTRUCTION_ROOT / "assets" / "videos")
OUTPUT_FOLDER = str(CONSTRUCTION_ROOT / "qa_pair" / "nature_v2")
# ==========================================

def get_video_fps(video_id):
    possible_paths = [
        os.path.join(VIDEO_ROOT, f"{video_id}.mp4"),
        os.path.join(VIDEO_ROOT, f"{video_id}.MP4"),
        os.path.join(VIDEO_ROOT, video_id), 
    ]
    for path in possible_paths:
        if os.path.exists(path):
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0: return fps
    return None

def frame_to_time_format(frame_idx, fps):
    try:
        f = float(str(frame_idx).split(':')[-1])
        total_seconds = (f - 1) / fps
        if total_seconds < 0: total_seconds = 0
        
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:05.2f}"
    except:
        return frame_idx

def process_item(item):
    video_id = item.get("video_id")
    if not video_id: return item

    fps = get_video_fps(video_id)
    if fps is None:
        print(f"  [Skip] Cannot find video or invalid FPS: {video_id}")
        return item
    ref_text = item.get("answer_evidence", "") or item.get("answer_complete", "")
    frame_pairs = re.findall(r"<(\d+)-(\d+)>", ref_text)
    if "evidence" in item and "temporal" in item["evidence"]:
        temporal = item["evidence"]["temporal"]
        for i, (f_start, f_end) in enumerate(frame_pairs):
            t_key = f"<T{i+1}>"
            t_start = frame_to_time_format(f_start, fps)
            t_end = frame_to_time_format(f_end, fps)
            temporal[t_key] = [t_start, t_end]
    spatial_matches = re.findall(r"\[([\w\s]+)\s(\d+)\]", ref_text)
    obj_id_map = {name.strip(): id_str for name, id_str in spatial_matches}

    if "evidence" in item and "spatial" in item["evidence"]:
        spatial = item["evidence"]["spatial"]
        for obj_name in spatial.keys():
            if obj_name in obj_id_map:
                spatial[obj_name] = obj_id_map[obj_name]
    def time_replacer(match):
        s_f = match.group(1)
        e_f = match.group(2)
        return f"<{frame_to_time_format(s_f, fps)}-{frame_to_time_format(e_f, fps)}>"

    if "answer_complete" in item:
        item["answer_complete"] = re.sub(r"<(\d+)-(\d+)>", time_replacer, item["answer_complete"])
    
    if "answer_evidence" in item:
        item["answer_evidence"] = re.sub(r"<(\d+)-(\d+)>", time_replacer, item["answer_evidence"])

    return item

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    json_files = [f for f in os.listdir(INPUT_JSON_FOLDER) if f.endswith('.json')]
    
    for json_name in json_files:
        input_path = os.path.join(INPUT_JSON_FOLDER, json_name)
        output_path = os.path.join(OUTPUT_FOLDER, json_name)
        
        print(f"Processing: {json_name}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            new_data = [process_item(item) for item in data]
        else:
            new_data = process_item(data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
        
        print(f"  Saved to: {output_path}")

if __name__ == "__main__":
    main()
