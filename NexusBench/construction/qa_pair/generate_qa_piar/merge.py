import os
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONSTRUCTION_ROOT = HERE.parent.parent
INPUT_DIR = str(HERE / "output_json")
OUTPUT_FILE = str(CONSTRUCTION_ROOT / "qa_pair" / "nature_v2.json")
VIDEO_URL_PREFIX = "/NexusBench/video/natural/"
# ===========================================

def format_time(seconds_str):
    try:
        s = float(seconds_str)
        minutes = int(s // 60)
        seconds = s % 60
        return f"{minutes:02d}:{seconds:05.2f}"
    except ValueError:
        return "00:00.00"

def parse_evidence(answer_text):
    temporal_dict = {}
    spatial_dict = {}
    evidence_text = ""
    time_pattern = r"<(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)>"
    matches = re.finditer(time_pattern, answer_text)
    
    t_count = 1
    for match in matches:
        start_str = match.group(1)
        end_str = match.group(2)
        key = f"<T{t_count}>"
        temporal_dict[key] = [format_time(start_str), format_time(end_str)]
        t_count += 1
    evidence_text = answer_text
    spatial_pattern = r"\[([^\]\d]+)\s*\d*\]"
    spatial_matches = re.findall(spatial_pattern, answer_text)
    
    for obj in spatial_matches:
        obj_name = obj.strip()
        if obj_name:
            spatial_dict[obj_name] = ""
            
    return temporal_dict, spatial_dict, evidence_text

def split_answer(answer_text):
    if "<" in answer_text:
        parts = answer_text.split("<", 1)
        summary = parts[0].strip()
        evidence = "<" + parts[1]
        return summary, evidence
    else:
        return answer_text.strip(), ""

def process_files():
    output_data = []
    sample_id = 1
    json_files = sorted(Path(INPUT_DIR).glob("*.json"))
    
    if not json_files:
        print(f"No .json found in {INPUT_DIR}")
        return

    print(f"Found {len(json_files)} files, start processing...")

    for json_path in json_files:
        video_id = json_path.stem
        video_url = f"{VIDEO_URL_PREFIX}{video_id}.mp4"
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                qa_list = json.load(f)
        except Exception as e:
            print(f"Failed to read file {json_path}: {e}")
            continue
            
        if not isinstance(qa_list, list):
            print(f"File format error (expected list), skipping: {json_path}")
            continue
        for qa_item in qa_list:
            question = qa_item.get("question", "")
            answer = qa_item.get("answer", "")
            answer_summary, answer_evidence = split_answer(answer)
            temporal, spatial, _ = parse_evidence(answer)
            item = {
                "sample_id": sample_id,
                "video_id": video_id,
                "video_url": video_url,
                "question": question,
                "answer_complete": answer,
                "answer_summary": answer_summary,
                "answer_evidence": answer_evidence,
                "evidence": {
                    "temporal": temporal,
                    "spatial": spatial
                },
                "type": "valid"
            }
            
            output_data.append(item)
            sample_id += 1
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"Processing complete. Generated {len(output_data)} items, saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Failed to save file: {e}")

if __name__ == "__main__":
    process_files()
