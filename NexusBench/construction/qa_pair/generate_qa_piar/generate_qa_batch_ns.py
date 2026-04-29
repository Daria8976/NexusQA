import os
import json
import re
import time
import sys
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONSTRUCTION_ROOT = HERE.parent.parent
input_json_path = str(HERE / "v1.json")
output_json_folder_path = str(HERE / "output_json")
system_prompt_path = str(HERE / "system_prompt.txt")
debug_message_folder = str(CONSTRUCTION_ROOT / "qa_pair" / "temp_message")

OPENAI_API_KEY = "xx"
OPENAI_BASE_URL = "xx"

if len(sys.argv) > 1:
    ak_idea = sys.argv[1]
else:
    ak_idea = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

if len(sys.argv) > 2:
    base_url = sys.argv[2]
else:
    base_url = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)

model_name = os.getenv("NEXUSBENCH_QA_MODEL", "gemini-3-flash-preview")
client = OpenAI(
    api_key=ak_idea,
    base_url=None if base_url == "xx" else base_url,
)

def get_entity_mapping(video_item):
    entity_map = {}
    for obj in video_item.get("objects", []):
        obj_id_str = str(obj["object_id"])
        obj_category = obj["category"]
        entity_map[obj_id_str] = obj_category

        if "parts" in obj:
            for part in obj["parts"]:
                parent_info = f"{obj_category} {obj_id_str}"
                entity_map[str(part["object_id"])] = f"{part['category']} (part of {parent_info})"
    return entity_map

def format_relations_logic(relations_list, entity_map):
    formatted = []
    for rel in relations_list:
        sub_id, obj_id, action, timespans = str(rel[0]), str(rel[1]), rel[2], rel[3]
        formatted.append({
            "subject_id": sub_id,
            "subject_name": entity_map.get(sub_id, "Unknown"),
            "object_id": obj_id,
            "object_name": entity_map.get(obj_id, "Unknown"),
            "action": action,
            "timespan": timespans
        })
    
    ##### SORT BY START TIMESTAMP #####
    formatted.sort(key=lambda x: x["timespan"][0][0] if x["timespan"] else 0)
    ##### END SORT BY START TIMESTAMP #####

    return formatted

def robust_json_cleaner(text):
    if not text: return None, False
    text = text.strip()
    clean_text = re.sub(r'```json\s*|```\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    is_truncated = not clean_text.strip().endswith(']')
    
    start = clean_text.find('[')
    end = clean_text.rfind(']')
    if start != -1 and end != -1 and end > start:
        json_str = clean_text[start:end+1]
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        return json_str, is_truncated
    return None, is_truncated

def parse_time_range(time_str):
    try:
        start, end = map(int, time_str.split('-'))
        return start, end
    except:
        return 0, 0

def get_detailed_objects_info(video_item, involved_ids):
    detailed_info = []
    
    for obj in video_item.get("objects", []):
        obj_id_str = str(obj["object_id"])
        parts = obj.get("parts", [])
        part_ids = {str(p["object_id"]) for p in parts}
        if obj_id_str in involved_ids or not involved_ids.isdisjoint(part_ids):
            info = {
                "object_id": obj["object_id"],
                "category": obj["category"],
                "parts": [
                    {"part_id": p["object_id"], "part_name": p["category"]} 
                    for p in parts
                ]
            }
            detailed_info.append(info)
            
    return detailed_info

def process_video_item(video_item, system_prompt):
    video_id = video_item["video_id"]
    output_path = os.path.join(output_json_folder_path, f"{video_id}.json")
    entity_map = get_entity_mapping(video_item)
    all_raw_relations = video_item.get("relations", [])
    if not all_raw_relations: return
    full_formatted_relations = format_relations_logic(all_raw_relations, entity_map)
    captions = video_item.get("captions", [])
    global_context_message = {
        "role": "user",
        "content": (
            f"--- GLOBAL VIDEO SCENE GRAPH CONTEXT ---\n"
            # f"Video ID: {video_id}\n"
            f"Global Event Captions: {json.dumps(video_item.get('captions', []), indent=2)}\n"
            f"Global Spatial-Temporal Relations (Timeline):\n{json.dumps(full_formatted_relations, indent=2)}\n"
            f"Visual Summary: {video_item.get('summary', '')}\n"
            "-----------------------------------"
        )
    }

    total_qa_pairs = []
    # processed_end_times = set()
    processed_captions = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    total_qa_pairs = json.loads(content)
            print(f"Resuming {video_id}: Found {len(total_qa_pairs)} existing QA pairs.")
        except Exception as e:
            print(f"Warning: Could not parse existing JSON for {video_id}: {e}")

    def is_caption_already_done(cap_start, cap_end, existing_qas):
        for qa in existing_qas:
            timestamps = re.findall(r'<(\d+)-(\d+)>', qa.get('answer', ''))
            for start_f, end_f in timestamps:
                if cap_start <= int(start_f) <= cap_end:
                    return True
        return False
    # -----------------------
    os.makedirs(debug_message_folder, exist_ok=True)
    debug_file_path = os.path.join(debug_message_folder, f"{video_id}.json")
    video_debug_data = []
    # for cap_idx, caption in enumerate(captions[:2]): 
    for cap_idx, caption in enumerate(captions):
        cap_time_str = caption.get("time", "0-0")
        cap_start, cap_end = parse_time_range(cap_time_str)
        
        if is_caption_already_done(cap_start, cap_end, total_qa_pairs):
            print(f"Skipping Caption {cap_idx} ({cap_time_str}) - Already Done.")
            continue
        all_cap_relations = []
        for rel in full_formatted_relations:
            valid_ts = [ts for ts in rel['timespan'] if cap_start <= ts[0] <= cap_end]
            if valid_ts:
                chunk_rel = rel.copy()
                chunk_rel['timespan'] = valid_ts
                all_cap_relations.append(chunk_rel)

        rel_count = len(all_cap_relations)
        if rel_count == 0:
            print(f"Caption {cap_idx} ({cap_time_str}): No matching relations.")
            continue



        print(f"\n>>> Processing Caption {cap_idx} ({cap_time_str})")
        print(f"    Total relations found in this window: {rel_count}")
        sub_idx = 0
        current_chunk_size = rel_count 

        while sub_idx < rel_count:
            current_chunk_size = min(current_chunk_size, rel_count - sub_idx)
            chunk = all_cap_relations[sub_idx : sub_idx + current_chunk_size]
            involved_ids_in_chunk = set()
            for r in chunk:
                involved_ids_in_chunk.add(str(r["subject_id"]))
                involved_ids_in_chunk.add(str(r["object_id"]))
            detailed_objects_meta = get_detailed_objects_info(video_item, involved_ids_in_chunk)

            print(f"    -> Attempting Rel Indices {sub_idx} to {sub_idx + current_chunk_size - 1} (Size: {current_chunk_size})")

            current_task_message = {
                "role": "user",
                "content": (
                    f"--- TASK SEGMENT ---\n"
                    f"Video Time: {cap_time_str}\n"
                    f"Event Description: {caption.get('description', '')}\n"
                    f"Involved Objects & Their Parts:\n{json.dumps(detailed_objects_meta, indent=2)}\n"
                    f"Current Relations:\n{json.dumps(chunk, indent=2)}\n\n"
                    "Generate VideoQA JSON array. Focus on these relations."
                )
            }

            messages = [{"role": "system", "content": system_prompt}, global_context_message, current_task_message]
            video_debug_data.append({"cap": cap_idx, "range": [sub_idx, sub_idx + current_chunk_size - 1], "msgs": messages})
            with open(debug_file_path, 'w', encoding='utf-8') as f:
                json.dump(video_debug_data, f, indent=4, ensure_ascii=False)
            print(f"      -> Messages saved to {debug_file_path}. Moving to next caption.")
            # break 

            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3
                )
                raw_resp = completion.choices[0].message.content
                clean_json_str, is_truncated = robust_json_cleaner(raw_resp)

                if is_truncated:
                    print(f"    [!] Truncated! Shrinking size from {current_chunk_size} to {current_chunk_size - 1}")
                    current_chunk_size = max(1, current_chunk_size - 1)
                    continue

                if clean_json_str:
                    try:
                        batch_qa = json.loads(clean_json_str)
                        total_qa_pairs.extend(batch_qa)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(total_qa_pairs, f, indent=4, ensure_ascii=False)
                        sub_idx += current_chunk_size
                        current_chunk_size = rel_count - sub_idx
                    except json.JSONDecodeError:
                        print(f"    [!] JSON Error. Shrinking size.")
                        current_chunk_size = max(1, current_chunk_size - 1)
                else:
                    print(f"    [!] No JSON found. Shrinking size.")
                    current_chunk_size = max(1, current_chunk_size - 1)

            except Exception as e:
                print(f"    [!] API Error: {e}")
                time.sleep(5)
                current_chunk_size = max(1, current_chunk_size - 1)

    print(f"Finished {video_id}. Total QA Count: {len(total_qa_pairs)}")


def main():
    if not ak_idea or ak_idea == "xx":
        raise RuntimeError("OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before running.")
    os.makedirs(output_json_folder_path, exist_ok=True)
    with open(system_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    with open(input_json_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    # for item in tqdm(full_data["data"][:5]): 
    for item in tqdm(full_data["data"]):
        process_video_item(item, system_prompt)

if __name__ == "__main__":
    main()
