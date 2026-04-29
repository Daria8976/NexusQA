# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


import json
import os
import cv2
import base64
import numpy as np
from pathlib import Path
import sys
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

HERE = Path(__file__).resolve().parent
HVSG_ROOT = HERE.parent
OPENAI_API_KEY = "xx"
OPENAI_BASE_URL = "xx"

if len(sys.argv) > 1:
    API_KEY = sys.argv[1]
else:
    API_KEY = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

if len(sys.argv) > 2:
    BASE_URL = sys.argv[2]
else:
    BASE_URL = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)

LLM_MODEL = os.getenv("NEXUSBENCH_QA_MODEL", "gemini-3-flash-preview")

if not API_KEY or API_KEY == "xx":
    raise RuntimeError("OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before running.")

try:
    client = OpenAI(api_key=API_KEY, base_url=None if BASE_URL == "xx" else BASE_URL)
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"OpenAI client initialization failed: {e}")
    exit()

INPUT_JSON_PATH = str(HERE / "input" / "pvsg_vidor_graph_with_part_label.json")
OUTPUT_JSON_PATH_1 = str(HERE / "v1.json")
OUTPUT_JSON_PATH_2 = str(HERE / "v1_new.json")

FRAMES_BASE_PATH = str(HVSG_ROOT / "assets" / "frames")
MASKS_BASE_PATH = str(HVSG_ROOT / "assets" / "masks")
TEMP_VIS_DIR = str(HERE / "temp_vis_results")
os.makedirs(TEMP_VIS_DIR, exist_ok=True)
max_retries = 3

def generate_visualization_image(video_id, frame_num, obj1_info, obj2_info, video_item):
    frame_name = f"{frame_num:04d}.png"
    rgb_path = os.path.join(FRAMES_BASE_PATH, video_id, frame_name)
    mask_path = os.path.join(MASKS_BASE_PATH, video_id, frame_name)
    if not os.path.exists(rgb_path) or not os.path.exists(mask_path): return None
    img = cv2.imread(rgb_path)
    try:
        mask_pil = Image.open(mask_path)
        mask = np.array(mask_pil) 
    except: return None
    all_obj_ids = [int(obj['object_id']) for obj in video_item['objects']]
    targets = [obj1_info, obj2_info]
    colors = [(0, 255, 0), (0, 0, 255)]
    found_count = 0
    for i, target in enumerate(targets):
        tid = int(target["id"])
        tname = target["name"]
        try:
            pixel_val = all_obj_ids.index(tid) + 1
            coords = np.where(mask == pixel_val)
        except ValueError: coords = (np.array([]),)
        if len(coords[0]) == 0: coords = np.where(mask == tid)
        if len(coords[0]) > 0:
            y_min, x_min = np.min(coords[0]), np.min(coords[1])
            y_max, x_max = np.max(coords[0]), np.max(coords[1])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), colors[i], 3)
            cv2.putText(img, tname, (x_min, max(y_min - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
            found_count += 1
    if found_count == 0: return None
    out_path = os.path.join(TEMP_VIS_DIR, f"{video_id}_{frame_num}_vis.jpg")
    cv2.imwrite(out_path, img)
    return out_path

def call_llm_expert(client, obj1_data, obj2_data, predicate, encoded_frames):
    obj1_name = obj1_data['category']
    obj2_name = obj2_data['category']

    obj1_id = int(obj1_data['object_id'])
    obj2_id = int(obj2_data['object_id'])

    part_list_1 = [p['category'] for p in obj1_data.get('parts', [])]
    part_list_2 = [p['category'] for p in obj2_data.get('parts', [])]
    prompt_text = f"""
        Role: You are an expert Visual Relationship Analyst. Your task is to perform "Part-level Grounding" to refine a general relationship between two objects into a specific interaction between their parts.

        Inputs:
        1. Image: An image with two objects highlighted (by bbox or mask).
        2. General Relation: A relationship in the format "{{object1}}-{{relation}}-{{object2}}".
        3. Object 1 Part List: A list of specific parts belonging to {{object1}}.
        4. Object 2 Part List: A list of specific parts belonging to {{object2}}.

        Task Instructions:
        - Carefully examine the visual interaction between {obj1_name} and {obj2_name} in the provided image.
        - From the "Object 1 Part List", identify which specific part is physically or functionally initiating the relation. **Identify ALL specific parts** involved in the interaction. If you clearly see one or more parts (e.g., both "left foot" and "right foot") performing the action, you MUST list them ALL, even if some or all of them are **NOT in the provided list**.  If the interaction involves the entire object, use the name of {obj1_name}.
        - From the "Object 2 Part List", identify which specific part is receiving or being the target of the relation. **Identify ALL specific parts** involved in the interaction. If you clearly see one or more parts (e.g., both "left foot" and "right foot") performing the action, you MUST list them ALL, even if some or all of them are **NOT in the provided list**.  If the interaction involves the entire object, use the name of {obj2_name}.
        - If multiple parts from the list are simultaneously involved in the interaction (e.g., both "left hand" and "right hand" are holding an object, or both "left foot" and "right foot" are walking), list all relevant parts separated by a comma.
        - Combine these into a refined relationship string.

        Output Requirement:
        - Provide only the final refined relationship in the following format:
        {{object1 or [object1_part_x_name, ...]}}-{{relation}}-{{object2 or [object2_part_y_name, ...]}}

        **Input:**
        - General Relation: {obj1_name}-{predicate}-{obj2_name}
        - Object 1 Part List: {part_list_1}
        - Object 2 Part List: {part_list_2}

        **Final Output:**
    """
    user_content = [{"type": "text", "text": prompt_text}]
    for f_num in sorted(encoded_frames.keys()):
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frames[f_num]}"}})
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": user_content}], temperature=0.1 + (attempt * 0.1))
            answer = response.choices[0].message.content.strip()
            name_to_id_1 = {p['category']: int(p['object_id']) for p in obj1_data.get('parts', [])}
            name_to_id_1[obj1_name] = int(obj1_data['object_id'])
            name_to_id_2 = {p['category']: int(p['object_id']) for p in obj2_data.get('parts', [])}
            name_to_id_2[obj2_name] = int(obj2_data['object_id'])
            sep = f"-{predicate}-"
            if sep in answer:
                raw_parts = answer.split(sep)
                if len(raw_parts) < 2: continue
                
                subj_side = raw_parts[0].strip()
                obj_side = raw_parts[1].strip()

                def extract_ids_logic(text, name_map, parent_id, parent_name):
                    if "[" in text and "]" in text:
                        content = text[text.find("[")+1 : text.find("]")]
                        llm_names = [n.strip() for n in content.split(",") if n.strip()]
                    else:
                        llm_names = [text.strip()]
                    normalized_map = {k.lower(): v for k, v in name_map.items()}
                    normalized_parent = parent_name.lower()

                    valid_ids = []
                    has_bilateral_missing = False
                    
                    for name in llm_names:
                        curr_name_lower = name.lower()
                        if curr_name_lower in normalized_map:
                            valid_ids.append(name_map[name])
                        elif curr_name_lower == normalized_parent:
                            valid_ids.append(parent_id)
                        else:
                            if "left" in curr_name_lower or "right" in curr_name_lower:
                                has_bilateral_missing = True
                    valid_ids = list(set(valid_ids))
                    if has_bilateral_missing:
                        return [parent_id]
                    if valid_ids:
                        return valid_ids
                    return [parent_id]

                final_s_ids = extract_ids_logic(raw_parts[0], name_to_id_1, obj1_id, obj1_name)
                final_o_ids = extract_ids_logic(raw_parts[1], name_to_id_2, obj2_id, obj2_name)

                if final_s_ids and final_o_ids:
                    return [(sid, oid) for sid in final_s_ids for oid in final_o_ids]
        except: pass
    return None

if __name__ == '__main__':
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    if os.path.exists(OUTPUT_JSON_PATH_2):
        with open(OUTPUT_JSON_PATH_2, 'r', encoding='utf-8') as f:
            all_refined_results = json.load(f)
        print(f"Loaded incremental file. Currently tracking {len(all_refined_results)} videos of progress.")
    else:
        all_refined_results = {}

    video_list = full_data.get('data', [])

    for video_item in tqdm(video_list, desc="Overall Progress"):
        video_id = video_item['video_id']
        object_map = {int(obj['object_id']): obj for obj in video_item['objects']}
        
        video_final_relations = []
        if video_id not in all_refined_results:
            all_refined_results[video_id] = {"refined_relations": {}}
        num_skipped = len(all_refined_results[video_id]["refined_relations"])
        if num_skipped > 0:
            print(f"\n>>> Video: {video_id} (skipped {num_skipped} already processed relations)")

        for rel_idx, relation in enumerate(video_item.get('relations', [])):
            s_id, o_id, pred, intervals = relation
            if str(rel_idx) in all_refined_results[video_id]["refined_relations"]:
                s_name = object_map.get(s_id, {}).get('category', 'unknown')
                o_name = object_map.get(o_id, {}).get('category', 'unknown')
                print(f"      [Skip] Relation Index {rel_idx}: {s_name}({s_id}) - {pred} - {o_name}({o_id})")
                
                saved_rels = all_refined_results[video_id]["refined_relations"][str(rel_idx)]
                video_final_relations.extend(saved_rels)
                continue
            obj1_data = object_map.get(int(s_id))
            obj2_data = object_map.get(int(o_id))
            if not obj1_data or not obj2_data or (not obj1_data.get('parts') and not obj2_data.get('parts')):
                processed_rel = [relation] 
                print(f"      [NoParts] Index {rel_idx}: {obj1_data['category']} - {pred} - {obj2_data['category']} (kept as-is)")
            else:
                print(f"      [Process] Index {rel_idx}: {obj1_data['category']} - {pred} - {obj2_data['category']} (requesting MLLM)")
                refined_segments = defaultdict(list)
                for interval in intervals:
                    mid_f = (interval[0] + interval[1]) // 2
                    encoded_frames = {}
                    vis_path = generate_visualization_image(video_id, mid_f, 
                        {"id": s_id, "name": obj1_data['category']},
                        {"id": o_id, "name": obj2_data['category']}, video_item)
                    
                    if vis_path:
                        with open(vis_path, "rb") as img_f:
                            encoded_frames[mid_f] = base64.b64encode(img_f.read()).decode('utf-8')
                    result = call_llm_expert(client, obj1_data, obj2_data, pred, encoded_frames) if encoded_frames else None
                    if result: 
                        for res_pair in result:
                            refined_segments[res_pair].append(interval)
                    else: 
                        refined_segments[(s_id, o_id)].append(interval)
                
                processed_rel = []
                for (final_s, final_o), final_intervals in refined_segments.items():
                    processed_rel.append([final_s, final_o, pred, final_intervals])
            video_final_relations.extend(processed_rel)
            all_refined_results[video_id]["refined_relations"][str(rel_idx)] = processed_rel
            
            with open(OUTPUT_JSON_PATH_2, 'w', encoding='utf-8') as f2:
                json.dump(all_refined_results, f2, indent=4, ensure_ascii=False)
        video_item['relations'] = video_final_relations
        with open(OUTPUT_JSON_PATH_1, 'w', encoding='utf-8') as f1:
            json.dump(full_data, f1, indent=4, ensure_ascii=False)

    print("\nAll tasks completed.")
