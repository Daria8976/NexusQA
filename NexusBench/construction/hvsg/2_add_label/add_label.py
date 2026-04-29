import os
import json
import re
import base64
import sys
import time
from pathlib import Path
from openai import OpenAI, APIConnectionError, APITimeoutError
from tqdm import tqdm
from collections import defaultdict

HERE = Path(__file__).resolve().parent
HVSG_ROOT = HERE.parent

# --- Configuration ---
IMG_DIR = str(HVSG_ROOT / "assets" / "test_frame_masks" / "0004_11566980553")
JSON_METADATA_PATH = str(HVSG_ROOT / "input" / "0004_11566980553.json")
SYSTEM_PROMPT_PATH = str(HERE / "system_prompt_add_label.txt")
OUTPUT_JSON_PATH = str(HERE / "predict_label_sample.json")

# --- API setup (keep placeholders in file for anonymous release) ---
OPENAI_API_KEY = "xx"
OPENAI_BASE_URL = "xx"

DEFAULT_AK = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

if len(sys.argv) > 1:
    AK_IDEA = sys.argv[1]
else:
    AK_IDEA = DEFAULT_AK

if len(sys.argv) > 2:
    BASE_URL = sys.argv[2]
else:
    BASE_URL = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)

MODEL_NAME = os.getenv("NEXUSBENCH_QA_MODEL", "gemini-3-flash-preview")
client = OpenAI(
    api_key=AK_IDEA,
    base_url=None if BASE_URL == "xx" else BASE_URL,
    timeout=60.0,
)

# --- Helper Functions ---

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_category_map(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {str(obj["object_id"]): obj["category"] for obj in data.get("objects", [])}

def parse_filename(filename):
    match = re.search(r'frame(\d+)_orig(\d+)', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def clean_json_response(response_text):
    if not response_text: return ""
    text = re.sub(r'^```json\s*', '', response_text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()

def load_existing_results(path):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed_names = {item['image_name'] for item in data}
                return data, processed_names
        except Exception as e:
            print(f"Error loading existing JSON: {e}")
    return [], set()

# --- Main Logic ---

def main():
    if not AK_IDEA or AK_IDEA == "xx":
        raise RuntimeError("OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before running.")
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        print(f"Error: Prompt not found at {SYSTEM_PROMPT_PATH}")
        return

    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        system_prompt_template = f.read()
    
    category_map = get_category_map(JSON_METADATA_PATH)
    all_results, processed_names = load_existing_results(OUTPUT_JSON_PATH)
    if processed_names:
        print(f"Found {len(processed_names)} already processed images. Skipping them.")
    groups = defaultdict(list)
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for filename in img_files:
        f_id, o_id = parse_filename(filename)
        if f_id and o_id:
            groups[(f_id, o_id)].append(filename)
    for (f_id, o_id), filenames in tqdm(groups.items(), desc="Processing Objects"):
        if all(fname in processed_names for fname in filenames):
            continue

        category = category_map.get(o_id, "unknown object")
        current_img_mapping = {}
        user_content = [{"type": "text", "text": f"Object Category: '{category}'. These are part masks of one instance. Return JSON."}]
        
        for i, fname in enumerate(filenames):
            idx_key = f"image_{i+1}"
            current_img_mapping[idx_key] = fname 
            img_path = os.path.join(IMG_DIR, fname)
            base64_img = encode_image(img_path)
            user_content.append({"type": "text", "text": f"{idx_key}:"})
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
        max_retries = 3
        success = False
        for attempt in range(max_retries):
            try:
                current_system_prompt = system_prompt_template.replace("{category_label}", category)
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": current_system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )

                response_text = completion.choices[0].message.content
                cleaned_text = clean_json_response(response_text)
                labels_json = json.loads(cleaned_text)

                new_batch = []
                for idx_key, label in labels_json.items():
                    lookup_key = idx_key.lower() 
                    if lookup_key in current_img_mapping:
                        real_filename = current_img_mapping[lookup_key]
                        if real_filename not in processed_names:
                            new_data = {
                                "image_name": real_filename,
                                "image_path": os.path.join(IMG_DIR, real_filename),
                                "object_category": category,
                                "part_level_label": str(label).lower()
                            }
                            new_batch.append(new_data)
                            processed_names.add(real_filename)
                
                all_results.extend(new_batch)
                with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)
                
                success = True
                break 

            except (APIConnectionError, APITimeoutError) as e:
                print(f"\n[Attempt {attempt+1}] Connection error for Obj {o_id}: {e}. Retrying in 5s...")
                time.sleep(5)
            except Exception as e:
                print(f"\nUnexpected error for Obj {o_id} Frame {f_id}: {e}")
                break

    print(f"\nProcessing complete. Total results: {len(all_results)}. Saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
