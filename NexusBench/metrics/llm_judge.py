import os
import json
import time
from openai import OpenAI
from tqdm import tqdm

# --- Configuration Parameters ---
RESULT_FOLDER_PATH = "path/NexusBench/results"
SCENARIO_LIST = ["Assembly", "Function", "Natural"]
SYSTEM_PROMPT_PATH = "system_prompt.txt"

# Specified Model Name List
MODEL_NAME_LIST = [
    "EgoQA-GeLM-7B",
    "gemini-3-flash-preview",
    "InternVL3_5-38B",
    "InternVL3_5-8B", 
    "NexusQA",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen3-VL-235B-A22B-Instruct",
    "Qwen3-VL-235B-A22B-Thinking",
    "Qwen3-VL-32B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "qwen-vl-max",
    "VideoLLaMA3-7B",
    "videomind"
]

# API Configuration
AK_IDEA = "xx" 
BASE_URL = "xx"
JUDGE_MODEL = "gemini-3-flash-preview"

# Initialize OpenAI Client
client = OpenAI(
    api_key=AK_IDEA,
    base_url=BASE_URL,
)

def load_system_prompt():
    """Load the system prompt from an external file; return empty string if file is missing"""
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        return ""
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_PROMPT_TEXT = load_system_prompt()

def get_answer_gt(scenario, item):
    """Select Ground Truth field based on scenario"""
    if scenario in ["Assembly", "Natural"]:
        return item.get("answer_summary", "")
    elif scenario == "Function":
        return item.get("answer_complete", "")
    return ""

def get_model_predict(model_name, item):
    """Select Model Prediction field based on model_name"""
    if model_name == "EgoQA-GeLM-7B":
        return item.get("A", "")
    elif model_name == "NexusQA":
        return item.get("final_answer", "")
    else:
        return item.get("model_answer", "")

def call_llm_judge(question, gt_answer, model_predict, retries=3):
    """Call LLM for judgment with retry mechanism and system role"""
    user_prompt_text = f"""
        ### Output Format:
        Only output 'right' or 'wrong'. Do not provide any explanation or additional text.

        Question: {question}
        Ground Truth: {gt_answer}
        Model Prediction: {model_predict}
    """

    for i in range(retries):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEXT},
                    {"role": "user", "content": [{"type": "text", "text": user_prompt_text}]}
                ],
                temperature=0.1,
            )
            model_output = completion.choices[0].message.content
            judge_result = model_output.strip().lower()
            
            if "right" in judge_result:
                return "right"
            elif "wrong" in judge_result:
                return "wrong"
            else:
                return judge_result
        except Exception as e:
            if i < retries - 1:
                time.sleep(2)
                continue
            else:
                print(f"\n      [Error] Failed after {retries} retries: {e}")
                return "error"
    return "error"

def run_judge():
    for scenario in SCENARIO_LIST:
        scenario_path = os.path.join(RESULT_FOLDER_PATH, scenario)
        if not os.path.exists(scenario_path):
            continue
            
        print(f"\n>>> Scenario: {scenario}")

        for model_name in MODEL_NAME_LIST:
            model_path = os.path.join(scenario_path, model_name)
            final_folder = os.path.join(model_path, "final")
            
            if not os.path.exists(final_folder):
                print(f"No folder found for {model_name} in {scenario}")
                continue
            
            target_files = [f for f in os.listdir(final_folder) if f.endswith("_direct.json")]
            
            for file_name in target_files:
                json_path = os.path.join(final_folder, file_name)
                
                # Read JSON data
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except:
                        print(f"    Error: Could not parse {json_path}")
                        continue
                
                print(f"  - Processing: {model_name} / {file_name}")
                
                # Iterate through each data item
                for idx, item in enumerate(tqdm(data, desc="    Judging", leave=False)):
                    
                    # --- Resume from breakpoint logic ---
                    # Skip if the field already exists and is not 'error' or empty
                    if "llm_judge" in item and item["llm_judge"] not in ["error", ""]:
                        continue
                    
                    # Get variables needed for judgment
                    question = item.get("question", "")
                    gt_answer = get_answer_gt(scenario, item)
                    model_predict = get_model_predict(model_name, item)
                    
                    # Call LLM for judgment
                    judge_res = call_llm_judge(question, gt_answer, model_predict)
                    
                    # Update current data row
                    item["llm_judge"] = judge_res
                    
                    # --- Real-time writing logic ---
                    # Save progress after each iteration to ensure safety
                    with open(json_path, 'w', encoding='utf-8') as f_out:
                        json.dump(data, f_out, indent=4, ensure_ascii=False)

                print(f"    Finish/Verified: {file_name}")

if __name__ == "__main__":
    run_judge()