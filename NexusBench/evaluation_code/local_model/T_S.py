import os
import re
import ast
import json
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI 

# --- 1. Environment and Configuration ---
# vLLM server configuration
VLLM_BASE_URL = "http://0.0.0.0:8001/v1"
VLLM_API_KEY = "dummy"
# Note: The model name here must match the one specified when starting the vLLM service
MODEL_NAME = "/path/OpenGVLab/InternVL3_5-8B" 

# Unified paths based on previous scripts
INPUT_JSON_PATH = "/path/NexusBench/qae_triplet/Natural_Scenes.json"

# Dynamically extract the model name for the output path
model_basename = os.path.basename(MODEL_NAME)
OUTPUT_FOLDER_PATH = f"/path/NexusBench/results/Natural/{model_basename}"
MAX_RETRIES = 3

# Base video path
BASE_VIS_FOLDER = "oss_url/NexusBench/visualized_video/"

def convert_to_seconds(time_str):
    """
    Convert mm:ss.ss format to seconds (float)
    """
    try:
        minutes, seconds = time_str.split(':')
        return round(int(minutes) * 60 + float(seconds), 2)
    except Exception:
        return 0.0

def parse_model_output(text):
    """
    Parse model output, extract Answer and Evidence_Chain
    """
    try:
        # Extract Answer
        answer_pattern = r"(?i)Answer:\s*(.*?)(?=\s*Evidence_Chain:|$)"
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        
        # Extract Evidence_Chain
        chain_pattern = r"(?i)Evidence_Chain:\s*(.*)"
        chain_match = re.search(chain_pattern, text, re.DOTALL)

        model_answer = answer_match.group(1).strip() if answer_match else None
        evidence_chain = chain_match.group(1).strip() if chain_match else None

        return model_answer, evidence_chain
    except Exception:
        return None, None

def main():
    # --- 2. Prepare paths and checkpoint check ---
    input_path = Path(INPUT_JSON_PATH)
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER_PATH, input_path.name)

    processed_results = []
    processed_ids = set()

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                processed_results = json.load(f)
                processed_ids = {item['sample_id'] for item in processed_results}
                print(f"Resuming: {len(processed_ids)} items already processed.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # --- 3. Initialize vLLM client ---
    client = OpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
    )

    # --- 4. Load input data ---
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # --- 5. Loop inference ---
    print(f"Starting processing. Total items: {len(input_data)}")

    for item in tqdm(input_data):
        if item['sample_id'] in processed_ids:
            continue

        sample_id = item.get("sample_id")
        question_text = item.get("question")
        video_id = item.get("video_id")
        
        # --- Construct visualized video URL (consistent with original script) ---
        if video_id:
            video_url = f"{BASE_VIS_FOLDER}{video_id}_{sample_id}.mp4"
        else:
            video_url = item.get("video_url")
        
        # --- Time parsing logic (kept unchanged) ---
        temporal_data = item.get("evidence", {}).get("temporal", {})
        reference_timestamps = []
        for key in sorted(temporal_data.keys()):
            time_range = temporal_data[key]
            if len(time_range) == 2:
                start_sec = convert_to_seconds(time_range[0])
                end_sec = convert_to_seconds(time_range[1])
                reference_timestamps.append([start_sec, end_sec])

        # Construct Prompt
        user_prompt_text = f"""
            Please analyze the provided video to answer the question. 

            ### Reference Context:
            To assist your analysis, we have provided two types of cues:
            1. **Temporal Evidence**: The following time intervals (in seconds) are identified as containing the key evidence: {reference_timestamps}.
            2. **Visual Highlights**: Key **instances** (including specific objects and the interacting parts) relevant to the question are **visually highlighted** in the video (using bounding boxes or masks) and are accompanied by **text labels**.

            ### Question:
            {question_text}

            ### Instructions:
            1. **Utilize Visual Cues**: In your analysis, you must explicitly refer to the objects and actions that are highlighted and labeled in the video.
            2. **Evidence Chain Analysis**: Explain the reasoning process by describing what specifically occurs within the provided reference timestamps.
            3. **Detail the Interaction**: Describe the specific actions, movements, or state changes of the labeled instances and explain how they lead to your final conclusion.
            4. **No Simple Answers**: Do NOT simply answer "Yes" or "No". Provide a comprehensive explanation.

            ### Output Format:
            Your response must strictly follow this format:
            Answer: <Your final conclusion based on the analysis.>
            Evidence_Chain: <A detailed, step-by-step description of what is happening during the reference timestamps, specifically mentioning the highlighted/labeled instances and how they proves the answer.>
        """

        final_ans, final_chain = None, None
        
        for attempt in range(MAX_RETRIES):
            try:
                # 4. Call vLLM API
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_url",
                                    "video_url": {"url": video_url}
                                },
                                {
                                    "type": "text", 
                                    "text": user_prompt_text
                                },
                            ],
                        }
                    ],
                    max_tokens=1024
                )
                
                raw_response = completion.choices[0].message.content.strip()
                final_ans, final_chain = parse_model_output(raw_response)
                
                if final_ans and final_chain:
                    break
                else:
                    print(f"\n[Warning] Parsing failed for ID {item['sample_id']}, attempt {attempt+1}")
            
            except Exception as e:
                print(f"\n[Error] API call failed for ID {item['sample_id']}: {e}")

        # Update data item
        item["model_answer"] = final_ans
        item["model_evidence_chain"] = final_chain
        
        # Real-time saving logic
        processed_results.append(item)
        processed_ids.add(item['sample_id'])

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, indent=4, ensure_ascii=False)

    print(f"\nTask Finished. Output: {output_path}")

if __name__ == "__main__":
    main()