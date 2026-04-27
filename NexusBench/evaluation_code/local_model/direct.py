import os
import re
import ast
import json
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI 

# --- 1. Configuration ---
# vLLM server configuration
VLLM_BASE_URL = "http://0.0.0.0:8001/v1"
VLLM_API_KEY = "dummy"
# Note: The model name here must match the one specified when starting the vLLM service
MODEL_NAME = "/path/OpenGVLab/InternVL3_5-8B" 

INPUT_JSON_PATH = "/path/NexusBench/qae_triplet/Natural_Scenes.json"

# Dynamically extract the model name for the output path
model_basename = os.path.basename(MODEL_NAME)
OUTPUT_FOLDER_PATH = f"/path/NexusBench/results/Natural/{model_basename}"
MAX_RETRIES = 3

def parse_model_output(text):
    """
    Parse model output and convert to standardized variables
    """
    try:
        # 1. Extract Answer
        answer_pattern = r"(?i)Answer:\s*(.*?)(?=\s*Temporal Evidence:|\s*Spatial Evidence:|$)"
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        model_answer = answer_match.group(1).strip() if answer_match else None

        # 2. Extract Temporal Evidence
        temp_match = re.search(r"(?i)Temporal Evidence:\s*(\[\[.*?\]\])", text, re.DOTALL)
        model_timestamps = ast.literal_eval(temp_match.group(1).strip()) if temp_match else None

        # 3. Extract Spatial Evidence
        spatial_match = re.search(r"(?i)Spatial Evidence:\s*(\{.*?\})", text, re.DOTALL)
        model_spatial = [ast.literal_eval(spatial_match.group(1).strip())] if spatial_match else None

        return model_answer, model_timestamps, model_spatial
    except Exception:
        return None, None, None

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
                print(f"Resuming from checkpoint: {len(processed_ids)} items already processed.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")

    # --- 3. Initialize vLLM client (OpenAI compatible) ---
    client = OpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
    )

    # --- 4. Load input data ---
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # --- 5. Loop API calls ---
    print(f"Starting processing. Total items: {len(input_data)}")
    
    for item in tqdm(input_data):
        if item['sample_id'] in processed_ids:
            continue

        video_url = item.get("video_url")
        question_text = item.get("question")
        
        user_prompt_text = f"""
            Please analyze the provided video to answer the question, providing both temporal and spatial evidence as grounding.

            ### Question:
            {question_text}

            ### Instructions:
            1. **Answer**: Provide a detailed text response. If the action is continuous, describe it as a single event. If it consists of multiple steps at different times, describe the sequence clearly.
            2. **Temporal Evidence**: Identify the time interval(s) where the evidence occurs.
            - **Format**: ALWAYS use a nested list: `[[start, end]]` for a single interval, or `[[s1, e1], [s2, e2]]` for multiple intervals.
            - Use seconds as the unit (e.g., 12.5). 
            - Even if there is only one time segment, you must wrap it in double brackets: `[[s, e]]`.
            3. **Spatial Evidence**: Pick the most representative frame within the temporal evidence.
            - **Timestamp**: The exact second of that frame.
            - **Bounding Box**: The coordinates [ymin, xmin, ymax, xmax] of the key object.
            - **Normalization**: Normalized to [0, 1000] relative to the frame height and width.

            ### Output Format:
            Your response must strictly follow this structure:

            Answer: <Answer the Question>
            Temporal Evidence: <[[s, e]] or [[s1, e1], [s2, e2], ...]>
            Spatial Evidence: {{"timestamp": <float>, "bbox": [ymin, xmin, ymax, xmax]}}
        """

        # Inference and retry
        final_ans, final_ts, final_sp = None, None, None
        
        for attempt in range(MAX_RETRIES):
            try:
                # Call vLLM API
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
                    max_tokens=512
                )
                
                raw_response = completion.choices[0].message.content.strip()
                final_ans, final_ts, final_sp = parse_model_output(raw_response)
                
                # Break the retry loop if parsing is successful
                if all([final_ans, final_ts, final_sp]):
                    break
                else:
                    print(f"\n[Warning] Parsing failed for ID {item['sample_id']}, attempt {attempt+1}")
            
            except Exception as e:
                print(f"\n[Error] API call failed for ID {item['sample_id']}: {e}")

        # Update data item
        item["model_answer"] = final_ans
        item["model_timestamps"] = final_ts
        item["model_spatial"] = final_sp
        
        # Real-time saving logic
        processed_results.append(item)
        processed_ids.add(item['sample_id'])

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, indent=4, ensure_ascii=False)

    print(f"\nProcessing complete. Final file saved at: {output_path}")

if __name__ == "__main__":
    main()