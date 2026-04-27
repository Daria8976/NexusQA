import json
import time
from pathlib import Path
from openai import OpenAI  

# --- 1. Configuration Area ---
INPUT_JSON_PATH = Path('/path/NexusBench/qae_triplet/Natural_Scenes.json') 

MODEL_IDENTIFIER = "gemini-3-flash-preview"
# MODEL_IDENTIFIER = "qwen-vl-max"
MODEL_ENDPOINT_NAME = MODEL_IDENTIFIER 

# --- 2. API Client Setup ---
API_KEY = "xx"
BASE_URL = "xx"

try:
    client = OpenAI(
        api_key=API_KEY,  # Insert your API Key here
        base_url=BASE_URL, # Insert your API base URL, e.g.
    )
except Exception as e:
    print(f"Failed to initialize OpenAI client. Please check API Key and Base URL configuration: {e}")
    client = None # Set to None for later checking


def process_file_realtime_save():
    # Check if the specific file exists
    if not INPUT_JSON_PATH.is_file():
        print(f"Error: File '{INPUT_JSON_PATH}' does not exist.")
        return
        
    if not client:
        print("Error: API client was not successfully initialized. Exiting program.")
        return

    print(f"\n--- Processing file: {INPUT_JSON_PATH.name} ---")

    # 1. Construct new filename (original filename + model identifier)
    new_filename = f"{INPUT_JSON_PATH.stem}_{MODEL_IDENTIFIER}{INPUT_JSON_PATH.suffix}"

    # 2. Construct target directory path (directory path only)
    target_dir = Path(f"/path/NexusBench/results/Natural/{MODEL_IDENTIFIER}")

    # 3. [Crucial] Ensure target directory exists! Create if it doesn't
    target_dir.mkdir(parents=True, exist_ok=True)

    # 4. Construct full target file path (directory + new filename)
    new_filepath = target_dir / new_filename

    # --- Resume/Checkpoint Logic ---
    processed_dataset = []
    processed_identifiers = set() # Use a set for quick lookup of processed items
    
    try:
        if new_filepath.exists():
            with open(new_filepath, 'r', encoding='utf-8') as f:
                processed_dataset = json.load(f)
            # Assuming the combination of 'video_url' and 'question' in each item is unique
            for item in processed_dataset:
                # Create a unique identifier
                identifier = (item.get("video_url"), item.get("question"))
                processed_identifiers.add(identifier)
            print(f"Existing output file '{new_filepath.name}' detected. Loaded {len(processed_dataset)} processed records. Resuming from checkpoint.")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load or parse existing output file '{new_filepath.name}'. Starting over. Error: {e}")
        processed_dataset = []
        processed_identifiers = set()

    # Load the original dataset
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            original_dataset = json.load(f)
    except Exception as e:
        print(f"Error: Could not read original file {INPUT_JSON_PATH.name}. Exiting. Error: {e}")
        return

    # Start processing
    for i, item in enumerate(original_dataset):
        # Use 'video_url' and 'question' as a unique identifier to check if already processed
        current_identifier = (item.get("video_url"), item.get("question"))
        
        if current_identifier in processed_identifiers:
            print(f"  - Item {i+1}/{len(original_dataset)} already processed, skipping.")
            continue
        
        sample_id = item.get("sample_id", "Unknown ID")
        print(f"  - Processing item {i+1}/{len(original_dataset)} (ID: {sample_id})...")
        
        try:
            question_text = item["question"]
            video_url = item["video_url"]
            
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

                Answer: <Your response text>
                Temporal Evidence: <[[s, e]] or [[s1, e1], [s2, e2], ...]>
                Spatial Evidence: {{"timestamp": <float>, "bbox": [ymin, xmin, ymax, xmax]}}
            """

            completion = client.chat.completions.create( 
                model=MODEL_ENDPOINT_NAME,
                messages=[{ "role": "user", "content": [{"type": "video_url", "video_url": {"url": video_url}}, {"type": "text", "text": user_prompt_text}]}],
            )

            model_answer = completion.choices[0].message.content
            
            # Create a new entry and add it to the dataset
            new_item = item.copy()
            new_item["model_answer"] = model_answer
            processed_dataset.append(new_item)

            # --- Core: Real-time saving ---
            # Write the updated list back to the file
            with open(new_filepath, 'w', encoding='utf-8') as f:
                json.dump(processed_dataset, f, indent=4, ensure_ascii=False)
            
            # Mark the recently processed item as completed
            processed_identifiers.add(current_identifier)
            print(f"    √ Processed and saved successfully.")

        except Exception as e:
            print(f"    ! Error occurred while processing item {i+1}: {e}")
            print(f"    ! Will retry this item on the next run.")
            # Brief sleep to prevent spamming the API if there's a temporary issue like rate limiting
            time.sleep(5) 

    print(f"File '{INPUT_JSON_PATH.name}' processing complete! All results saved to: {new_filepath.name}")


if __name__ == "__main__":
    process_file_realtime_save()
    print("\nAll tasks completed!")