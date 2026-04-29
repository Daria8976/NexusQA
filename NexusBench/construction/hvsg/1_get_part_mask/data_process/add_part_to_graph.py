import json
import os
import re

# =============================================================================
# Configuration — update these paths before running
# =============================================================================
INPUT_PATH = "/path/to/data/pvsg.json"
VIS_BASE_DIR = "/path/to/output/visualization/"
OUTPUT_PATH = "/path/to/output/pvsg_with_parts.json"

IOU_MIN = 0
IOU_MAX = 900

# Regex for filenames like: frame0000_orig7_gen39_full_iou118_vis.png
FILENAME_PATTERN = re.compile(r"frame(\d+)_orig(\d+)_gen(\d+)_full_iou(\d+)_vis\.png")


def process_data(data_obj):
    """Walk through each video's objects, initialise part fields, then scan
    the visualisation directory to create / update part objects based on
    filename metadata (orig_id, gen_id, IoU score)."""

    # Step 1: initialise part-related fields on all existing objects.
    for video in data_obj["data"]:
        for obj in video["objects"]:
            obj["is_part"] = False
            obj["belonging"] = []
            obj["frame_ids"] = []

    # Step 2: scan visualisation files and create / update generated (part) objects.
    for video in data_obj["data"]:
        video_id = video["video_id"]
        video_path = os.path.join(VIS_BASE_DIR, video_id)

        if "objects" not in video:
            video["objects"] = []
        if not os.path.exists(video_path):
            continue

        for filename in os.listdir(video_path):
            match = FILENAME_PATTERN.match(filename)
            if not match:
                continue

            frame_id = int(match.group(1))
            orig_id = int(match.group(2))
            gen_id = int(match.group(3))
            iou_score = int(match.group(4))

            if not (IOU_MIN <= iou_score <= IOU_MAX):
                continue

            # Find the parent (original) object to derive category.
            orig_obj = next((item for item in video["objects"] if item["object_id"] == orig_id), None)
            if not orig_obj:
                continue

            gen_obj = next((item for item in video["objects"] if item["object_id"] == gen_id), None)

            if gen_obj is None:
                # Create a new part object.
                new_object = {
                    "object_id": gen_id,
                    "category": f"{orig_obj['category']} (part)",
                    "is_thing": True,
                    "is_part": True,
                    "belonging": [orig_id],
                    "status": [],
                    "frame_ids": [frame_id],
                }
                video["objects"].append(new_object)
                print(f"Created object: Video {video_id} | Object {gen_id} from {orig_id}")
            else:
                # Update existing part object.
                if orig_id not in gen_obj["belonging"]:
                    gen_obj["belonging"].append(orig_id)
                if frame_id not in gen_obj["frame_ids"]:
                    gen_obj["frame_ids"].append(frame_id)
                    gen_obj["frame_ids"].sort()
                gen_obj["category"] += f", {orig_obj['category']} (part)"

    return data_obj


if __name__ == "__main__":
    with open(INPUT_PATH, "r") as f:
        json_data = json.load(f)

    processed_json = process_data(json_data)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(processed_json, f, indent=4)
    print(f"Done. Saved to {OUTPUT_PATH}")
