import json
import os
import glob
import shutil
from pathlib import Path

# =============================================================================
# Configuration — update these paths before running
# =============================================================================
INPUT_JSON = "/path/to/output/vidor_key_frames_val.json"
OUTPUT_JSON = "/path/to/output/vidor_key_frame_val_final.json"

TARGET_BASE_DIR = "/path/to/output/overlap_analysis_vidor_val_final"
DIR_V4 = "/path/to/output/overlap_analysis"
DIR_VIDOR = "/path/to/output/overlap_analysis_vidor"


def process_json_and_collect_images(json_path, output_json_path=None):
    """Process the key-frame JSON, collect corresponding mask images from two
    source directories (chosen by frame-diff threshold), and copy them to a
    unified target directory.

    For timeline items where ``|key_frame - start_frame| > 10`` the mask is
    sourced from *DIR_V4* using the start_frame number; otherwise from
    *DIR_VIDOR* using the key_frame number.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {
        "total_videos": 0,
        "total_timeline_items": 0,
        "total_objects": 0,
        "frame_diff_gt_10": 0,
        "frame_diff_lte_10": 0,
        "images_collected": 0,
        "images_not_found": 0,
    }

    for video_id, video_data in data.items():
        stats["total_videos"] += 1
        print(f"\nProcessing video: {video_id}")

        target_video_dir = os.path.join(TARGET_BASE_DIR, video_id)
        os.makedirs(target_video_dir, exist_ok=True)

        timeline = video_data.get("timeline", [])
        for timeline_item in timeline:
            stats["total_timeline_items"] += 1

            start_frame = timeline_item.get("start_frame")
            key_frame = timeline_item.get("key_frame")
            objects = timeline_item.get("objects", [])

            if start_frame is None or key_frame is None:
                print(f"  Warning: missing start_frame or key_frame in timeline item")
                continue

            frame_diff = abs(key_frame - start_frame)

            if frame_diff > 10:
                stats["frame_diff_gt_10"] += 1
                source_dir = os.path.join(DIR_V4, video_id, "masks", "full_masks")
                target_frame_num = start_frame
            else:
                stats["frame_diff_lte_10"] += 1
                source_dir = os.path.join(DIR_VIDOR, video_id, "masks", "full_masks")
                target_frame_num = key_frame

            timeline_item["final_frame"] = target_frame_num

            if not os.path.exists(source_dir):
                print(f"  Warning: source directory not found: {source_dir}")
                continue

            for obj in objects:
                stats["total_objects"] += 1
                object_id = obj.get("object_id")
                if object_id is None:
                    continue

                obj_pattern = os.path.join(
                    source_dir, f"frame{target_frame_num:04d}_orig{object_id}*"
                )
                matching_files = glob.glob(obj_pattern)

                if not matching_files:
                    print(f"  No match for pattern: {os.path.basename(obj_pattern)} in {source_dir}")
                    stats["images_not_found"] += 1
                    continue

                for src_file in matching_files:
                    dst_file = os.path.join(target_video_dir, os.path.basename(src_file))
                    try:
                        shutil.copy2(src_file, dst_file)
                        stats["images_collected"] += 1
                    except Exception as e:
                        print(f"  Copy failed for {os.path.basename(src_file)}: {e}")

    print(f"\n{'=' * 50}")
    print("Processing complete! Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  Target directory: {TARGET_BASE_DIR}")

    if output_json_path:
        output_dir = os.path.dirname(output_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nUpdated JSON saved to: {output_json_path}")

    return data, stats


if __name__ == "__main__":
    process_json_and_collect_images(INPUT_JSON, OUTPUT_JSON)
