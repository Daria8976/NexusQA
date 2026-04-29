import os
import json
import glob
import re
import random
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
# Configuration — update these paths before running
# =============================================================================
CONFIG = {
    "paths": {
        "key_frames": "/path/to/output/key_frame_val_final.json",
        "original_masks": "/path/to/data/vidor/masks",
        "generated_masks": "/path/to/pipeline/Semantic-SAM/output_individual_masks",
        "output_base": "/path/to/output/overlap_analysis",
    },
    "params": {
        "iou_threshold": 0,
        "debug_video_limit": None,
    },
}

# =============================================================================
# Utility functions
# =============================================================================

def load_key_frames_data(json_path, video_id=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if video_id is not None:
        return {video_id: data[video_id]} if video_id in data else {}
    return data


def get_frames_and_objects_for_video(video_data, video_id):
    if video_id not in video_data:
        return {}
    video_info = video_data[video_id]
    frames_objects = {}

    frames_data = video_info.get("timeline", video_info)
    if isinstance(frames_data, dict):
        for f_str, f_info in frames_data.items():
            try:
                _extract_objects_from_frame_info(f_info, int(f_str), frames_objects)
            except Exception:
                continue
    elif isinstance(frames_data, list):
        for f_info in frames_data:
            fid = f_info.get("final_frame", f_info.get("frame_id"))
            _extract_objects_from_frame_info(f_info, fid, frames_objects)
    return frames_objects


def _extract_objects_from_frame_info(frame_info, frame_id, result_dict):
    if frame_id is None or not isinstance(frame_info, dict):
        return
    objects = frame_info.get("objects", [])
    obj_ids = [int(obj["object_id"]) if isinstance(obj, dict) else int(obj) for obj in objects]
    if obj_ids:
        result_dict[frame_id] = obj_ids


def load_original_mask_for_frame(original_masks_dir, frame_id, target_object_ids):
    mask_path = os.path.join(original_masks_dir, f"{frame_id:04d}.png")
    if not os.path.exists(mask_path):
        return {}
    try:
        mask_array = np.array(Image.open(mask_path))
        return {
            obj_id: (mask_array == obj_id).astype(np.uint8) * 255
            for obj_id in target_object_ids
            if np.any(mask_array == obj_id)
        }
    except Exception:
        return {}


def load_generated_masks_for_frame(generated_masks_base_path, video_id, frame_id):
    frame_dir = os.path.join(generated_masks_base_path, video_id, "masks", f"frame{frame_id:04d}")
    if not os.path.exists(frame_dir):
        return {}
    masks = {}
    for pf in glob.glob(os.path.join(frame_dir, "*.png")):
        try:
            nums = re.findall(r"\d+", os.path.basename(pf))
            obj_id = int(nums[-1])
            m = np.array(Image.open(pf))
            bm = (m.max(-1) > 0 if m.ndim == 3 else m > 0).astype(np.uint8) * 255
            if bm.any():
                masks[obj_id] = bm
        except Exception:
            continue
    return masks


def find_overlaps_between_masks(original_masks, generated_masks, iou_threshold=0.0):
    overlaps = []
    for gen_id, gen_mask in generated_masks.items():
        best_match = None
        max_overlap = -1
        for orig_id, orig_mask in original_masks.items():
            if orig_mask.shape != gen_mask.shape:
                g_res = cv2.resize(gen_mask, (orig_mask.shape[1], orig_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            else:
                g_res = gen_mask
            intersect = np.logical_and(orig_mask > 0, g_res > 0).sum()
            if intersect > max_overlap:
                max_overlap = intersect
                union = np.logical_or(orig_mask > 0, g_res > 0).sum()
                iou = intersect / union if union > 0 else 0
                best_match = {
                    "original_id": orig_id, "generated_id": gen_id, "iou": iou,
                    "overlap_pixels": int(intersect), "orig_area": int((orig_mask > 0).sum()),
                    "gen_area": int((gen_mask > 0).sum()),
                    "orig_shape": orig_mask.shape, "gen_shape": gen_mask.shape,
                }
        if best_match and best_match["overlap_pixels"] > 0 and best_match["iou"] > iou_threshold:
            overlaps.append(best_match)
    return overlaps


# =============================================================================
# Save & visualise
# =============================================================================

def create_combined_mask(original_mask, generated_mask, mode="full"):
    if original_mask.shape != generated_mask.shape:
        g_res = cv2.resize(generated_mask, (original_mask.shape[1], original_mask.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    else:
        g_res = generated_mask
    if mode == "overlap":
        return np.logical_and(original_mask > 0, g_res > 0).astype(np.uint8) * 255
    return g_res


def save_mask(mask_array, output_path, palette_mode=False):
    img = Image.fromarray(mask_array, mode="P" if palette_mode else "L")
    if palette_mode:
        pal = [0, 0, 0]
        for _ in range(255):
            pal.extend([random.randint(0, 255) for _ in range(3)])
        img.putpalette(pal)
    img.save(output_path)


def visualize_masks(orig_mask, gen_mask, overlap_mask, output_path, orig_id, gen_id, iou):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_mask, cmap="gray"); axes[0].set_title(f"Orig {orig_id}")
    axes[1].imshow(gen_mask, cmap="gray"); axes[1].set_title(f"Gen {gen_id}")
    axes[2].imshow(overlap_mask, cmap="hot"); axes[2].set_title(f"IoU {iou:.3f}")
    for ax in axes:
        ax.axis("off")
    plt.savefig(output_path, bbox_inches="tight"); plt.close()


# =============================================================================
# Core business logic
# =============================================================================

def process_frame_for_masks(original_masks_base_path, generated_masks_base_path,
                            video_id, frame_id, target_objects, video_output_root,
                            iou_threshold=0.1):
    original_masks = load_original_mask_for_frame(
        os.path.join(original_masks_base_path, video_id), frame_id, target_objects)
    if not original_masks:
        return {}
    generated_masks = load_generated_masks_for_frame(generated_masks_base_path, video_id, frame_id)
    if not generated_masks:
        return {}
    overlaps = find_overlaps_between_masks(original_masks, generated_masks, iou_threshold)
    if not overlaps:
        return {}

    results = {}
    for overlap in overlaps:
        oid, gid = overlap["original_id"], overlap["generated_id"]
        iou_val = overlap["iou"]
        iou_str = f"iou{int(iou_val * 1000):03d}"
        full_mask_name = f"frame{frame_id:04d}_orig{oid}_gen{gid}_full_{iou_str}.png"
        full_mask_data = create_combined_mask(original_masks[oid], generated_masks[gid], mode="full")
        save_mask(full_mask_data, os.path.join(video_output_root, full_mask_name), palette_mode=False)
        results[f"orig{oid}_gen{gid}"] = {
            **overlap, "full_file": full_mask_name,
            "orig_shape": list(overlap["orig_shape"]),
            "gen_shape": list(overlap["gen_shape"]),
        }
    return results


def process_video_for_overlaps(key_frames_path, original_masks_base_path,
                               generated_masks_base_path, output_base_path,
                               video_id, iou_threshold=0.1):
    video_output_root = os.path.join(output_base_path, video_id)
    os.makedirs(video_output_root, exist_ok=True)
    vis_dir = os.path.join(video_output_root, "visualizations")

    video_data = load_key_frames_data(key_frames_path, video_id)
    frames_objects = get_frames_and_objects_for_video(video_data, video_id)
    if not frames_objects:
        return None

    print(f"\n>>> Processing video: {video_id} ({len(frames_objects)} valid frames)")
    video_results = {"video_id": video_id, "frames": {}, "iou_threshold": iou_threshold}

    for frame_id, target_objects in frames_objects.items():
        check_pattern = os.path.join(video_output_root, f"frame{frame_id:04d}_*_full_iou*.png")
        if glob.glob(check_pattern):
            print(f"  [Skip] Frame {frame_id:04d} already exists.")
            continue
        try:
            frame_res = process_frame_for_masks(
                original_masks_base_path, generated_masks_base_path,
                video_id, frame_id, target_objects, video_output_root, iou_threshold)
            if frame_res:
                video_results["frames"][str(frame_id)] = {"frame_id": frame_id, "overlaps": frame_res}
                os.makedirs(vis_dir, exist_ok=True)
                orig_m_set = load_original_mask_for_frame(
                    os.path.join(original_masks_base_path, video_id), frame_id, target_objects)
                gen_m_set = load_generated_masks_for_frame(generated_masks_base_path, video_id, frame_id)
                for res in frame_res.values():
                    oid, gid = res["original_id"], res["generated_id"]
                    iou_str = f"iou{int(res['iou'] * 1000):03d}"
                    vis_name = f"frame{frame_id:04d}_orig{oid}_gen{gid}_{iou_str}_vis.png"
                    visualize_masks(orig_m_set[oid], gen_m_set[gid],
                                    create_combined_mask(orig_m_set[oid], gen_m_set[gid], "overlap"),
                                    os.path.join(vis_dir, vis_name), oid, gid, res["iou"])
        except Exception as e:
            print(f"  [Error] Frame {frame_id} failed: {e}")

    json_path = os.path.join(video_output_root, f"{video_id}_overlap_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(video_results, f, ensure_ascii=False, indent=2)
    return video_results


# =============================================================================
# Main entrance
# =============================================================================

def main():
    paths = CONFIG["paths"]
    iou_thresh = CONFIG["params"]["iou_threshold"]
    os.makedirs(paths["output_base"], exist_ok=True)

    video_ids = [d.name for d in Path(paths["original_masks"]).iterdir() if d.is_dir()]
    if CONFIG["params"]["debug_video_limit"]:
        video_ids = video_ids[: CONFIG["params"]["debug_video_limit"]]

    all_results = {}
    summary_json_path = os.path.join(paths["output_base"], "all_videos_overlap_results.json")

    if os.path.exists(summary_json_path):
        try:
            with open(summary_json_path, "r") as f:
                all_results = json.load(f)
            print(f"Loaded existing summary ({len(all_results)} records).")
        except Exception:
            pass

    for idx, vid in enumerate(video_ids):
        try:
            res = process_video_for_overlaps(
                paths["key_frames"], paths["original_masks"],
                paths["generated_masks"], paths["output_base"], vid, iou_thresh)
            if res:
                all_results[vid] = res
                if (idx + 1) % 5 == 0:
                    with open(summary_json_path, "w") as f:
                        json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"Video {vid} raised an exception: {e}")

    with open(summary_json_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nAll done! Results saved to: {paths['output_base']}")


if __name__ == "__main__":
    main()
