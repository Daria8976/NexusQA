import os
import sys
import json
import time
import torch
import numpy as np
import cv2
import warnings
from datetime import datetime
from torchvision import transforms
from PIL import Image

try:
    from semantic_sam.BaseModel import BaseModel
    from semantic_sam import build_model
    from utils.arguments import load_opt_from_config_file
    from tasks.automatic_mask_generator import SemanticSamAutomaticMaskGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Run this script from the Semantic-SAM root directory.")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# =============================================================================
# Paths (edit before running)
CKPT_PATH   = "/path/to/checkpoints/swinl_only_sam_many2many.pth"
CONFIG_PATH = "configs/semantic_sam_only_sa-1b_swinL.yaml"


# =============================================================================
# JSON helpers
# =============================================================================

def load_or_create_json(json_path, default_data=None):
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: malformed JSON at {json_path}, starting fresh")
    return default_data if default_data is not None else {}


def save_json(json_path, data):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    tmp = f"{json_path}.tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.rename(tmp, json_path)


# =============================================================================
# Mask I/O
# =============================================================================

def save_individual_masks(image_array, masks_info, start_object_id, output_dir, frame_id, img_size=None):
    H, W = image_array.shape[:2]
    target_w, target_h = img_size if img_size else (W, H)
    os.makedirs(output_dir, exist_ok=True)

    current_id = start_object_id
    parts_info = []
    for ann in masks_info:
        m = ann['segmentation']
        mask_img = (m * 255).astype(np.uint8)
        if img_size and (W != target_w or H != target_h):
            mask_img = cv2.resize(mask_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        filename = f"frame{frame_id:04d}_{current_id:05d}.png"
        cv2.imwrite(os.path.join(output_dir, filename), mask_img)

        scale_w, scale_h = (target_w / W, target_h / H) if img_size else (1.0, 1.0)
        parts_info.append({
            "part_id":   current_id,
            "mask_file": filename,
            "area":      int(ann['area'] * scale_w * scale_h),
            "bbox":      ann['bbox'].tolist() if hasattr(ann['bbox'], 'tolist') else ann['bbox'],
        })
        current_id += 1
    return current_id, parts_info


def infer_frame(model, image_pil, level, text_size, output_dir, frame_id, start_id=1, img_size=None):
    transform = transforms.Compose([transforms.Resize(int(text_size), interpolation=Image.BICUBIC)])
    image_ori = transform(image_pil)
    image_np  = np.asarray(image_ori)
    images    = torch.from_numpy(image_np.copy()).permute(2, 0, 1).cuda()

    generator = SemanticSamAutomaticMaskGenerator(
        model, points_per_side=32,
        pred_iou_thresh=0.88, stability_score_thresh=0.92,
        min_mask_region_area=10, level=level,
    )
    outputs = generator.generate(images)
    if not outputs:
        return start_id, []

    print(f"  {len(outputs)} masks detected")
    return save_individual_masks(image_np, outputs, start_id, output_dir, frame_id, img_size)


# =============================================================================
# Per-video processing
# =============================================================================

def process_video(video_id, info, base_frames_path, output_base_path, model,
                  level, text_size, img_size=None, parts_info_path=None):
    frame_ids       = sorted(set(info["final_frames_union"]))
    max_object_id   = info["max_object_id"]
    frames_path     = os.path.join(base_frames_path, video_id)
    masks_out_path  = os.path.join(output_base_path, video_id, "masks")

    print(f"\n=== Video {video_id} | {len(frame_ids)} frames ===")
    current_id = max_object_id + 1

    # pre-load existing frame data to maintain ID continuity on resume
    existing_frames = {}
    if parts_info_path and os.path.exists(parts_info_path):
        existing_frames = (load_or_create_json(parts_info_path)
                           .get("videos", {}).get(video_id, {}).get("frames", {}))

    video_info = {
        "video_id": video_id,
        "max_original_object_id": max_object_id,
        "masks_output_dir": masks_out_path,
        "frames": {},
        "start_time": datetime.now().isoformat(),
    }

    done = 0
    for idx, frame_id in enumerate(frame_ids):
        frame_path  = os.path.join(frames_path, f"{frame_id:04d}.png")
        frame_dir   = os.path.join(masks_out_path, f"frame{frame_id:04d}")

        if not os.path.exists(frame_path):
            print(f"  Skip: {frame_id:04d}.png not found")
            continue

        # resume: if output folder already has PNG files, skip inference
        if os.path.exists(frame_dir) and any(f.endswith('.png') for f in os.listdir(frame_dir)):
            print(f"  Frame {frame_id:04d} already done, skipping")
            str_fid = str(frame_id)
            if str_fid in existing_frames:
                fi = existing_frames[str_fid]
                video_info["frames"][frame_id] = fi
                current_id = fi.get("end_object_id", current_id - 1) + 1
            continue

        print(f"  [{idx+1}/{len(frame_ids)}] frame {frame_id:04d}")
        try:
            image_pil = Image.open(frame_path).convert('RGB')
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    next_id, new_parts = infer_frame(
                        model, image_pil, level, text_size,
                        frame_dir, frame_id, current_id, img_size,
                    )

            video_info["frames"][frame_id] = {
                "start_object_id": current_id,
                "end_object_id":   next_id - 1,
                "num_masks":       len(new_parts),
                "masks_dir":       frame_dir,
                "new_parts":       new_parts,
            }
            current_id = next_id
            done += 1

            # incremental JSON update
            if parts_info_path:
                all_data = load_or_create_json(parts_info_path, {"summary": {}, "videos": {}})
                if video_id not in all_data["videos"]:
                    all_data["videos"][video_id] = video_info
                else:
                    all_data["videos"][video_id]["frames"][str(frame_id)] = video_info["frames"][frame_id]
                save_json(parts_info_path, all_data)

        except Exception as e:
            print(f"  Error on frame {frame_id}: {e}")

    video_info["final_object_id"]  = current_id - 1
    video_info["processed_frames"] = done
    return current_id, video_info


def save_results(parts_data, output_path):
    data = load_or_create_json(output_path, {"summary": {}, "videos": {}})
    data["videos"].update(parts_data)
    v = data["videos"].values()
    data["summary"] = {
        "total_videos":  len(data["videos"]),
        "total_frames":  sum(len(vi.get("frames", {})) for vi in v),
        "total_masks":   sum(sum(f.get("num_masks", 0) for f in vi.get("frames", {}).values()) for vi in v),
        "last_updated":  datetime.now().isoformat(),
    }
    save_json(output_path, data)


# =============================================================================
# Entry point
# =============================================================================

def main():
    opt   = load_opt_from_config_file(CONFIG_PATH)
    model = BaseModel(opt, build_model(opt)).from_pretrained(CKPT_PATH).eval().cuda()

    # --- edit these paths ---
    json_path        = "/path/to/data_process/output/vidor_key_frame_val_final.json"
    base_frames_path = "/path/to/data/vidor/frames"
    output_base_path = "/path/to/pipeline/Semantic-SAM/output_individual_masks_vidor"
    parts_info_path  = "/path/to/data_process/output/individual_masks_info_vidor.json"
    # ---

    with open(json_path, 'r', encoding='utf-8') as f:
        video_data = json.load(f)

    for video_id, info in video_data.items():
        try:
            t0 = time.time()
            _, video_parts_info = process_video(
                video_id, info, base_frames_path, output_base_path,
                model, [1, 2, 3, 4, 5, 6], 640,
                img_size=None, parts_info_path=parts_info_path,
            )
            video_parts_info["processing_time"] = f"{int((time.time()-t0)//60)}m"
            save_results({video_id: video_parts_info}, parts_info_path)
        except Exception as e:
            print(f"Fatal error on video {video_id}: {e}")


if __name__ == "__main__":
    main()
