import av
import cv2
import os
import json

# =============================================================================
# Configuration — update these paths before running
# =============================================================================
INPUT_JSON_FILE = "/path/to/output/start_frames_epic.json"
VIDEOS_DIR = "/path/to/data/vidor/videos"
OUTPUT_DIR = "/path/to/output/vidor_key_frames"
OUTPUT_JSON_FILE = "/path/to/output/vidor_key_frames.json"


def find_and_save_nearest_keyframe(video_path, start_frame_index, output_folder):
    """Search bidirectionally for the nearest keyframe to *start_frame_index*,
    save it as a JPEG, and return its frame index."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    prev_kf_index = -1
    prev_kf_img = None
    next_kf_index = -1
    next_kf_img = None

    print(f"Target index: {start_frame_index}, searching for nearest keyframe (bidirectional)...")

    for i, frame in enumerate(container.decode(video_stream)):
        if frame.key_frame:
            if i <= start_frame_index:
                prev_kf_index = i
                prev_kf_img = frame.to_ndarray(format="bgr24")
                if i == start_frame_index:
                    next_kf_index = i
                    next_kf_img = prev_kf_img
                    break
            else:
                next_kf_index = i
                next_kf_img = frame.to_ndarray(format="bgr24")
                break

        if i > start_frame_index + 500:
            break

    container.close()

    final_index = -1
    final_img = None

    if prev_kf_index != -1 and next_kf_index != -1:
        if (start_frame_index - prev_kf_index) <= (next_kf_index - start_frame_index):
            final_index, final_img = prev_kf_index, prev_kf_img
        else:
            final_index, final_img = next_kf_index, next_kf_img
    elif prev_kf_index != -1:
        final_index, final_img = prev_kf_index, prev_kf_img
    elif next_kf_index != -1:
        final_index, final_img = next_kf_index, next_kf_img

    if final_index != -1:
        save_filename = f"{video_name}_nearest_kf_{final_index}.jpg"
        cv2.imwrite(os.path.join(output_folder, save_filename), final_img)
        print(f"Target: {start_frame_index}, prev_kf: {prev_kf_index}, next_kf: {next_kf_index}, "
              f"chosen: {final_index} (saved)")
    else:
        print("No keyframe found.")

    return final_index


def main():
    with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in data.keys():
        video_file = os.path.join(VIDEOS_DIR, f"{key}.mp4")
        if not os.path.exists(video_file):
            print(f"Video not found: {video_file}, skipping...")
            continue

        data[key]["key_frames_union"] = []
        for item in data[key]["timeline"]:
            start_index = item["start_frame"]
            output_dir = os.path.join(OUTPUT_DIR, key)
            final_index = find_and_save_nearest_keyframe(video_file, start_index, output_dir)
            item["key_frame"] = final_index
            data[key]["key_frames_union"].append(final_index)

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Done. Saved to {OUTPUT_JSON_FILE}")


if __name__ == "__main__":
    main()
