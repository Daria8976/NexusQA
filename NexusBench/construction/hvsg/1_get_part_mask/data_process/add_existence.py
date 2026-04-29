import json
import os

# =============================================================================
# Configuration — update these paths before running
# =============================================================================
INPUT_PATH = "/path/to/output/pvsg_parts.json"


def merge_intervals(intervals):
    """Merge overlapping time intervals (standard sweep-line algorithm)."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(current)
    return merged


def process_json_file(file_path, save_path=None):
    """Add ``existence`` time segments to every object (and part) in the JSON.

    For parent objects the existence is derived from all relation segments they
    participate in.  For part objects the existence is the single parent
    segment that contains the part's first ``frame_id``.
    """
    print(f"Reading file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found — {file_path}")
        return

    data_list = json_data.get("data", [])
    total_processed = 0

    for item in data_list:
        relations = item.get("relations", [])
        objects = item.get("objects", [])

        # Step 1: build {object_id: [segments...]} from relations.
        obj_time_map = {}
        for rel in relations:
            if len(rel) < 4:
                continue
            sub_id, obj_id, _predicate, segments = rel
            obj_time_map.setdefault(sub_id, []).extend(segments)
            obj_time_map.setdefault(obj_id, []).extend(segments)

        # Step 2: assign existence to each object and its parts.
        for obj in objects:
            oid = obj.get("object_id")

            # Parent object existence.
            parent_merged_segments = merge_intervals(obj_time_map.get(oid, []))
            obj["existence"] = parent_merged_segments

            # Part existence: pick the parent segment that contains the part's
            # first frame_id.
            for part in obj.get("parts", []):
                part_existence = []
                frame_ids = part.get("frame_ids", [])
                if frame_ids and parent_merged_segments:
                    start_frame = frame_ids[0]
                    for seg in parent_merged_segments:
                        if seg[0] <= start_frame <= seg[1]:
                            part_existence.append(seg)
                            break
                part["existence"] = part_existence

        total_processed += 1

    # Step 3: save results.
    if save_path is None:
        base, ext = os.path.splitext(file_path)
        save_path = f"{base}_processed{ext}"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    print(f"Done. Processed {total_processed} video entries.")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    process_json_file(INPUT_PATH)
