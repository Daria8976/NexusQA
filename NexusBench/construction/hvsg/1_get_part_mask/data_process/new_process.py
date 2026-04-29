import json
from collections import Counter, defaultdict
import re

# ---------------------------------------------------------------------------
# Configuration — update these paths before running
# ---------------------------------------------------------------------------

INPUT_PATH = "/path/to/pvsg.json"
OUTPUT_PATH = "/path/to/output/start_frames.json"

GAP_THRESHOLD = 1


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def merge_intervals(intervals, gap_threshold=1):
    """Merge a list of (start, end) intervals, treating gaps <= gap_threshold as
    continuous."""
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])

    merged = []
    curr_start, curr_end = intervals[0]

    for next_start, next_end in intervals[1:]:
        if next_start <= curr_end + gap_threshold:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end

    merged.append((curr_start, curr_end))
    return merged


def process_video_data(json_data):
    output_result = {}

    for video in json_data.get("data", []):
        video_id = video.get("video_id")

        # Build object lookup table.
        obj_lookup = {}
        for obj in video.get("objects", []):
            obj_lookup[obj["object_id"]] = {
                "category": obj["category"],
                "is_thing": obj["is_thing"],
            }

        valid_relations = []
        for rel in video.get("relations", []):
            if len(rel) < 4:
                continue
            sub_id, obj_id, _predicate, segments = rel
            sub_info = obj_lookup.get(sub_id)
            obj_info = obj_lookup.get(obj_id)
            if sub_info and obj_info:
                valid_relations.append({"sub": sub_id, "obj": obj_id, "segments": segments})

        # Count object mentions in captions.
        object_mentions = []
        for caption in video.get("captions", []):
            matches = re.findall(r"\((\d+)\)", caption["description"])
            object_mentions.extend(int(m) for m in matches)

        freq_counter = Counter(object_mentions)
        top_3_ids = {pid for pid, _ in freq_counter.most_common(3)}

        # Expand to include interaction partners of the top-3 objects.
        target_object_ids = set(top_3_ids)
        for item in valid_relations:
            s, o = item["sub"], item["obj"]
            if s in top_3_ids:
                target_object_ids.add(o)
            if o in top_3_ids:
                target_object_ids.add(s)

        # Collect raw time intervals per object.
        obj_raw_intervals = defaultdict(list)
        for item in valid_relations:
            sub, obj = item["sub"], item["obj"]
            if sub in top_3_ids:
                for seg in item["segments"]:
                    obj_raw_intervals[sub].append(tuple(seg))
                if obj in target_object_ids:
                    for seg in item["segments"]:
                        obj_raw_intervals[obj].append(tuple(seg))
            elif obj in top_3_ids:
                for seg in item["segments"]:
                    obj_raw_intervals[obj].append(tuple(seg))
                if sub in target_object_ids:
                    for seg in item["segments"]:
                        obj_raw_intervals[sub].append(tuple(seg))

        # Merge intervals per object and build timeline.
        timeline_map = defaultdict(list)
        for oid in target_object_ids:
            raw_segs = obj_raw_intervals.get(oid, [])
            if not raw_segs:
                continue
            for start, _end in merge_intervals(raw_segs, gap_threshold=GAP_THRESHOLD):
                timeline_map[start].append(oid)

        # Assemble the final output list.
        video_timeline = []
        for start_frame in sorted(timeline_map.keys()):
            active_obj_ids = timeline_map[start_frame]
            formatted_objects = sorted(
                [
                    {
                        "object_id": oid,
                        "category": obj_lookup[oid]["category"],
                        "count": freq_counter[oid],
                        "is_thing": obj_lookup[oid]["is_thing"],
                    }
                    for oid in active_obj_ids
                ],
                key=lambda x: x["object_id"],
            )
            video_timeline.append(
                {
                    "start_frame": start_frame,
                    "object_count": len(formatted_objects),
                    "objects": formatted_objects,
                }
            )

        output_result[video_id] = video_timeline

    return output_result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        print("Reading data...")
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        print("Processing data...")
        result_dict = process_video_data(json_data)

        print(f"Writing results to: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

        print("Done!")

    except Exception:
        import traceback
        traceback.print_exc()
