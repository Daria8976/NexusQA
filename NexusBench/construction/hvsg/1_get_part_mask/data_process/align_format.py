import json

# =============================================================================
# Configuration — update these paths before running
# =============================================================================
INPUT_PATH = "/path/to/output/pvsg_with_parts.json"
OUTPUT_PATH = "/path/to/output/pvsg_parts.json"


def process_video_data(data):
    """Restructure flat objects list into a parent-child hierarchy.

    Objects with ``is_part=True`` are moved under their parent object's
    ``parts`` list.  Non-part objects remain at the top level and each
    receive an empty ``parts`` list.
    """
    for video in data.get("data", []):
        raw_objects = video.get("objects", [])

        # Build an ID-to-object map and initialise ``parts`` on non-part objects.
        obj_map = {}
        for obj in raw_objects:
            obj_map[obj.get("object_id")] = obj
            if not obj.get("is_part", False):
                obj["parts"] = []

        root_objects = []
        for obj in raw_objects:
            if obj.get("is_part", False):
                belonging_list = obj.get("belonging", [])
                if belonging_list:
                    parent_id = belonging_list[0]
                    parent_obj = obj_map.get(parent_id)
                    if parent_obj and "parts" in parent_obj:
                        parent_obj["parts"].append(obj)
                    else:
                        root_objects.append(obj)
                else:
                    root_objects.append(obj)
            else:
                root_objects.append(obj)

        video["objects"] = root_objects

    return data


if __name__ == "__main__":
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = process_video_data(data)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Done. Saved to {OUTPUT_PATH}")
