#!/usr/bin/env bash
# Local runner using scripts copied under NexusBench/construction
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PROCESS="$SCRIPT_DIR/data_process"

# ===== Local paths under hvsg/1_get_part_mask =====
INPUT_DIR="$SCRIPT_DIR/input"
ASSETS_DIR="$SCRIPT_DIR/assets"
OUTPUT_DIR="$SCRIPT_DIR/output"

PVSG_JSON="$INPUT_DIR/pvsg_vidor_graph_v2.json"
VIDEOS_DIR="$ASSETS_DIR/videos"
ORIG_MASKS_DIR="$ASSETS_DIR/original_masks"
SAM_MASKS_DIR="$OUTPUT_DIR/output_individual_masks"

START_FRAMES_JSON="$OUTPUT_DIR/start_frames.json"
KEY_FRAMES_JSON="$OUTPUT_DIR/vidor_key_frames.json"
KEY_FRAME_FINAL_JSON="$OUTPUT_DIR/vidor_key_frame_final.json"
KEY_FRAME_IMAGES_DIR="$OUTPUT_DIR/key_frame_images"
FRAME_MASKS_DIR="$OUTPUT_DIR/frame_masks"
OVERLAP_DIR="$OUTPUT_DIR/overlap_analysis"
OVERLAP_DIR_VIDOR="$OUTPUT_DIR/overlap_analysis_vidor"
PVSG_WITH_PARTS_JSON="$OUTPUT_DIR/pvsg_with_parts.json"
PVSG_PARTS_JSON="$OUTPUT_DIR/pvsg_parts.json"
PVSG_FINAL_JSON="$OUTPUT_DIR/pvsg_final.json"

STEP="${1:-all}" # 1..5 or all
mkdir -p \
  "$INPUT_DIR" \
  "$ASSETS_DIR" \
  "$VIDEOS_DIR" \
  "$ORIG_MASKS_DIR" \
  "$OUTPUT_DIR" \
  "$SAM_MASKS_DIR" \
  "$KEY_FRAME_IMAGES_DIR" \
  "$FRAME_MASKS_DIR" \
  "$OVERLAP_DIR" \
  "$OVERLAP_DIR_VIDOR"

step1() {
  python - <<PY
import json,sys
sys.path.insert(0, "$DATA_PROCESS")
import new_process as m
m.INPUT_PATH = "$PVSG_JSON"
m.OUTPUT_PATH = "$START_FRAMES_JSON"
with open(m.INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
out = m.process_video_data(data)
with open(m.OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(m.OUTPUT_PATH)
PY
}

step2() {
  python - <<PY
import sys
sys.path.insert(0, "$DATA_PROCESS")
import get_key_frame as m
m.INPUT_JSON_FILE = "$START_FRAMES_JSON"
m.VIDEOS_DIR = "$VIDEOS_DIR"
m.OUTPUT_DIR = "$KEY_FRAME_IMAGES_DIR"
m.OUTPUT_JSON_FILE = "$KEY_FRAMES_JSON"
m.main()
PY
}

step3() {
  echo "Run Semantic-SAM inference with local script: $SCRIPT_DIR/semantic_sam_part/pipeline_infer.py"
  python "$SCRIPT_DIR/semantic_sam_part/pipeline_infer.py"
}

step4() {
  python - <<PY
import sys
sys.path.insert(0, "$DATA_PROCESS")
import get_frame as m
m.INPUT_JSON = "$KEY_FRAMES_JSON"
m.OUTPUT_JSON = "$KEY_FRAME_FINAL_JSON"
m.TARGET_BASE_DIR = "$FRAME_MASKS_DIR"
m.DIR_V4 = "$OVERLAP_DIR"
m.DIR_VIDOR = "$OVERLAP_DIR_VIDOR"
m.process_json_and_collect_images(m.INPUT_JSON, m.OUTPUT_JSON)
PY

  python - <<PY
import sys
sys.path.insert(0, "$DATA_PROCESS")
import merge_masks as m
m.CONFIG["paths"]["key_frames"] = "$KEY_FRAME_FINAL_JSON"
m.CONFIG["paths"]["original_masks"] = "$ORIG_MASKS_DIR"
m.CONFIG["paths"]["generated_masks"] = "$SAM_MASKS_DIR"
m.CONFIG["paths"]["output_base"] = "$OVERLAP_DIR"
m.main()
PY
}

step5() {
  python - <<PY
import json,sys
sys.path.insert(0, "$DATA_PROCESS")
import add_part_to_graph as m
m.INPUT_PATH = "$PVSG_JSON"
m.VIS_BASE_DIR = "$OVERLAP_DIR"
m.OUTPUT_PATH = "$PVSG_WITH_PARTS_JSON"
with open(m.INPUT_PATH) as f:
    data = json.load(f)
out = m.process_data(data)
with open(m.OUTPUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(m.OUTPUT_PATH)
PY

  python - <<PY
import json,sys
sys.path.insert(0, "$DATA_PROCESS")
import align_format as m
m.INPUT_PATH = "$PVSG_WITH_PARTS_JSON"
m.OUTPUT_PATH = "$PVSG_PARTS_JSON"
with open(m.INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)
out = m.process_video_data(data)
with open(m.OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(m.OUTPUT_PATH)
PY

  python - <<PY
import sys
sys.path.insert(0, "$DATA_PROCESS")
import add_existence as m
m.INPUT_PATH = "$PVSG_PARTS_JSON"
m.process_json_file(m.INPUT_PATH, save_path="$PVSG_FINAL_JSON")
print("$PVSG_FINAL_JSON")
PY
}

case "$STEP" in
  1) step1 ;;
  2) step2 ;;
  3) step3 ;;
  4) step4 ;;
  5) step5 ;;
  all) step1; step2; step3; step4; step5 ;;
  *) echo "usage: bash hvsg/1_get_part_mask/run_part_graph_local.sh [1|2|3|4|5|all]"; exit 1 ;;
esac
