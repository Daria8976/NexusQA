#!/usr/bin/env bash
# HVSG construction pipeline wrapper
# Flow: 1_get_part_mask -> 2_add_label -> 3_refine_label (manual) -> 4_add_relation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP="${1:-all}" # all | 1 | 2 | 3 | 4

step1() {
  echo "[HVSG-1] Build part-level masks / part-aware graph"
  bash "$SCRIPT_DIR/hvsg/1_get_part_mask/run_part_graph_local.sh" all
}

step2() {
  echo "[HVSG-2] Add part labels"
  python "$SCRIPT_DIR/hvsg/2_add_label/add_label.py"
}

step3() {
  echo "[HVSG-3] Refine labels (manual step)."
  echo "Please place refined outputs under: $SCRIPT_DIR/hvsg/3_refine_label"
}

step4() {
  echo "[HVSG-4] Add/refine part relations"
  python "$SCRIPT_DIR/hvsg/4_add_relation/update_relation.py"
}

case "$STEP" in
  1) step1 ;;
  2) step2 ;;
  3) step3 ;;
  4) step4 ;;
  all) step1; step2; step3; step4 ;;
  *) echo "usage: bash run_hvsg_local.sh [all|1|2|3|4]"; exit 1 ;;
esac
