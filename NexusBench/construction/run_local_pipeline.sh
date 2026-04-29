#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# 1) HVSG (4 steps inside run_hvsg_local.sh)
bash NexusBench/construction/run_hvsg_local.sh all

# 2) QA pair generation
python NexusBench/construction/qa_pair/generate_qa_piar/generate_qa_batch_ns.py

# 3) Build final Natural_Scenes.json
python NexusBench/construction/build_natural_scenes.py \
  --input-dir NexusBench/construction/qa_pair/generate_qa_piar/output_json \
  --output-json NexusBench/qae_triplet/Natural_Scenes.json \
  --video-root NexusBench/video/natural \
  --video-url-prefix /NexusBench/video/natural \
  --time-unit frame
