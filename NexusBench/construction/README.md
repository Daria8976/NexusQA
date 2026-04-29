# NexusBench Construction

## 1. Directory Structure

```text
NexusBench/construction/
├── hvsg/
│   ├── 1_get_part_mask/
│   ├── 2_add_label/
│   ├── 3_refine_label/
│   └── 4_add_relation/
├── qa_pair/
│   ├── generate_qa_piar/
│   └── utils/convert.py
├── build_natural_scenes.py
├── run_hvsg_local.sh
├── run_construction_pipeline.py
└── run_local_pipeline.sh
```

## 2. Pipeline

1. Build HVSG
   - Implemented in `hvsg/` with four stages:
   - `1_get_part_mask` -> `2_add_label` -> `3_refine_label` -> `4_add_relation`
2. Build QA pairs
   - Implemented by `qa_pair/generate_qa_piar/generate_qa_batch_ns.py`
3. Build final `Natural_Scenes.json`
   - Implemented by `build_natural_scenes.py`

## 3. Quick Start

Run a single script:

```bash
bash NexusBench/construction/run_local_pipeline.sh
```


## 4. Prerequisites

- Set OpenAI runtime environment variables (for label/relation/QA generation):

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...      
export NEXUSBENCH_QA_MODEL=...  
```
