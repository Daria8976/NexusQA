# NexusBench & NexusQA

[![Dataset](https://img.shields.io/badge/Dataset-NexusBench-CFAFD4)](https://huggingface.co/datasets/anonymous-anonymous-anonymous/NexusBench) 

## Introduction

This repository addresses the "blind reasoning" issue in Multimodal Large Language Models (MLLMs) for VideoQA—where models rely on language priors rather than genuine visual grounding—by providing a comprehensive framework for exploring multi-grained spatio-temporal evidence. 

**NexusBench** is a novel benchmark constructed upon the Hierarchical Video Scene Graph (HVSG) that advances visual evidence from coarse-grained objects to part-level dynamic mask tubes, compelling models to perform joint reasoning and verification across multi-grained spatio-temporal dimensions. 

As a highly competitive training-free baseline, **NexusQA** employs a "plan-act" collaborative architecture, leveraging dynamic interaction between a Strategy Agent and a Grounding Agent to precisely retrieve fine-grained evidence and drive reliable visual reasoning.

![alt text](assets/overview.png)


## Release Process
- [x] **NexusBench**
  - [x] Dataset: Question-answer pairs with evidence annotations
  - [x] Model evaluation code
  - [x] Metrics and judging scripts
  - [x] Construction Pipeline
- [ ] **NexusQA**
  - [ ] Inference pipeline
  - [ ] Prompts



## NexusBench
![alt text](assets/nexusbench.png)

**Structure:**
- **Nature Scenes**: 527 QA pairs about daily environment interactions
- **Function Verification**: 399 QA pairs examining functional relationships
- **Assembly Process**: 417 QA pairs focusing on object assembly sequences

Each QA pair follows a standardized template with:
- Video ID and URL
- Question and complete answer
- Evidence annotations (temporal intervals and spatial bounding boxes)

```json
{
    "video_id": "xxx",
    "video_url": "xxx",
    "question": "xxx",
    "answer_complete": "xxx",
    "evidence": {
        "temporal": {
            "<T1>": ["start time/frame index", "end time/frame index"],
            "<T2>": ["start time/frame index", "end time/frame index"]
        },
        "spatial": {
            "instance_1_name": "{id_1}",
            "instance_2_name": "{id_2}"
        }
    },
    "type": "{QA type classification}"
}
```
![alt text](assets/nexusbench_statistics.png)

### Evaluating Models with NexusBench

We provide model evaluation scripts in [`NexusBench/evaluation_code`](NexusBench/evaluation_code) for benchmarking different VideoQA systems on NexusBench. The scripts cover both API-based models and locally hosted models, and support the evaluation settings used in this repository:

- **API models**: [`api_model/api_model.py`](NexusBench/evaluation_code/api_model/api_model.py) evaluates OpenAI-compatible multimodal API endpoints by prompting the model to produce an answer, temporal evidence, and spatial evidence.
- **Local models**: [`local_model/direct.py`](NexusBench/evaluation_code/local_model/direct.py), [`local_model/T.py`](NexusBench/evaluation_code/local_model/T.py), and [`local_model/T_S.py`](NexusBench/evaluation_code/local_model/T_S.py) evaluate locally served models through a vLLM/OpenAI-compatible interface under direct answering, temporal-grounded, and temporal+spatial-grounded settings.
- **Model-specific scripts**: [`gelm/gelm.py`](NexusBench/evaluation_code/gelm/gelm.py) and [`videomind/eval_videomind.py`](NexusBench/evaluation_code/videomind/eval_videomind.py) provide evaluation adapters for GeLM and VideoMind-style pipelines.

Before running an evaluation, update the dataset path, model name, API/base URL, video path, and output directory in the corresponding script. The expected output is a JSON file that keeps the original NexusBench fields and appends model predictions such as `model_answer`, `model_timestamps`, `model_spatial`, or model-specific grounding fields.

### Metrics

Evaluation utilities are provided in [`NexusBench/metrics`](NexusBench/metrics) to measure both answer correctness and grounding quality:

- [`llm_judge.py`](NexusBench/metrics/llm_judge.py) uses an LLM judge to compare model predictions with the ground-truth answer and writes the judgment result back to the result JSON.
- [`t_IoU_IoP_IoG.py`](NexusBench/metrics/t_IoU_IoP_IoG.py) computes temporal grounding scores, including IoU, IoP, and IoG, by matching predicted temporal intervals with annotated evidence intervals.
- [`S_IoU.py`](NexusBench/metrics/S_IoU.py) evaluates spatial grounding by comparing predicted bounding boxes with annotated key-frame object boxes.
- [`summary.py`](NexusBench/metrics/summary.py) summarizes dataset-level and category-level results, including QA accuracy, mIoU, IoU@0.3, mIoP, mIoG, and Acc@IoU.

The typical workflow is to first run a model evaluation script, then apply the temporal/spatial metric scripts and LLM judge to the generated result file, and finally use `summary.py` to report the overall and category-wise performance.

## NexusQA

NexusQA provides an advanced evaluation framework featuring:
- "Plan-Act" Collaborative Architecture: Decouples complex VideoQA tasks into specialized roles, utilizing a Strategy Agent for task concretization and a Grounding Agent for dynamic evidence collection.
- Multi-Grained Grounding Tools: Integrates specialized tool calls ([Temporal Scout Tool], [Spatial Inspector Tool], and [Reasoning Tool]) to precisely locate temporal event intervals and track fine-grained spatial details (e.g., part-level masklets).
- Dynamic Context Accumulation: Employs a shared Nexus Context that prevents attention dilution, allowing models to synthesize long temporal sequences and minute spatial details effectively.

The framework supports various evaluation modes:
- Direct answering
- Temporal grounding (+T)
- Temporal + Spatial grounding (+T+S)


![alt text](assets/nexusqa_architecture.png)


## Evaluation
![alt text](assets/result1.png)
![alt text](assets/result2.png)
![alt text](assets/result3.png)
