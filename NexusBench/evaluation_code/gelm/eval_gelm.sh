MODEL_NAME=EgoQA-GeLM-7B
MODEL_PATH=/path/SurplusDeficit/GeLM/EgoQA-GeLM-7B
QUESTION_FILE=/path/NexusBench/qae_triplet/Natural_Scenes.json
FEATURE_FOLDER=/path/NexusBench/feature/Natural_Scenes

CUDA_VISIBLE_DEVICES=3 python3 /path/NexusBench/evaluation_code/gelm/gelm.py \
    --model-path $MODEL_PATH  \
    --question-file $QUESTION_FILE \
    --image-folder $FEATURE_FOLDER \
    --output-dir /path/NexusBench/results/Natural/$MODEL_NAME \
    --conv-mode v1 \