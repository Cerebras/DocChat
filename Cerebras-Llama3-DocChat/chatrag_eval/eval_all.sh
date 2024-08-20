#!/bin/bash

# Usage: ./eval_all.sh cerebras/llama3-DocChat-1.0-8b
PYTHON="python" # replace this with path to your python venv (if applicable)
DATA_PATH="../../Nvidia-ChatQA/ChatRAG-Bench/data/"
MODEL_ID=$1
OUTPUT_FOLDER=$(basename "$(dirname "$MODEL_ID")")_$(basename "$MODEL_ID")

DATASETS=("sqa" "convfinqa" "hybridial" "doqa_cooking" "doqa_movies" "doqa_travel" "doc2dial" "inscit" "coqa"  "qrecc"  "quac"  "topiocqa")

for dataset in ${DATASETS[@]}; do
    echo "Generating output & scores for: ${dataset}"
    $PYTHON run_generation_vllm.py --model-id $MODEL_ID --eval-dataset $dataset --data-folder $DATA_PATH --output-folder $OUTPUT_FOLDER --max-tokens 256
    $PYTHON get_scores.py --eval-dataset $dataset --prediction-file "${OUTPUT_FOLDER}/${dataset}_output.txt" --data-folder $DATA_PATH | tee "${OUTPUT_FOLDER}/${dataset}_scores.txt"
done
