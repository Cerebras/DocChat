#!/bin/bash
this_dir=$(dirname "$0")

if [ "$1" == "tatqa-cot" ]; then
    outdir="${this_dir}/tatqa-cot"
    mkdir -p $outdir
    wget https://huggingface.co/datasets/cerebras/TAT-QA-Arithmetic-CoT/resolve/main/train_arithmetic_cot.json --directory-prefix $outdir
elif [ "$1" == "hybrid-dialogue" ]; then
    outdir="${this_dir}/hybrid-dialogue"
    mkdir -p $outdir
    wget https://huggingface.co/datasets/cerebras/HybridDialogue/resolve/main/train.jsonl --directory-prefix $outdir
else
    echo "Input must either be 'tatqa-cot' or 'hybrid-dialogue'"
    exit 1
fi







