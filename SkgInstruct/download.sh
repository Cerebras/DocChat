#!/bin/bash
outdir=$(dirname "$0")

wget https://huggingface.co/datasets/TIGER-Lab/SKGInstruct/resolve/main/skginstruct.json --directory-prefix $outdir