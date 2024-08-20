The evaluation scripts are based on Nvidia's Dragon Multi-turn eval code found https://huggingface.co/nvidia/dragon-multiturn-query-encoder/tree/main/evaluation. The provided eval code works on CPU & GPU in order to ensure that everyone can evaluate our model (including those who don't have access to CSX).

### Commands for running evaluation

```console
python evaluate.py --eval-dataset doc2dial --query-encoder-path /path/to/query/encoder  --context-encoder-path /path/to/context/encoder
python evaluate.py --eval-dataset quac --query-encoder-path /path/to/query/encoder  --context-encoder-path /path/to/context/encoder
python evaluate.py --eval-dataset qrecc --query-encoder-path /path/to/query/encoder  --context-encoder-path /path/to/context/encoder
```

For the evaluations of topiocqa and inscit, the Wikipedia corpora need to be downloaded from their original repositories.

- topiocqa: https://github.com/McGill-NLP/topiocqa
- inscit: https://github.com/ellenmellon/INSCIT