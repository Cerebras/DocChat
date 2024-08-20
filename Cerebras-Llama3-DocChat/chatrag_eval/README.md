## Evaluation
The original ChatRAG repo does not support the Llama3 Instruct chat template, as it manually injects special tokens & formats the chat history. This is a fork of the evaluation code which supports instruct models like ours. The provided eval code works on CPU & GPU in order to ensure that everyone can evaluate our model (including those who don't have access to CSX).

### Usage
```
./eval_all.sh cerebras/Llama3-DocChat-1.0-8B
```