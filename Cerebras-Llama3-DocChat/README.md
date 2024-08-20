# Cerebras Llama3-DocChat

Cerebras Llama3-DocChat was built on top of Llama 3 base using insights from the latest research on document-based Q&A, most notably Nvidia’s ChatQA model series. As part of this work, we leveraged our experience in LLM model training and dataset curation to reverse engineer ChatQA’s training recipe. Additionally, we employed synthetic data generation to address limitations that couldn't be fully resolved with the available real data.

Links:
* [Blog post](https://www.cerebras.net/blog/train-a-gpt-4-level-conversational-qa-in-a-few-hours)
* [LLM model weights on HuggingFace](https://huggingface.co/cerebras/Llama3-DocChat-1.0-8B)
* Embedding model weights on HuggingFace: [Query Encoder](https://huggingface.co/cerebras/Dragon-DocChat-Query-Encoder), [Context Encoder](https://huggingface.co/cerebras/Dragon-DocChat-Context-Encoder)
* [Data preparation, training, and evaluation code](https://github.com/Cerebras/DocChat)

## Replication

This repository is designed to be used in conjunction with [Cerebras Model Zoo](https://github.com/Cerebras/modelzoo) (rel 2.3). Before you get started, make sure you've already downloaded and installed Model Zoo.

**Setup**
```console
source /path/to/your/mz/venv/bin/activate
export PYTHONPATH=/path/to/modelzoo/src     # this should point to the cerebras dir in MZ
```

You will need to access the Llama3 8B base model and the Llama3 8B instruct's tokenizer via HuggingFace. Either log in via Hugging Face's CLI or manually set your HuggingFace token:
```console
export HF_TOKEN="YOUR_AUTH_TOKEN"
```

Next, prepare the dataset & process it into HDF5:
```console
python create_train_dataset.py
```
In the `./processed_datasets/<dataset name>/` folders, you will find readable jsonl files and corresponding pre-processed h5 files.

Before we start training, we also need to acquire the Llama3 base HF checkpoint and convert it into a Model Zoo compatible checkpoint.
```console
python convert_llm_checkpoint.py --direction hf_to_cs
```
The result of this process will be the initial checkpoint for training. This checkpoint is located at `checkpoints/CS/Meta-Llama-3-8B/`.

We're now ready to train on CSX. The configs that we will be using during the two stage training procedure are located in the `train_configs/` directory. If you want to experiment with hyperparameters, feel free to modify this file. Update `train.sh` with your Model-Zoo path, virtual environment, and cluster namespace.

Finally to train, simply run `./train.sh`. You can expect the training process to take a few hours in total when running on a single CS system.

While the training runs, you can set up tensorboard by pointing it to your model directory (which stores training artifacts):
```console
tensorboard --logdir_spec PHASE1:chatqa_inst_stage_1,PHASE2:chatqa_inst_stage_2 --bind_all
```

Once training is complete, convert the final checkpoint back to HuggingFace.
```console
python convert_llm_checkpoint.py \
    --direction cs_to_hf \
    --llm_trained_checkpoint_path ./chatqa_inst_stage_2/checkpoint_4555.mdl
```
The converted model will be saved to `checkpoints/HF/Cerebras-Llama3-DocChat-8B/`.

## Evaluation
The original ChatRAG repo does not support the Llama3 Instruct chat template, as it manually injects special tokens & formats the chat history. We've created a fork of the evaluation code located in the `chatrag_eval` folder which supports instruct models like ours. The provided eval code works on CPU & GPU in order to ensure that everyone can evaluate our model (including those who don't have access to CSX).


## Additional Notes

The mixture of datasets that we use have varying means/variance of sequence lengths, and so a non-trivial amount of the 8K MSL is actually padding tokens. In order to train faster, we train with Variable Sequence Length (VSL). VSL packs together multiple samples into the same sequence while also masking the attention between the samples in the same sequence. Note that VSL affects the number of unmasked tokens per batch. In order to retain the original intended training dynamics, we need to 1) correspondingly change the batch size and 2) correspondingly update the mixture weights.

Phase 1 of training is intended to be to have a global batch size of 128. Accounting for the "packing factor" from VSL, we train with a batch size of 10.
Similarly, the phase 2 non-VSL batch size is 64. This corresponds to a batch size of 5 with VSL.
The following table shows the mixture weights before and after VSL:

| Dataset                         | Pre-VSL Weights | Packing Factor | Post-VSL Weight |
| ------------------------------- | --------------- | -------------- | --------------- |
| SKG Instruct                    | 0.1371          | 16.8           | 0.0791          |
| NuminaMath-CoT                  | 0.1334          | 16             | 0.0812          |
| Synthetic ConvQA (Answerable)   | 0.132           | 5.2            | 0.2448          |
| Hybrid Dialogue                 | 0.1165          | 4.4            | 0.259           |
| SFT Mix                         | 0.0951          | 13.1           | 0.0703          |
| TAT QA Arithmetic               | 0.0713          | 12.4           | 0.0558          |
| NarrativeQA                     | 0.0452          | 9.8            | 0.045           |
| NewsQA                          | 0.0452          | 9.5            | 0.0463          |
| Squad 1.1                       | 0.0452          | 33.8           | 0.013           |
| Squad 2.0                       | 0.0452          | 33.8           | 0.013           |
| TAT QA Others                   | 0.038           | 14.2           | 0.026           |
| Drop                            | 0.0328          | 22.2           | 0.0144          |
| TAT QA Arithmetic CoT           | 0.0276          | 9.8            | 0.0274          |
| Quoref                          | 0.0124          | 16.9           | 0.0071          |
| Ropes                           | 0.0124          | 35.4           | 0.0034          |
| Synthetic ConvQA (Unanswerable) | 0.0107          | 7.4            | 0.0141          |
