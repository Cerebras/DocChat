# Cerebras Dragon-DocChat

Cerebras Dragon-DocChat was built on top of the Dragon+ model and trained on ChatQA’s conversational Q&A dataset. By finetuning using contrastive loss with hard negatives, we see absolute improvements in recall of 8.9% over Dragon+ and 3.5% over ChatQA Dragon-Multiturn respectively (top-1). More details about DocChat can be found in our blog post.

Links:
* [Blog post](https://www.cerebras.net/blog/train-a-gpt-4-level-conversational-qa-in-a-few-hours)
* [LLM model weights on HuggingFace](https://huggingface.co/cerebras/Llama3-DocChat-1.0-8B)
* Embedding model weights on HuggingFace: [Query Encoder](https://huggingface.co/cerebras/Dragon-DocChat-Query-Encoder), [Context Encoder](https://huggingface.co/cerebras/Dragon-DocChat-Context-Encoder)
* [Data preparation, training, and evaluation code](https://github.com/Cerebras/DocChat)

## Replication

This repository is designed to be used in conjunction with Cerebras Model-Zoo (rel 2.3). Before you get started, make sure you've already downloaded and installed Model-Zoo.

**Setup**
```console
source /path/to/your/mz/venv/bin/activate
pip install langchain_text_splitters    # install text splitters for data processing
pip install -U pydantic spacy           # only required if you get a pdyantic typing error in scripts below
export PYTHONPATH=/path/to/modelzoo/src     # this should point to the cerebras dir in MZ
```

Next, prepare the dataset & process it into HDF5:
```console
python create_train_dataset.py
```
The prepared dataset will be located at `./processed_datasets/syntheticConvQA_hdf5/`

Before we start training, we also need to acquire Dragon+ HF checkpoint and convert it into CS compatible checkpoint.
```console
python convert_embedding_checkpoint.py --direction hf_to_cs
```
The result of this process will be the initial checkpoint for training. This checkpoint is located at `checkpoints/CS/facebook_dragon/`.

We're now ready to train on CSX. The config that we will be using for training is located at `train_configs/params_train.yaml`. If you want to experiment with hyperparameters, feel free to modify this file. Update `train.sh` with your Model-Zoo path, virtual environment, and cluster namespace.

Finally to train, simply run `./train.sh`. Once the model compiles, the training process will take a few minutes.

You can up tensorboard by pointing it to your model directory (which stores training artifacts):
```console
tensorboard --logdir ./docchat_retrieval_model_dir --bind_all
```

Once training is complete, we can convert the final checkpoint back to HuggingFace.
```console
python convert_embedding_checkpoint.py \
    --direction cs_to_hf \
    --embedding_trained_checkpoint_path ./docchat_retrieval_model_dir/checkpoint_900.mdl
```
The converted model will be saved to `checkpoints/HF/cerebras_dragon/`.

## Evaluation
For instructions on evaluation, look at `embedding_eval/README.md`. The evaluation scripts are based on Nvidia's Dragon Multi-turn eval code found https://huggingface.co/nvidia/dragon-multiturn-query-encoder/tree/main/evaluation. The provided eval code works on CPU & GPU in order to ensure that everyone can evaluate our model (including those who don't have access to CSX).