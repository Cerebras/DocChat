import argparse
import json
import os
import random
import subprocess
from enum import Enum

import h5py
import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class DatasetType(Enum):
    SHORT_QA = 1
    LONG_QA = 2
    NUMERICAL_QA = 3
    NUMERICAL_QA_COT = 4
    SHORT_AND_LONG_QA = 5
    SFT = 6


dataset_type_to_instruction = {
    DatasetType.SHORT_QA: "Answer the following question with a short span.",
    DatasetType.LONG_QA: "Please give a full and complete answer for the question.",
    DatasetType.NUMERICAL_QA: "Answer the following question with a number from context or the math arithmetic using +, -, *, or /.",
    DatasetType.NUMERICAL_QA_COT: "Answer the following question using the context by thinking step by step to produce a final number or math arithmetic using +, -, *, or /.",
    DatasetType.SHORT_AND_LONG_QA: "Answer the following question with a short span, or a full and complete answer.",
    DatasetType.SFT: None,
}


dataset_types = {
    "drop": DatasetType.SHORT_QA,
    "narrativeqa": DatasetType.SHORT_QA,
    "quoref": DatasetType.SHORT_QA,
    "ropes": DatasetType.SHORT_QA,
    "squad1.1": DatasetType.SHORT_QA,
    "squad2.0": DatasetType.SHORT_QA,
    "newsqa": DatasetType.SHORT_QA,
    "tatqa_arithmetic": DatasetType.NUMERICAL_QA,
    "tatqa_arithmetic_cot": DatasetType.NUMERICAL_QA_COT,
    "tatqa_others": DatasetType.SHORT_AND_LONG_QA,
    "synthetic_convqa": DatasetType.LONG_QA,
    "sft": DatasetType.SFT,
    "skginstruct": DatasetType.SFT,
    "hybridial": DatasetType.LONG_QA,
}


def apply_chat_template(tokenizer, messages):
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert formatted.startswith("<|begin_of_text|>")
    return formatted[len("<|begin_of_text|>") :]


def get_formatted_input(tokenizer, messages, context, instruction):
    system = (
        "This is a chat between a user and an artificial intelligence "
        "assistant. The assistant gives helpful, detailed, and polite answers "
        "to the user's questions based on the context. The assistant should "
        "also indicate when the answer cannot be found in the context."
    )

    if instruction is not None or context is not None:
        for item in messages:
            if item['role'] == "user":
                # apply instruction to the first user turn
                if instruction is not None:
                    item['content'] = instruction + " " + item['content']
                # apply context to the first user turn
                if context is not None:
                    formatted_context = "<context>\n" + context
                    if not context.endswith("\n"):
                        formatted_context += "\n"
                    formatted_context += "</context>\n"
                    item["content"] = formatted_context + item["content"]
                break

    messages = [{"role": "system", "content": system}] + messages

    return apply_chat_template(tokenizer, messages)


def load_dataset(dataset_name):
    if dataset_name == "hybridial":
        file = os.path.join(
            "cerebras_datasets", "hybrid-dialogue", "train.jsonl"
        )
        if not os.path.exists(file):
            proc = subprocess.run(
                "cerebras_datasets/download.sh hybrid-dialogue", shell=True
            )
            assert proc.returncode == 0
        with open(file) as f:
            dataset = [json.loads(line) for line in f]
    elif dataset_name == "skginstruct":
        file = os.path.join("../SkgInstruct/skginstruct.json")
        if not os.path.exists(file):
            proc = subprocess.run("../SkgInstruct/download.sh", shell=True)
            assert proc.returncode == 0
        with open(file, "r") as f:
            dataset = json.load(f)
    else:
        if dataset_name.startswith("tatqa"):
            tatqa_type = dataset_name[len("tatqa_") :]
            if tatqa_type == "arithmetic_cot":
                file = os.path.join(
                    "cerebras_datasets", "tatqa-cot", f"train_{tatqa_type}.json"
                )
                if not os.path.exists(file):
                    proc = subprocess.run(
                        "cerebras_datasets/download.sh tatqa-cot", shell=True
                    )
                    assert proc.returncode == 0
            else:
                file = os.path.join(
                    "../Nvidia-ChatQA/ChatQA-Training-Data/tatqa",
                    f"train_{tatqa_type}.json",
                )
        else:
            file = os.path.join(
                "../Nvidia-ChatQA/ChatQA-Training-Data",
                dataset_name,
                "train.json",
            )
        with open(file, "r") as f:
            dataset = json.load(f)
    return dataset


def generate_dataset_chatqa(dataset_name, tokenizer, shuffle=True):
    dataset_type = dataset_types[dataset_name]
    instruction = dataset_type_to_instruction[dataset_type]

    dataset = load_dataset(dataset_name)

    if shuffle:
        random.shuffle(dataset)

    for sample in tqdm(dataset, desc=dataset_name):
        messages, answers = sample["messages"], sample["answers"]
        document = sample["document"] if "document" in sample else None

        if len(answers) == 0:
            continue
        assert len(answers) == 1
        completion = answers[0]

        # NarrativeQA has multiple answers. Pick one:
        if isinstance(completion, list):
            completion = random.choice(completion)
        # Squad returns spans + text. We'll only use text:
        if isinstance(completion, dict):
            completion = completion["text"]
        if completion is None:
            continue

        formatted_prompt = get_formatted_input(
            tokenizer, messages, document, instruction
        )

        yield {"prompt": formatted_prompt, "completion": completion}


def generate_dataset_skginstruct(dataset_name, tokenizer, shuffle=True):
    dataset = load_dataset(dataset_name)
    if shuffle:
        random.shuffle(dataset)

    disallow_tasks = set(
        [
            "slimorca",
            "grailqa",
            "sql2text",
            "logic2text",
            "mtop",
            "mmqa",
            "spider_with_cell",
            "sparc_with_cell",
        ]
    )

    for sample in tqdm(dataset, desc=dataset_name):
        if sample["task_name"] in disallow_tasks:
            continue
        if sample["is_truncated"]:
            continue

        if sample["sys_prompt"] is None or sample["sys_prompt"] == "":
            continue

        messages = [
            {"role": "system", "content": sample["sys_prompt"]},
            {"role": "user", "content": sample["input"]},
        ]

        formatted_prompt = apply_chat_template(tokenizer, messages)
        completion = sample["label"]

        yield {"prompt": formatted_prompt, "completion": completion}


def read_and_stack_h5_files(directory):
    h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]

    total_examples = 0
    total_active_tokens = 0
    sequence_length = 0
    for i, file_name in enumerate(h5_files):
        file_path = os.path.join(directory, file_name)
        try:
            with h5py.File(file_path, 'r') as f:
                data = f['data']
                total_examples += data.shape[0]
                sequence_length = data.shape[-1]
                total_active_tokens += np.sum(data[:, 1, :])

        except Exception as e:
            print(e)

    return total_examples, total_active_tokens, sequence_length


def print_h5_stats(hdf5_output_dir, readable_file=None):
    total_examples, total_active_tokens, sequence_length = (
        read_and_stack_h5_files(hdf5_output_dir)
    )
    print(hdf5_output_dir)
    print(f"Total number of examples: {total_examples}")
    print(
        f"Avg active tokens per seq: {total_active_tokens/total_examples:.1f}"
    )
    print(f"Seq len: {sequence_length}")
    if readable_file:
        with open(readable_file, "r") as f:
            num_lines = sum(1 for _ in f)
            print(f"Number of lines:", num_lines)
        print("VSL Compression factor:", num_lines / total_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cerebras DocChat LLM Dataset Processing"
    )
    parser.add_argument(
        '--mz_path',
        default=os.environ.get("PYTHONPATH"),
        type=str,
        help='Path to modelzoo',
    )

    parser.add_argument(
        '--processes',
        default=16,
        type=int,
        help='Number of processes to use for HDF5 processing',
    )

    args = parser.parse_args()
    assert (
        args.mz_path is not None
    ), "mz_path must be provided or implicitly set via PYTHONPATH"

    tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    MSL = 8192
    output_dir = "processed_datasets"
    SHUFFLE = True
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Synthetic ConvQA is a special case: we break the samples into two
    # different datasets (based on answerable/unanswerable) so that we can
    # upsample unanswerable samples during training.
    datasets_names = set(dataset_types.keys())
    datasets_names.remove("synthetic_convqa")
    datasets_names.add("synthetic_convqa_answer")
    datasets_names.add("synthetic_convqa_no_answer")

    for dataset_name in datasets_names:
        if not os.path.exists(os.path.join(output_dir, dataset_name)):
            os.makedirs(os.path.join(output_dir, dataset_name, "jsonl"))

        output_file = os.path.join(
            output_dir, dataset_name, "jsonl", "train.jsonl"
        )
        if os.path.exists(output_file):
            print(f"Skipping {dataset_name} because file already exists")
            continue

        with open(output_file, "w") as f:
            if dataset_name.startswith("synthetic_convqa_"):
                aliased_dataset_name = "synthetic_convqa"
                dataset_sub_type = dataset_name[len("synthetic_convqa_") :]
                assert dataset_sub_type in ["answer", "no_answer"]
            else:
                aliased_dataset_name = dataset_name

            dataset_generator_fn = (
                generate_dataset_skginstruct
                if aliased_dataset_name == "skginstruct"
                else generate_dataset_chatqa
            )

            for sample in dataset_generator_fn(
                aliased_dataset_name, tokenizer, shuffle=SHUFFLE
            ):

                if aliased_dataset_name == "synthetic_convqa":
                    if (
                        dataset_sub_type == "answer"
                        and sample["completion"]
                        == "Sorry. I cannot find the answer based on the context."
                    ):
                        continue
                    elif (
                        dataset_sub_type == "no_answer"
                        and sample["completion"]
                        != "Sorry. I cannot find the answer based on the context."
                    ):
                        continue

                f.write(json.dumps(sample) + "\n")

        hdf5_dir = "with_vsl_hdf5"

        input_dir = os.path.join(output_dir, dataset_name, "jsonl")
        hdf5_output_dir = os.path.join(output_dir, dataset_name, hdf5_dir)

        config = {
            'setup': {
                'input_dir': input_dir,
                'output_dir': hdf5_output_dir,
                'processes': args.processes,
                'dataset_processor': ('VSLSummarizationPreprocessor'),
            },
            'processing': {
                'tokenizer_type': 'HuggingFaceTokenizer',
                'huggingface_tokenizer': tokenizer_path,
                'max_seq_length': MSL,
                'short_seq_prob': 0.0,
                'output_name': 'examples',
                'files_per_record': 5000,
                'write_in_batch': True,
                'write_remainder': True,
                'resume_from_checkpoint': False,
                'display_pbar': True,
                'seed': 0,
            },
            'dataset': {
                'use_ftfy': True,
                'ftfy_normalizer': 'NFC',
                'wikitext_detokenize': False,
                'prompt_key': 'prompt',
                'completion_key': 'completion',
                'eos_after_prompt': False,
            },
        }

        config_file = os.path.join(
            output_dir, dataset_name, "data_processing.yaml"
        )
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        dataset_creator_path = os.path.join(
            args.mz_path,
            'cerebras/modelzoo/data_preparation/nlp/hdf5_preprocessing/create_hdf5_dataset.py',
        )

        command = f"python {dataset_creator_path} Summarization_VSL --params {config_file}"
        subprocess.run(command, shell=True)

        print_h5_stats(hdf5_output_dir, output_file)

    # Process Numina Math:
    os.makedirs(os.path.join(output_dir, "numina_math_cot"))
    hdf5_dir = "with_vsl_hdf5"
    hdf5_output_dir = os.path.join(output_dir, "numina_math_cot", hdf5_dir)
    dataset_creator_path = os.path.join(
        args.mz_path,
        'cerebras/modelzoo/data_preparation/data_preprocessing/preprocess_data.py',
    )
    command = f"python {dataset_creator_path} --config cerebras_datasets/numina_math_cot.yaml"
    subprocess.run(command, shell=True)
    print_h5_stats(hdf5_output_dir)
