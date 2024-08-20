import argparse
import logging
import os
import shutil
import subprocess

from transformers import AutoModelForCausalLM, AutoTokenizer


def create_dirs_overwrite(path):
    # Get the parent directory and the last directory name
    parent_dir = os.path.dirname(path)
    last_dir = os.path.basename(path)

    # Path to the target directory to remove and recreate
    target_dir = os.path.join(parent_dir, last_dir)

    # If the last directory exists, delete it
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Recreate the last directory
    os.makedirs(target_dir)


def save_llm_hf_model():
    print("Downloading Meta-Llama-3-8B from HF")
    create_dirs_overwrite("checkpoints/HF/Meta-Llama-3-8B")
    llama_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B"
    )
    llama_model.save_pretrained("checkpoints/HF/Meta-Llama-3-8B/")


def convert_llm_hf_to_cs(mz_path):
    if not os.path.exists("checkpoints/HF/Meta-Llama-3-8B/"):
        save_llm_hf_model()

    create_dirs_overwrite("checkpoints/CS/Meta-Llama-3-8B/")
    command = " ".join(
        [
            "python",
            os.path.join(
                mz_path, "cerebras/modelzoo/tools/convert_checkpoint.py"
            ),
            "convert",
            "--model",
            "llama",
            "--src-fmt",
            "hf",
            "--tgt-fmt",
            "cs-2.3",
            "--output-dir",
            "checkpoints/CS/Meta-Llama-3-8B/",
            "--config",
            "checkpoints/HF/Meta-Llama-3-8B/config.json",
            "checkpoints/HF/Meta-Llama-3-8B/model.safetensors.index.json",
        ]
    )
    subprocess.run(command, shell=True)


def convert_llm_cs_to_hf(mz_path, checkpoint_path):
    create_dirs_overwrite("checkpoints/HF/Cerebras-Llama-3-8B/")
    parent_dir = os.path.dirname(checkpoint_path)

    command = " ".join(
        [
            "python",
            os.path.join(
                mz_path, "cerebras/modelzoo/tools/convert_checkpoint.py"
            ),
            "convert",
            "--model",
            "llama",
            "--src-fmt",
            "cs-2.3",
            "--tgt-fmt",
            "hf",
            "--output-dir",
            "checkpoints/HF/Cerebras-Llama3-DocChat-8B/",
            os.path.join(parent_dir, "train", "params_train.yaml"),
            checkpoint_path,
        ]
    )
    subprocess.run(command, shell=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct"
    )
    tokenizer.save_pretrained(
        "checkpoints/HF/Cerebras-Llama3-DocChat-8B/"
    )


def main(args):
    assert (
        args.mz_path is not None
    ), "mz_path must be provided or implicitly set via PYTHONPATH"
    if args.direction == "hf_to_cs":
        logging.info("Converting LLM HF checkpoint of CS for training...")
        convert_llm_hf_to_cs(args.mz_path)
    else:
        logging.info(
            "Converting Trained LLM CS checkpoint of HF for evaluation..."
        )
        assert (
            args.llm_trained_checkpoint_path is not None
        ), "llm_trained_checkpoint_path must be provided..."
        convert_llm_cs_to_hf(args.mz_path, args.llm_trained_checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cerebras DocChat LLM Checkpoint Conversion"
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=[
            "cs_to_hf",
            "hf_to_cs",
        ],
        required=True,
        help="Choose LLM checkpoint conversion direction",
    )
    parser.add_argument(
        "--mz_path",
        type=str,
        default=os.environ.get("PYTHONPATH"),
        help="Path of modelzoo",
    )
    parser.add_argument(
        "--llm_trained_checkpoint_path",
        type=str,
        help='Trained checkpoint path of LLM model',
    )
    args = parser.parse_args()
    main(args)
