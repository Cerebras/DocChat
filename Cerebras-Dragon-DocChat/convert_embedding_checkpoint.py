import argparse
import os
import shutil
import sys

from transformers import AutoModel, AutoTokenizer


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


def save_embedding_hf_model():
    print("Downloading dragon+ model from HF")
    dragon_query_encoder = AutoModel.from_pretrained(
        'facebook/dragon-plus-query-encoder'
    )
    dragon_ctx_encoder = AutoModel.from_pretrained(
        'facebook/dragon-plus-context-encoder'
    )

    create_dirs_overwrite("checkpoints/HF/facebook_dragon/q_encoder")
    create_dirs_overwrite("checkpoints/HF/facebook_dragon/ctx_encoder")
    dragon_query_encoder.save_pretrained(
        "checkpoints/HF/facebook_dragon/q_encoder/", safe_serialization=False
    )
    dragon_ctx_encoder.save_pretrained(
        "checkpoints/HF/facebook_dragon/ctx_encoder/", safe_serialization=False
    )


def main(args):
    from dragon_converter import Converter_DragonModel_HF_CS23

    from cerebras.modelzoo.tools.checkpoint_converters.registry import (
        converters,
    )

    # Add dpr to converter registry (was missing in 2.3.0)
    converters["dragon"] = [
        Converter_DragonModel_HF_CS23,
    ]

    from cerebras.modelzoo.tools.convert_checkpoint import (
        convert_checkpoint_from_file,
    )

    if args.direction == "cs_to_hf":
        assert (
            args.embedding_trained_checkpoint_path is not None
        ), "embedding_trained_checkpoint_path must be provided..."
        assert args.embedding_trained_checkpoint_path.endswith(".mdl"), (
            "embedding_trained_checkpoint_path must be a checkpoint file "
            "(checkpoint_<step number>.mdl)"
        )

        checkpoint_path = args.embedding_trained_checkpoint_path
        parent_dir = os.path.dirname(checkpoint_path)
        config_file = os.path.join(parent_dir, "train", "params_train.yaml")
        outputdir = "checkpoints/HF/cerebras_dragon/"
        create_dirs_overwrite("checkpoints/HF/cerebras_dragon/")
    else:
        if not os.path.exists("checkpoints/HF/facebook_dragon/"):
            save_embedding_hf_model()
        checkpoint_path = "checkpoints/HF/facebook_dragon/"
        config_file = checkpoint_path
        outputdir = "checkpoints/CS/facebook_dragon/"
        create_dirs_overwrite("checkpoints/CS/facebook_dragon/")

    formats = ["hf", "cs-2.3"]
    if args.direction == "cs_to_hf":
        formats.reverse()

    print(f"Running converter ({args.direction})")

    (
        checkpoint_output_path,
        config_output_path,
    ) = convert_checkpoint_from_file(
        "dragon",
        *formats,
        checkpoint_path,
        config_file,
        outputdir=outputdir,
        no_progress_bar=False,
    )

    if checkpoint_output_path is None or config_output_path is None:
        print("\nConversion failed.")
        sys.exit(1)
    else:
        print("Checkpoint saved to {}".format(checkpoint_output_path))
        print("Config saved to {}".format(config_output_path))

        if args.direction == "cs_to_hf":
            print("Downloading dragon+ tokenizer to converted model's dir")
            tokenizer = AutoTokenizer.from_pretrained(
                "facebook/dragon-plus-query-encoder"
            )
            tokenizer.save_pretrained(
                "checkpoints/HF/cerebras_dragon/q_encoder/"
            )
            tokenizer.save_pretrained(
                "checkpoints/HF/cerebras_dragon/ctx_encoder/"
            )

        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cerebras DocChat EmbeddingCheckpoint Conversion"
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["hf_to_cs", "cs_to_hf"],
        required=True,
        help="Choose Embedding checkpoint conversion direction",
    )
    parser.add_argument(
        "--embedding_trained_checkpoint_path",
        type=str,
        help='Trained checkpoint path of embedding model',
    )
    args = parser.parse_args()
    main(args)
