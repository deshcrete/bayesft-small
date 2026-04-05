"""
Train a mixture LoRA adapter on the uniform mixture dataset.

Usage:
    python train_mixture.py --data_dir data/ --output_dir models/
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="models/")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=512)
    args = parser.parse_args()

    data_file = Path(args.data_dir) / "mixture_uniform.json"
    if not data_file.exists():
        print(f"Mixture data not found: {data_file}")
        sys.exit(1)

    model_out = Path(args.output_dir) / "mixture_uniform_diff"

    print(f"Training mixture model")
    print(f"  Data: {data_file}")
    print(f"  Output: {model_out}")

    cmd = [
        sys.executable, "finetune.py",
        "--model_name", BASE_MODEL,
        "--dataset_name", "json",
        "--data_files", str(data_file),
        "--dataset_split", "train",
        "--text_column", "text",
        "--output_dir", str(model_out),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--num_train_epochs", str(args.epochs),
        "--learning_rate", str(args.lr),
        "--per_device_train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--max_seq_length", str(args.max_seq_length),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("FAILED")
        sys.exit(1)

    print("Mixture model trained.")


if __name__ == "__main__":
    main()
