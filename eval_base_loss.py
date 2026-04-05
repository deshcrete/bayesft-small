"""
Evaluate base model loss on a dataset to check if fine-tuning is learning anything.

Usage:
    python eval_base_loss.py --data data/mixture_uniform.json
    python eval_base_loss.py --data data/a_poet.json
"""

import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to JSON data file")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    dataset = load_dataset("json", data_files=args.data, split="train")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    losses = []
    accuracies = []

    with torch.no_grad():
        for i, example in enumerate(dataset):
            text = example[args.text_column]
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=args.max_seq_length).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())

            # Token accuracy: how often does argmax prediction match the target?
            logits = outputs.logits[:, :-1, :]  # shift
            targets = inputs["input_ids"][:, 1:]
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()
            accuracies.append(acc)

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(dataset)}] running loss={np.mean(losses):.4f} acc={np.mean(accuracies):.4f}")

    print(f"\nDataset: {args.data}")
    print(f"Examples: {len(losses)}")
    print(f"Base model loss:     {np.mean(losses):.4f} (std={np.std(losses):.4f})")
    print(f"Base model accuracy: {np.mean(accuracies):.4f} (std={np.std(accuracies):.4f})")


if __name__ == "__main__":
    main()
