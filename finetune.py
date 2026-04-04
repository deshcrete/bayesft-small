"""
Fine-tune a Hugging Face model on a Hugging Face dataset.

Usage:
    python finetune.py --model_name meta-llama/Llama-3.2-1B \
                       --dataset_name tatsu-lab/alpaca \
                       --output_dir ./output

    # With a specific dataset split and text column:
    python finetune.py --model_name gpt2 \
                       --dataset_name wikitext --dataset_config wikitext-2-raw-v1 \
                       --text_column text \
                       --output_dir ./output

    # Instruction-tuning (chat/instruct format):
    python finetune.py --model_name meta-llama/Llama-3.2-1B \
                       --dataset_name tatsu-lab/alpaca \
                       --instruct \
                       --output_dir ./output
"""

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a HF model on a HF dataset")

    # Model / dataset
    parser.add_argument("--model_name", type=str, required=True, help="HF model id (e.g. gpt2, meta-llama/Llama-3.2-1B)")
    parser.add_argument("--dataset_name", type=str, required=True, help="HF dataset id or 'json' for local files")
    parser.add_argument("--data_files", type=str, default=None, help="Path to local data file (used with --dataset_name json)")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config/subset name")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--text_column", type=str, default=None, help="Column containing text for CLM. Auto-detected if not set.")
    parser.add_argument("--instruct", action="store_true", help="Format as instruction-tuning (expects instruction/input/output columns)")

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA (default: True)")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA and do full fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    # Training hyperparameters
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=None, help="Use bf16 (auto-detected if not set)")
    parser.add_argument("--fp16", action="store_true", default=None, help="Use fp16")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples (useful for testing)")
    parser.add_argument("--seed", type=int, default=42)

    # Misc
    parser.add_argument("--push_to_hub", action="store_true", help="Push final model to HF Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="HF Hub model id for push")

    return parser.parse_args()


def format_instruct(example):
    """Format instruction/input/output columns into a single text prompt."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if inp:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": text}


def detect_text_column(dataset):
    """Try to auto-detect which column has the training text."""
    columns = dataset.column_names
    for candidate in ["text", "content", "sentence", "document"]:
        if candidate in columns:
            return candidate
    # Fall back to first string column
    for col in columns:
        if dataset.features[col].dtype == "string":
            return col
    raise ValueError(
        f"Could not auto-detect text column from {columns}. "
        "Pass --text_column explicitly."
    )


def main():
    args = parse_args()

    # --- Load dataset ---
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        data_files=args.data_files,
        split=args.dataset_split,
    )

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # --- Format dataset ---
    if args.instruct:
        print("Formatting as instruction-tuning dataset")
        dataset = dataset.map(format_instruct, remove_columns=dataset.column_names)
        text_column = "text"
    else:
        text_column = args.text_column or detect_text_column(dataset)
        print(f"Using text column: {text_column}")
        # Rename to "text" if needed (SFTTrainer expects "text" by default)
        if text_column != "text":
            dataset = dataset.rename_column(text_column, "text")

    print(f"Dataset size: {len(dataset)} examples")

    # --- Load model & tokenizer ---
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # --- LoRA config ---
    peft_config = None
    if args.use_lora and not args.no_lora:
        print(f"Using LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    # --- Auto-detect precision ---
    if args.bf16 is None and args.fp16 is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            args.bf16 = True
            args.fp16 = False
        elif torch.cuda.is_available():
            args.bf16 = False
            args.fp16 = True
        else:
            args.bf16 = False
            args.fp16 = False

    # --- Training config ---
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16 or False,
        fp16=args.fp16 or False,
        optim="adamw_torch",
        seed=args.seed,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        dataset_text_field="text",
    )

    # --- Train ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()

    # --- Save ---
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print("Pushing to Hub...")
        trainer.push_to_hub()

    print("Done!")


if __name__ == "__main__":
    main()
