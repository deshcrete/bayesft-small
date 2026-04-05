"""
Compute per-token logprobs for eval data under each model
(base, 6 personas, mixture).

Saves per-token logprob vectors (masked by attention) and token counts.
Sequence-level and delta logprobs can be derived in post-processing.

Usage:
    python compute_logprobs.py --data_dir data/ --models_dir models/ --output results/logprobs.pt
    python compute_logprobs.py --data_dir data/ --models_dir models/ --batch_size 32
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


class StoryDataset(Dataset):
    def __init__(self, stories, tokenizer, max_length=512):
        self.encodings = tokenizer(
            stories,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


@torch.no_grad()
def compute_logprobs_for_model(model, dataloader, device):
    """Returns per-token logprobs (padded) and token counts per example."""
    model.eval()
    all_token_lps = []
    all_masks = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_lps = token_lps * shift_mask

        all_token_lps.append(token_lps.cpu())
        all_masks.append(shift_mask.cpu())

    return torch.cat(all_token_lps), torch.cat(all_masks)


def load_base_model(device):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    return model, tokenizer


def load_lora_model(base_model, lora_path, device):
    model = PeftModel.from_pretrained(base_model, lora_path).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--models_dir", type=str, default="models/")
    parser.add_argument("--output", type=str, default="results/logprobs_diff.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load eval data
    eval_path = Path(args.data_dir) / "eval.json"
    with open(eval_path) as f:
        eval_data = json.load(f)
    stories = [e["text"] for e in eval_data]
    indices = [e["index"] for e in eval_data]
    print(f"Eval stories: {len(stories)}")

    # Load base model + tokenizer
    print(f"\nLoading base model: {BASE_MODEL}")
    base_model, tokenizer = load_base_model(device)

    dataset = StoryDataset(stories, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # {model_name: (token_logprobs, mask)} — mask is shared but we store once
    model_logprobs = {}

    # Base model logprobs
    print("Computing logprobs: base")
    token_lps, mask = compute_logprobs_for_model(base_model, dataloader, device)
    model_logprobs["base"] = token_lps
    token_counts = mask.sum(dim=-1)  # (n_examples,)

    # Discover persona models from metadata or filesystem
    models_dir = Path(args.models_dir)
    meta_path = Path(args.data_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        persona_dirs = [f"persona_{slug}" for slug in meta["slugs"].values()]
    else:
        persona_dirs = sorted(
            d.name for d in models_dir.iterdir()
            if d.is_dir() and d.name.startswith("persona_")
        )

    for persona_dir in persona_dirs:
        lora_path = models_dir / persona_dir
        if not lora_path.exists():
            print(f"Skipping {persona_dir}: not found")
            continue

        print(f"Computing logprobs: {persona_dir}")
        lora_model = load_lora_model(base_model, lora_path, device)
        token_lps, _ = compute_logprobs_for_model(lora_model, dataloader, device)
        model_logprobs[persona_dir] = token_lps
        del lora_model
        torch.cuda.empty_cache()

    # Mixture model
    mixture_path = models_dir / "mixture_uniform_diff"
    if mixture_path.exists():
        print("Computing logprobs: mixture_uniform")
        lora_model = load_lora_model(base_model, mixture_path, device)
        token_lps, _ = compute_logprobs_for_model(lora_model, dataloader, device)
        model_logprobs["mixture_uniform"] = token_lps
        del lora_model
        torch.cuda.empty_cache()
    else:
        print("Skipping mixture: not found")

    # Save as a single .pt file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "indices": indices,
        "mask": mask,                   # (n_examples, seq_len-1) — shared across models
        "token_counts": token_counts,   # (n_examples,)
        "token_logprobs": model_logprobs,  # {model_name: (n_examples, seq_len-1)}
    }, out_path)
    print(f"\nSaved logprobs -> {out_path}")
    print(f"Models: {list(model_logprobs.keys())}")
    print(f"Shape: {list(model_logprobs.values())[0].shape}")


if __name__ == "__main__":
    main()
