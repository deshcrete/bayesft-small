"""
Run the full BayesFT pipeline across multiple training data sizes.

For each data size N (per persona):
  1. Subsample N examples per persona for persona training
  2. Subsample N examples per persona for mixture training (from mixture split)
  3. Train N_PERSONAS persona LoRAs
  4. Train 1 mixture LoRA
  5. Generate completions from the mixture model
  6. Score generations with all persona models
  7. Run EM to recover mixture weights
  8. Also run EM on the mixture training data (oracle baseline)

Uses the story_dataset with SimpleStories-35M base model.

Usage:
    python run_data_scaling.py --sizes 50 100 200 500 1000
    python run_data_scaling.py --sizes 100 --n_samples 500
"""

import argparse
import json
import subprocess
import sys
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from scipy.special import logsumexp
from collections import Counter

BASE_MODEL = "SimpleStories/SimpleStories-35M"
STORY_DATA = Path("data/story_dataset")
N_PERSONAS = 6  # how many of the 12 personas to use


# ── Data helpers ──────────────────────────────────────────────────────

def load_story_data():
    """Load the full story dataset splits and persona list."""
    with open(STORY_DATA / "meta.json") as f:
        meta = json.load(f)
    persona_ids = [p["id"] for p in meta["personas"]][:N_PERSONAS]
    persona_set = set(persona_ids)

    with open(STORY_DATA / "train.json") as f:
        train_data = [e for e in json.load(f) if e["persona_id"] in persona_set]
    with open(STORY_DATA / "mixture.json") as f:
        mixture_data = [e for e in json.load(f) if e["persona_id"] in persona_set]
    with open(STORY_DATA / "eval.json") as f:
        eval_data = [e for e in json.load(f) if e["persona_id"] in persona_set]

    return persona_ids, train_data, mixture_data, eval_data


def subsample_per_persona(data, persona_ids, n_per_persona, seed=42):
    """Take n_per_persona examples from each persona."""
    rng = np.random.RandomState(seed)
    by_persona = {pid: [] for pid in persona_ids}
    for ex in data:
        pid = ex["persona_id"]
        if pid in by_persona:
            by_persona[pid].append(ex)

    sampled = []
    for pid in persona_ids:
        pool = by_persona[pid]
        n = min(n_per_persona, len(pool))
        idxs = rng.choice(len(pool), size=n, replace=False)
        sampled.extend([pool[i] for i in idxs])
    return sampled


def write_persona_files(data, persona_ids, out_dir):
    """Write per-persona JSON files for training."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_persona = {pid: [] for pid in persona_ids}
    for ex in data:
        by_persona[ex["persona_id"]].append(ex)
    for pid in persona_ids:
        with open(out_dir / f"{pid}.json", "w") as f:
            json.dump(by_persona[pid], f)
    return by_persona


def write_mixture_file(data, out_dir):
    """Write mixture JSON file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "mixture.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ── Training ──────────────────────────────────────────────────────────

def train_persona(persona_id, data_file, output_dir, args):
    """Train one persona LoRA."""
    cmd = [
        sys.executable, "finetune.py",
        "--model_name", BASE_MODEL,
        "--dataset_name", "json",
        "--data_files", str(data_file),
        "--dataset_split", "train",
        "--text_column", "text",
        "--output_dir", str(output_dir),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--num_train_epochs", str(args.epochs),
        "--learning_rate", str(args.lr),
        "--per_device_train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--max_seq_length", str(args.max_seq_length),
        "--save_steps", "9999",  # don't save intermediate checkpoints
    ]
    result = subprocess.run(cmd, capture_output=not args.verbose)
    if result.returncode != 0:
        if not args.verbose and result.stderr:
            print(result.stderr.decode()[-2000:])
        raise RuntimeError(f"Training failed for {persona_id}")


def train_mixture(data_file, output_dir, args):
    """Train the mixture LoRA."""
    cmd = [
        sys.executable, "finetune.py",
        "--model_name", BASE_MODEL,
        "--dataset_name", "json",
        "--data_files", str(data_file),
        "--dataset_split", "train",
        "--text_column", "text",
        "--output_dir", str(output_dir),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--num_train_epochs", str(args.epochs),
        "--learning_rate", str(args.lr),
        "--per_device_train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--max_seq_length", str(args.max_seq_length),
        "--save_steps", "9999",
    ]
    result = subprocess.run(cmd, capture_output=not args.verbose)
    if result.returncode != 0:
        if not args.verbose and result.stderr:
            print(result.stderr.decode()[-2000:])
        raise RuntimeError("Mixture training failed")


# ── Generation & scoring ──────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


@torch.no_grad()
def compute_seq_logprobs(model, dataloader, device):
    """Compute sequence log-probs for each example."""
    model.eval()
    all_lps, all_masks = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]
        lp = F.log_softmax(shift_logits, dim=-1)
        tok_lp = lp.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1) * shift_mask
        all_lps.append(tok_lp.cpu())
        all_masks.append(shift_mask.cpu())
    tok_lps = torch.cat(all_lps)
    masks = torch.cat(all_masks)
    return (tok_lps * masks).sum(dim=-1).float().numpy()


@torch.no_grad()
def generate_texts(model, tokenizer, prompts, device,
                   max_new_tokens=150, batch_size=32, temperature=0.8):
    """Generate completions from model given prompts."""
    model.eval()
    all_texts = []
    tokenizer.padding_side = "left"

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            if len(text.strip()) > 10:
                all_texts.append(text)

    tokenizer.padding_side = "right"
    return all_texts


def em_mixture_weights(persona_logprobs, n_iter=200, tol=1e-10):
    """EM for mixture weights. persona_logprobs: (n_personas, n_examples)."""
    n_p, n_ex = persona_logprobs.shape
    w = np.ones(n_p) / n_p
    ll_history = []

    for it in range(n_iter):
        log_weighted = persona_logprobs + np.log(w + 1e-300)[:, None]
        log_denom = logsumexp(log_weighted, axis=0)
        log_r = log_weighted - log_denom[None, :]
        r = np.exp(log_r)
        ll = log_denom.sum()
        ll_history.append(ll)
        w_new = r.mean(axis=1)
        w_new = np.clip(w_new, 1e-15, None)
        w_new /= w_new.sum()
        if it > 0 and abs(ll_history[-1] - ll_history[-2]) < tol:
            break
        w = w_new

    return w, ll_history


# ── Main pipeline ─────────────────────────────────────────────────────

def run_one_size(n_per_persona, persona_ids, train_data, mixture_data,
                 eval_data, args):
    """Run the full pipeline for one data size. Returns results dict."""
    n_personas = len(persona_ids)
    true_w = np.ones(n_personas) / n_personas
    run_dir = Path(args.output_base) / f"n{n_per_persona}"
    data_dir = run_dir / "data"
    models_dir = run_dir / "models"

    print(f"\n{'='*70}")
    print(f"  DATA SIZE: {n_per_persona} per persona ({n_per_persona * n_personas} total)")
    print(f"  Output: {run_dir}")
    print(f"{'='*70}")

    # ── 1. Subsample data ──
    print(f"\n[1/6] Subsampling data...")
    persona_train = subsample_per_persona(train_data, persona_ids,
                                          n_per_persona, seed=args.seed)
    persona_by_id = write_persona_files(persona_train, persona_ids, data_dir)

    # For the mixture, take n_per_persona from each persona in the mixture split
    mix_sub = subsample_per_persona(mixture_data, persona_ids,
                                    n_per_persona, seed=args.seed + 1)
    mix_path = write_mixture_file(mix_sub, data_dir)

    for pid in persona_ids:
        n = len(persona_by_id[pid])
        print(f"  {pid}: {n} train, ", end="")
    mix_counts = Counter(e["persona_id"] for e in mix_sub)
    print(f"\n  Mixture: {len(mix_sub)} total ({dict(mix_counts)})")

    # ── 2. Train persona LoRAs ──
    print(f"\n[2/6] Training {n_personas} persona LoRAs...")
    for pid in persona_ids:
        pdata = data_dir / f"{pid}.json"
        pout = models_dir / f"persona_{pid}"
        if pout.exists() and not args.retrain:
            print(f"  {pid}: already exists, skipping")
            continue
        print(f"  Training: {pid} ({len(persona_by_id[pid])} examples)")
        train_persona(pid, pdata, pout, args)

    # ── 3. Train mixture LoRA ──
    print(f"\n[3/6] Training mixture LoRA...")
    mix_out = models_dir / "mixture"
    if mix_out.exists() and not args.retrain:
        print("  Already exists, skipping")
    else:
        print(f"  Training on {len(mix_sub)} examples")
        train_mixture(mix_path, mix_out, args)

    # ── 4. Generate from mixture model ──
    print(f"\n[4/6] Generating {args.n_samples} samples from mixture model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    # Use eval prompts — take the first few tokens of each eval example as prompt
    prompts = []
    for ex in eval_data[:args.n_samples]:
        # Use first ~20 tokens as prompt to condition generation
        toks = tokenizer.encode(ex["text"], add_special_tokens=False)[:20]
        prompts.append(tokenizer.decode(toks))

    mix_model = PeftModel.from_pretrained(base_model, mix_out).to(device)
    generated = generate_texts(
        mix_model, tokenizer, prompts, device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch_size,
        temperature=args.temperature,
    )
    del mix_model
    torch.cuda.empty_cache()
    print(f"  Got {len(generated)} generations")

    # Save generations
    gen_path = run_dir / "generations.json"
    with open(gen_path, "w") as f:
        json.dump(generated, f, indent=2)

    # ── 5. Score with persona models ──
    print(f"\n[5/6] Scoring generations with {n_personas} persona models...")
    gen_ds = TextDataset(generated, tokenizer, args.max_seq_length)
    gen_dl = DataLoader(gen_ds, batch_size=args.score_batch_size, shuffle=False)

    # Also score the mixture training data (oracle baseline)
    mix_texts = [e["text"] for e in mix_sub]
    mix_ds = TextDataset(mix_texts, tokenizer, args.max_seq_length)
    mix_dl = DataLoader(mix_ds, batch_size=args.score_batch_size, shuffle=False)

    gen_lps = []   # (n_personas, n_generated)
    train_lps = [] # (n_personas, n_mix_train)

    for pid in persona_ids:
        ppath = models_dir / f"persona_{pid}"
        print(f"  Scoring: {pid}")
        lora_model = PeftModel.from_pretrained(base_model, ppath).to(device)
        gen_lps.append(compute_seq_logprobs(lora_model, gen_dl, device))
        train_lps.append(compute_seq_logprobs(lora_model, mix_dl, device))
        del lora_model
        torch.cuda.empty_cache()

    gen_lps = np.stack(gen_lps)
    train_lps = np.stack(train_lps)

    del base_model
    torch.cuda.empty_cache()

    # ── 6. EM inference ──
    print(f"\n[6/6] Running EM...")

    # On generations
    w_gen, ll_gen = em_mixture_weights(gen_lps)
    err_gen = np.max(np.abs(w_gen - true_w))

    # On training data (oracle)
    w_train, ll_train = em_mixture_weights(train_lps)
    err_train = np.max(np.abs(w_train - true_w))

    # Diagnostics
    corr_gen = np.corrcoef(gen_lps)
    offdiag_gen = corr_gen[np.triu_indices(n_personas, k=1)]
    corr_train = np.corrcoef(train_lps)
    offdiag_train = corr_train[np.triu_indices(n_personas, k=1)]

    results = {
        "n_per_persona": n_per_persona,
        "n_personas": n_personas,
        "n_generated": len(generated),
        "n_mix_train": len(mix_sub),
        "em_generations": {
            "weights": w_gen.tolist(),
            "max_error": float(err_gen),
            "iters": len(ll_gen),
        },
        "em_training_data": {
            "weights": w_train.tolist(),
            "max_error": float(err_train),
            "iters": len(ll_train),
        },
        "diagnostics": {
            "gen_mean_corr": float(offdiag_gen.mean()),
            "gen_min_corr": float(offdiag_gen.min()),
            "train_mean_corr": float(offdiag_train.mean()),
            "train_min_corr": float(offdiag_train.min()),
        },
    }

    # Print summary
    print(f"\n  ── Results (n={n_per_persona}) ──")
    print(f"  EM on generations ({len(generated)} samples):")
    print(f"    Max error: {err_gen:.4f}")
    print(f"    Weights: {np.array2string(w_gen, precision=3)}")
    print(f"    Persona corr: mean={offdiag_gen.mean():.4f}, min={offdiag_gen.min():.4f}")
    print(f"  EM on training data ({len(mix_sub)} samples):")
    print(f"    Max error: {err_train:.4f}")
    print(f"    Weights: {np.array2string(w_train, precision=3)}")
    print(f"    Persona corr: mean={offdiag_train.mean():.4f}, min={offdiag_train.min():.4f}")

    # Save results
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500, 1000],
                        help="Training examples per persona to sweep")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of generations from mixture model")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output_base", type=str, default="results/data_scaling")

    # Training args
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=512)

    # Generation / scoring
    parser.add_argument("--gen_batch_size", type=int, default=32)
    parser.add_argument("--score_batch_size", type=int, default=16)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain", action="store_true", help="Retrain even if models exist")
    parser.add_argument("--verbose", action="store_true", help="Show training output")
    args = parser.parse_args()

    # Load data
    print("Loading story dataset...")
    persona_ids, train_data, mixture_data, eval_data = load_story_data()
    print(f"  {len(persona_ids)} personas, {len(train_data)} train, "
          f"{len(mixture_data)} mixture, {len(eval_data)} eval")

    # Run for each size
    all_results = []
    for n in sorted(args.sizes):
        max_available = min(
            Counter(e["persona_id"] for e in train_data).most_common()[-1][1],
            Counter(e["persona_id"] for e in mixture_data).most_common()[-1][1],
        )
        if n > max_available:
            print(f"\nSkipping n={n}: only {max_available} examples per persona available")
            continue
        res = run_one_size(n, persona_ids, train_data, mixture_data,
                           eval_data, args)
        all_results.append(res)

    # ── Final summary ──
    print(f"\n{'='*70}")
    print("SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"{'N/persona':>10} {'N total':>8} {'Gen Err':>8} {'Train Err':>10} "
          f"{'Gen Corr':>9} {'Train Corr':>11}")
    print("-" * 70)
    for r in all_results:
        n = r["n_per_persona"]
        nt = n * r["n_personas"]
        ge = r["em_generations"]["max_error"]
        te = r["em_training_data"]["max_error"]
        gc = r["diagnostics"]["gen_mean_corr"]
        tc = r["diagnostics"]["train_mean_corr"]
        print(f"{n:>10} {nt:>8} {ge:>8.4f} {te:>10.4f} {gc:>9.4f} {tc:>11.4f}")

    # Save combined results
    out_path = Path(args.output_base) / "summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary -> {out_path}")


if __name__ == "__main__":
    main()
