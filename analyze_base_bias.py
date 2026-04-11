"""
Analyze whether the base model's pretrained preferences explain EM weight bias.

Scores existing generations and per-persona eval data with the base model (no LoRA)
to check if the base model prefers certain persona styles, and whether that correlates
with which personas get overweighted by EM.

No training needed — just reads existing results and runs forward passes.

Usage:
    python analyze_base_bias.py --run_dir results/data_scaling/n300
    python analyze_base_bias.py --all  # run on all sizes
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

BASE_MODEL = "SimpleStories/SimpleStories-35M"
STORY_DATA = Path("data/story_dataset")


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
    seq_lps = (tok_lps * masks).sum(dim=-1).float().numpy()
    tok_counts = masks.sum(dim=-1).float().numpy()
    return seq_lps, tok_counts


def analyze_run(run_dir, base_model, tokenizer, device, batch_size=16):
    """Analyze base model bias for one run."""
    run_dir = Path(run_dir)
    print(f"\n{'='*70}")
    print(f"  Analyzing: {run_dir}")
    print(f"{'='*70}")

    # Load results
    with open(run_dir / "results.json") as f:
        results = json.load(f)

    em_weights = np.array(results["em_generations"]["weights"])
    n_personas = results["n_personas"]
    true_w = np.ones(n_personas) / n_personas
    weight_bias = em_weights - true_w  # positive = overweighted by EM

    # Load persona list
    with open(STORY_DATA / "meta.json") as f:
        meta = json.load(f)
    persona_ids = [p["id"] for p in meta["personas"]][:n_personas]
    persona_names = [p["name"] for p in meta["personas"]][:n_personas]

    # ── 1. Score generations with base model ──
    print("\n  [1/3] Scoring generations with base model...")
    with open(run_dir / "generations.json") as f:
        generations = json.load(f)

    gen_ds = TextDataset(generations, tokenizer, 512)
    gen_dl = DataLoader(gen_ds, batch_size=batch_size, shuffle=False)
    gen_base_lps, gen_tok_counts = compute_seq_logprobs(base_model, gen_dl, device)

    print(f"    Mean base logprob on generations: {gen_base_lps.mean():.1f}")
    print(f"    Mean per-token: {(gen_base_lps / gen_tok_counts).mean():.4f}")

    # ── 2. Score per-persona eval data with base model ──
    print("\n  [2/3] Scoring per-persona eval data with base model...")
    with open(STORY_DATA / "eval.json") as f:
        eval_data = json.load(f)

    persona_base_lps = {}
    for pid in persona_ids:
        texts = [e["text"] for e in eval_data if e["persona_id"] == pid]
        ds = TextDataset(texts, tokenizer, 512)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        lps, tcs = compute_seq_logprobs(base_model, dl, device)
        mean_per_tok = (lps / tcs).mean()
        persona_base_lps[pid] = {
            "mean_seq": float(lps.mean()),
            "mean_per_tok": float(mean_per_tok),
            "n": len(texts),
        }
        print(f"    {pid[:40]:42s} per-tok={mean_per_tok:.4f}  seq={lps.mean():.1f}  (n={len(texts)})")

    # ── 3. Correlation analysis ──
    print("\n  [3/3] Correlation between base preference and EM weight bias...")

    base_per_tok = np.array([persona_base_lps[pid]["mean_per_tok"] for pid in persona_ids])
    base_seq = np.array([persona_base_lps[pid]["mean_seq"] for pid in persona_ids])

    # Higher base logprob = base model prefers this style
    # Positive weight bias = EM overweights this persona
    corr_per_tok = np.corrcoef(base_per_tok, weight_bias)[0, 1]
    corr_seq = np.corrcoef(base_seq, weight_bias)[0, 1]

    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    spear_per_tok, p_per_tok = spearmanr(base_per_tok, weight_bias)
    spear_seq, p_seq = spearmanr(base_seq, weight_bias)

    print(f"\n    Pearson  (per-token LP vs weight bias): {corr_per_tok:+.4f}")
    print(f"    Pearson  (sequence LP vs weight bias):  {corr_seq:+.4f}")
    print(f"    Spearman (per-token LP vs weight bias): {spear_per_tok:+.4f}  (p={p_per_tok:.3f})")
    print(f"    Spearman (sequence LP vs weight bias):  {spear_seq:+.4f}  (p={p_seq:.3f})")

    # ── Summary table ──
    print(f"\n    {'Persona':<44s} {'Base LP/tok':>11s} {'EM Weight':>10s} {'Bias':>8s}")
    print(f"    {'-'*75}")
    order = np.argsort(base_per_tok)[::-1]  # best base LP first
    for i in order:
        pid = persona_ids[i]
        print(f"    {pid[:44]:44s} {base_per_tok[i]:>+11.4f} {em_weights[i]:>10.3f} {weight_bias[i]:>+8.3f}")

    # Save analysis
    analysis = {
        "run_dir": str(run_dir),
        "n_per_persona": results["n_per_persona"],
        "em_weights": em_weights.tolist(),
        "weight_bias": weight_bias.tolist(),
        "base_per_tok_lp": {pid: persona_base_lps[pid]["mean_per_tok"] for pid in persona_ids},
        "base_seq_lp": {pid: persona_base_lps[pid]["mean_seq"] for pid in persona_ids},
        "correlation": {
            "pearson_per_tok": float(corr_per_tok),
            "pearson_seq": float(corr_seq),
            "spearman_per_tok": float(spear_per_tok),
            "spearman_per_tok_p": float(p_per_tok),
            "spearman_seq": float(spear_seq),
            "spearman_seq_p": float(p_seq),
        },
    }
    with open(run_dir / "base_bias_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n    Saved -> {run_dir / 'base_bias_analysis.json'}")

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, help="Single run directory to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all runs in results/data_scaling/")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("Loading base model (no LoRA)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    if args.all:
        scaling_dir = Path("results/data_scaling")
        run_dirs = sorted(scaling_dir.glob("n*"))
        if not run_dirs:
            print(f"No runs found in {scaling_dir}")
            return
    elif args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        print("Specify --run_dir or --all")
        return

    all_analyses = []
    for rd in run_dirs:
        if not (rd / "results.json").exists():
            print(f"Skipping {rd}: no results.json")
            continue
        analysis = analyze_run(rd, base_model, tokenizer, device, args.batch_size)
        all_analyses.append(analysis)

    # ── Cross-size summary ──
    if len(all_analyses) > 1:
        print(f"\n{'='*70}")
        print("CROSS-SIZE SUMMARY: Base Model Preference vs EM Weight Bias")
        print(f"{'='*70}")
        print(f"  {'N/persona':>10s} {'Pearson':>10s} {'Spearman':>10s} {'p-value':>10s}")
        print(f"  {'-'*45}")
        for a in all_analyses:
            n = a["n_per_persona"]
            r = a["correlation"]["pearson_per_tok"]
            s = a["correlation"]["spearman_per_tok"]
            p = a["correlation"]["spearman_per_tok_p"]
            print(f"  {n:>10d} {r:>+10.4f} {s:>+10.4f} {p:>10.3f}")


if __name__ == "__main__":
    main()
