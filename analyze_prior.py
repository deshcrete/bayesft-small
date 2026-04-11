"""
Recover the base model's "prior" over persona styles.

Generates from the base model (no LoRA), scores with persona models
from each data scaling run, and runs EM to get the base model's
implicit style distribution.

Usage:
    python analyze_prior.py --run_dir results/data_scaling/n300
    python analyze_prior.py --all
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from scipy.special import logsumexp

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
    return (tok_lps * masks).sum(dim=-1).float().numpy()


@torch.no_grad()
def generate_texts(model, tokenizer, n_samples, device,
                   max_new_tokens=150, batch_size=32, temperature=0.8):
    model.eval()
    all_texts = []
    tokenizer.padding_side = "left"

    # Use simple story starters as prompts
    with open(STORY_DATA / "eval.json") as f:
        eval_data = json.load(f)

    prompts = []
    for ex in eval_data[:n_samples]:
        toks = tokenizer.encode(ex["text"], add_special_tokens=False)[:20]
        prompts.append(tokenizer.decode(toks))

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(prompts), batch_size)):
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
        print(f"    Batch {batch_idx+1}/{n_batches}: {len(all_texts)}/{len(prompts)} generated", flush=True)

    tokenizer.padding_side = "right"
    return all_texts


def em_mixture_weights(persona_logprobs, n_iter=200, tol=1e-10):
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


def analyze_run(run_dir, base_generations, tokenizer, device, batch_size=16):
    run_dir = Path(run_dir)

    with open(run_dir / "results.json") as f:
        results = json.load(f)
    n_personas = results["n_personas"]

    with open(STORY_DATA / "meta.json") as f:
        meta = json.load(f)
    persona_ids = [p["id"] for p in meta["personas"]][:n_personas]

    print(f"\n  Scoring base generations with persona models from {run_dir}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    gen_ds = TextDataset(base_generations, tokenizer, 512)
    gen_dl = DataLoader(gen_ds, batch_size=batch_size, shuffle=False)

    persona_lps = []
    for pid in persona_ids:
        ppath = run_dir / "models" / f"persona_{pid}"
        if not ppath.exists():
            print(f"    WARNING: {ppath} not found, skipping")
            continue
        print(f"    Scoring: {pid[:40]}", flush=True)
        lora_model = PeftModel.from_pretrained(base_model, ppath).to(device)
        lps = compute_seq_logprobs(lora_model, gen_dl, device)
        persona_lps.append(lps)
        # Unload LoRA to reuse base model cleanly
        lora_model = lora_model.unload()
        del lora_model
        torch.cuda.empty_cache()

    del base_model
    torch.cuda.empty_cache()

    persona_lps = np.stack(persona_lps)

    # EM
    w_prior, ll_hist = em_mixture_weights(persona_lps)

    # Compare to mixture EM weights
    w_mix = np.array(results["em_generations"]["weights"])

    # Correlation diagnostics
    corr = np.corrcoef(persona_lps)
    offdiag = corr[np.triu_indices(n_personas, k=1)]

    print(f"\n  Prior (base model) weights:")
    for pid, w in zip(persona_ids, w_prior):
        print(f"    {pid[:44]:44s} {w:.4f}")
    print(f"  EM converged in {len(ll_hist)} iterations")
    print(f"  Persona corr on base generations: mean={offdiag.mean():.4f}, min={offdiag.min():.4f}")

    print(f"\n  Comparison: prior vs mixture vs uniform")
    print(f"    {'Persona':<44s} {'Prior':>7s} {'Mixture':>8s} {'Uniform':>8s}")
    print(f"    {'-'*70}")
    true_w = 1.0 / n_personas
    for i, pid in enumerate(persona_ids):
        print(f"    {pid[:44]:44s} {w_prior[i]:>7.3f} {w_mix[i]:>8.3f} {true_w:>8.3f}")

    # Is the mixture posterior "between" prior and uniform?
    # Measure: for each persona, does the mixture weight lie between prior and uniform?
    between_count = 0
    for i in range(n_personas):
        lo, hi = min(w_prior[i], true_w), max(w_prior[i], true_w)
        if lo <= w_mix[i] <= hi:
            between_count += 1
    print(f"\n  Mixture weight between prior and uniform: {between_count}/{n_personas}")

    # Correlation between prior and mixture weights
    from scipy.stats import spearmanr
    pearson = np.corrcoef(w_prior, w_mix)[0, 1]
    spear, p_spear = spearmanr(w_prior, w_mix)
    print(f"  Correlation (prior vs mixture weights): Pearson={pearson:+.4f}, Spearman={spear:+.4f} (p={p_spear:.3f})")

    analysis = {
        "run_dir": str(run_dir),
        "n_per_persona": results["n_per_persona"],
        "n_base_generations": len(base_generations),
        "prior_weights": w_prior.tolist(),
        "mixture_weights": w_mix.tolist(),
        "true_weights": [true_w] * n_personas,
        "persona_ids": persona_ids,
        "between_count": between_count,
        "correlation": {
            "pearson_prior_vs_mix": float(pearson),
            "spearman_prior_vs_mix": float(spear),
            "spearman_p": float(p_spear),
        },
        "gen_corr_mean": float(offdiag.mean()),
        "gen_corr_min": float(offdiag.min()),
        "em_iters": len(ll_hist),
    }
    with open(run_dir / "prior_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Saved -> {run_dir / 'prior_analysis.json'}")

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate from base model once, reuse across runs
    print("Loading base model for generation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    print(f"\nGenerating {args.n_samples} samples from base model (no LoRA)...")
    base_generations = generate_texts(
        base_model, tokenizer, args.n_samples, device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size * 2,  # base model is lighter
        temperature=args.temperature,
    )
    del base_model
    torch.cuda.empty_cache()

    print(f"Got {len(base_generations)} base generations")
    print(f"Sample: {base_generations[0][:200]}...")

    # Save base generations
    out_dir = Path("results/data_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "base_generations.json", "w") as f:
        json.dump(base_generations, f, indent=2)

    # Determine runs
    if args.all:
        run_dirs = sorted(out_dir.glob("n*"))
    elif args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        print("Specify --run_dir or --all")
        return

    all_analyses = []
    for rd in run_dirs:
        if not (rd / "results.json").exists():
            continue
        a = analyze_run(rd, base_generations, tokenizer, device, args.batch_size)
        all_analyses.append(a)

    # Summary
    if all_analyses:
        print(f"\n{'='*70}")
        print("PRIOR RECOVERY SUMMARY")
        print(f"{'='*70}")
        for a in all_analyses:
            n = a["n_per_persona"]
            print(f"\n  n={n} (persona models trained on {n} examples each):")
            print(f"    {'Persona':<44s} {'Prior':>7s} {'Mixture':>8s} {'Uniform':>8s}")
            print(f"    {'-'*70}")
            for i, pid in enumerate(a["persona_ids"]):
                print(f"    {pid[:44]:44s} {a['prior_weights'][i]:>7.3f} {a['mixture_weights'][i]:>8.3f} {a['true_weights'][i]:>8.3f}")
            print(f"    Prior-Mixture corr: Spearman={a['correlation']['spearman_prior_vs_mix']:+.3f} (p={a['correlation']['spearman_p']:.3f})")
            print(f"    Mixture between prior & uniform: {a['between_count']}/{len(a['persona_ids'])}")


if __name__ == "__main__":
    main()
