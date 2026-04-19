"""
Non-uniform mixture weights experiment.

Reuses existing per-persona LoRAs (persona training is independent of mixture
composition). Only the mixture LoRA is retrained, on a skewed composition of
the mixture split.

Weights are assigned to the persona list *in the order given by --personas*
(default: natural meta.json order). Total mixture size = sum of
(weight_i * total_mix), so no weight can exceed (pool_i / total_mix).

Also recovers the prior from cached base-model generations scored with the
same persona LoRAs, and applies the Bayes' rule correction
w_lik ∝ w_post / w_prior.

Usage:
    python run_nonuniform.py \\
        --weights 0.35 0.25 0.15 0.12 0.08 0.05 \\
        --personas simple_cheerful_dialogue_animals literary_melancholic_narration_nature \\
                   simple_tense_secondperson_adventure archaic_whimsical_narration_fantasy \\
                   blunt_factual_dialogue_everyday warm_nostalgic_firstperson_family \\
        --persona_models_dir results/data_scaling/n100/models \\
        --total_mix 600 \\
        --output_dir results/data_scaling/n100_nonuniform
"""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from collections import Counter

from run_data_scaling import (
    BASE_MODEL, STORY_DATA, N_PERSONAS,
    load_story_data, write_mixture_file, train_mixture,
    TextDataset, compute_seq_logprobs, generate_texts,
    em_mixture_weights,
)


DEFAULT_PERSONAS = [
    "simple_cheerful_dialogue_animals",
    "literary_melancholic_narration_nature",
    "simple_tense_secondperson_adventure",
    "archaic_whimsical_narration_fantasy",
    "blunt_factual_dialogue_everyday",
    "warm_nostalgic_firstperson_family",
]
BASE_GEN_PATH = Path("results/data_scaling/base_generations.json")


def subsample_nonuniform(data, persona_ids, counts, seed=42):
    """Take `counts[i]` examples from persona `persona_ids[i]`."""
    rng = np.random.RandomState(seed)
    by_persona = {pid: [] for pid in persona_ids}
    for ex in data:
        if ex["persona_id"] in by_persona:
            by_persona[ex["persona_id"]].append(ex)
    sampled = []
    for pid, n in zip(persona_ids, counts):
        pool = by_persona[pid]
        if n > len(pool):
            raise ValueError(f"{pid}: need {n}, only {len(pool)} available")
        idxs = rng.choice(len(pool), size=n, replace=False)
        sampled.extend([pool[i] for i in idxs])
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=float, nargs="+", required=True,
                        help="True mixture weights (must sum to 1)")
    parser.add_argument("--personas", type=str, nargs="+", default=DEFAULT_PERSONAS,
                        help="Persona IDs in the order matching --weights")
    parser.add_argument("--persona_models_dir", type=str, required=True,
                        help="Dir containing persona_{id}/ LoRA subdirs")
    parser.add_argument("--total_mix", type=int, required=True,
                        help="Total mixture examples (sum of per-persona counts)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--gen_batch_size", type=int, default=32)
    parser.add_argument("--score_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    persona_ids = list(args.personas)
    true_w = np.array(args.weights, dtype=float)
    assert len(true_w) == len(persona_ids) == N_PERSONAS, \
        f"Need {N_PERSONAS} weights and personas"
    assert abs(true_w.sum() - 1.0) < 1e-6, f"Weights must sum to 1, got {true_w.sum()}"

    counts = [int(round(w * args.total_mix)) for w in args.weights]
    diff = args.total_mix - sum(counts)
    counts[0] += diff  # fix any rounding drift on the largest weight
    print(f"Per-persona mixture counts: {counts} (total {sum(counts)})")

    persona_models_dir = Path(args.persona_models_dir)
    run_dir = Path(args.output_dir)
    data_dir = run_dir / "data"
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading story dataset...")
    # Filter the loaded data to the personas we actually care about (load_story_data
    # filters to the first N_PERSONAS from meta.json; our personas arg may reorder them)
    _, _, mixture_data, eval_data = load_story_data()
    chosen = set(persona_ids)
    mixture_data = [e for e in mixture_data if e["persona_id"] in chosen]
    eval_data = [e for e in eval_data if e["persona_id"] in chosen]
    print(f"  {len(persona_ids)} personas, {len(mixture_data)} mixture, {len(eval_data)} eval")

    for pid in persona_ids:
        pdir = persona_models_dir / f"persona_{pid}"
        if not pdir.exists():
            raise FileNotFoundError(f"Missing persona LoRA: {pdir}")

    if not BASE_GEN_PATH.exists():
        raise FileNotFoundError(f"Missing base generations: {BASE_GEN_PATH}")

    # ── 1. Subsample skewed mixture ──
    print(f"\n[1/5] Subsampling skewed mixture...")
    mix_sub = subsample_nonuniform(mixture_data, persona_ids, counts, seed=args.seed + 1)
    mix_path = write_mixture_file(mix_sub, data_dir)
    actual = Counter(e["persona_id"] for e in mix_sub)
    for pid, w in zip(persona_ids, true_w):
        print(f"  {pid}: {actual[pid]} (true w={w})")

    # ── 2. Train mixture LoRA ──
    print(f"\n[2/5] Training mixture LoRA on skewed data...")
    mix_out = models_dir / "mixture"
    if mix_out.exists() and not args.retrain:
        print("  Already exists, skipping")
    else:
        train_mixture(mix_path, mix_out, args)

    # ── 3. Generate from mixture model ──
    print(f"\n[3/5] Generating {args.n_samples} samples from mixture model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    prompts = []
    for ex in eval_data[:args.n_samples]:
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

    with open(run_dir / "generations.json", "w") as f:
        json.dump(generated, f, indent=2)

    # ── 4. Score generations, training data, and cached base generations ──
    print(f"\n[4/5] Scoring with {N_PERSONAS} persona models (reusing n=100)...")

    with open(BASE_GEN_PATH) as f:
        base_gen = json.load(f)

    gen_ds = TextDataset(generated, tokenizer, args.max_seq_length)
    gen_dl = DataLoader(gen_ds, batch_size=args.score_batch_size, shuffle=False)

    mix_texts = [e["text"] for e in mix_sub]
    mix_ds = TextDataset(mix_texts, tokenizer, args.max_seq_length)
    mix_dl = DataLoader(mix_ds, batch_size=args.score_batch_size, shuffle=False)

    base_ds = TextDataset(base_gen, tokenizer, args.max_seq_length)
    base_dl = DataLoader(base_ds, batch_size=args.score_batch_size, shuffle=False)

    gen_lps, train_lps, base_lps = [], [], []
    for pid in persona_ids:
        print(f"  Scoring: {pid}")
        ppath = persona_models_dir / f"persona_{pid}"
        lora_model = PeftModel.from_pretrained(base_model, ppath).to(device)
        gen_lps.append(compute_seq_logprobs(lora_model, gen_dl, device))
        train_lps.append(compute_seq_logprobs(lora_model, mix_dl, device))
        base_lps.append(compute_seq_logprobs(lora_model, base_dl, device))
        del lora_model
        torch.cuda.empty_cache()

    gen_lps = np.stack(gen_lps)
    train_lps = np.stack(train_lps)
    base_lps = np.stack(base_lps)

    del base_model
    torch.cuda.empty_cache()

    # ── 5. EM + Bayes correction ──
    print(f"\n[5/5] Running EM...")

    w_gen, _ = em_mixture_weights(gen_lps)
    w_train, _ = em_mixture_weights(train_lps)
    w_prior, _ = em_mixture_weights(base_lps)

    err_gen = float(np.max(np.abs(w_gen - true_w)))
    err_train = float(np.max(np.abs(w_train - true_w)))

    # Bayes correction: w_lik ∝ w_post / w_prior
    w_corrected_raw = w_gen / (w_prior + 1e-12)
    w_corrected = w_corrected_raw / w_corrected_raw.sum()
    err_corrected = float(np.max(np.abs(w_corrected - true_w)))

    corr_gen = np.corrcoef(gen_lps)
    corr_train = np.corrcoef(train_lps)
    corr_base = np.corrcoef(base_lps)
    triu = np.triu_indices(N_PERSONAS, k=1)

    results = {
        "true_weights": true_w.tolist(),
        "persona_ids": persona_ids,
        "mixture_counts": counts,
        "n_generated": len(generated),
        "n_mix_train": len(mix_sub),
        "n_base_gen": len(base_gen),
        "em_generations": {
            "weights": w_gen.tolist(),
            "max_error": err_gen,
        },
        "em_training_data": {
            "weights": w_train.tolist(),
            "max_error": err_train,
        },
        "em_prior": {
            "weights": w_prior.tolist(),
        },
        "em_bayes_corrected": {
            "weights": w_corrected.tolist(),
            "max_error": err_corrected,
        },
        "diagnostics": {
            "gen_mean_corr": float(corr_gen[triu].mean()),
            "train_mean_corr": float(corr_train[triu].mean()),
            "base_mean_corr": float(corr_base[triu].mean()),
        },
    }

    print(f"\n  ── Results ──")
    print(f"  True weights:       {np.array2string(true_w, precision=3)}")
    print(f"  EM on generations:  {np.array2string(w_gen, precision=3)}  (err {err_gen:.4f})")
    print(f"  EM on train data:   {np.array2string(w_train, precision=3)}  (err {err_train:.4f})")
    print(f"  EM on base gens:    {np.array2string(w_prior, precision=3)}  (prior)")
    print(f"  Bayes corrected:    {np.array2string(w_corrected, precision=3)}  (err {err_corrected:.4f})")
    print(f"  gen_corr={results['diagnostics']['gen_mean_corr']:.4f}, "
          f"train_corr={results['diagnostics']['train_mean_corr']:.4f}, "
          f"base_corr={results['diagnostics']['base_mean_corr']:.4f}")

    print(f"\n  ── Per-persona breakdown ──")
    print(f"  {'Persona':<45} {'True':>6} {'Post':>6} {'Train':>6} {'Prior':>6} {'Corr':>6}")
    for i, pid in enumerate(persona_ids):
        print(f"  {pid:<45} {true_w[i]:>6.3f} {w_gen[i]:>6.3f} "
              f"{w_train[i]:>6.3f} {w_prior[i]:>6.3f} {w_corrected[i]:>6.3f}")

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
