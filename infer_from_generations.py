"""
Infer persona mixture weights from model generations.

1. Generate completions from the mixture model
2. Score them with each persona model
3. Run EM to recover the mixture weights

This tests whether you can characterize a model's persona composition
without access to its training data — only its outputs.

Usage:
    python infer_from_generations.py --n_samples 200
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from scipy.special import logsumexp

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


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
def compute_logprobs(model, dataloader, device):
    """Returns (seq_logprobs, token_counts) per example."""
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


@torch.no_grad()
def generate_from_prompts(model, tokenizer, prompts, device,
                          max_new_tokens=256, batch_size=16, temperature=0.8):
    """Generate completions conditioned on prompts."""
    model.eval()
    all_texts = []
    tokenizer.padding_side = "left"

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        # Tokenize prompts with left padding for generation
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
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
            if len(text.strip()) > 20:
                all_texts.append(text)

        print(f"  Generated {len(all_texts)}/{len(prompts)}...")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--models_dir", type=str, default="models/")
    parser.add_argument("--eval_data", type=str, default="data/eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(args.models_dir)

    persona_dirs = sorted([
        "persona_compassionate_nurturing_demeanor_radiating",
        "persona_competitive_driven_individual_thrives",
        "persona_deeply_introspective_passionate_about",
        "persona_enthusiastic_curious_individual_deeply",
        "persona_highly_educated_individual_with",
        "persona_passionate_inquisitive_mindset_that",
    ])
    n_p = len(persona_dirs)
    true_w = np.ones(n_p) / n_p

    # Load base model + tokenizer
    print("Loading base model...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype, trust_remote_code=True
    ).to(device)

    # ================================================================
    # Step 1: Generate from mixture model, conditioned on eval prompts
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"STEP 1: GENERATING {args.n_samples} SAMPLES FROM MIXTURE MODEL")
    print(f"{'=' * 70}")

    # Load eval prompts
    import json as _json
    with open(args.eval_data) as f:
        eval_data = _json.load(f)

    # Extract just the user question (strip persona system prompt so it doesn't
    # confuse scoring — the generation model will produce persona-flavored text
    # without needing the system prompt at generation time)
    prompts = []
    for e in eval_data[:args.n_samples]:
        text = e["text"]
        # Find the user message
        user_start = text.find("<|im_start|>user\n")
        user_end = text.find("<|im_end|>", user_start)
        if user_start >= 0 and user_end >= 0:
            user_msg = text[user_start + len("<|im_start|>user\n"):user_end]
        else:
            user_msg = text.split("\n\n", 1)[0]
        # Format as ChatML with just user + assistant start (no system prompt)
        prompts.append(
            f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        )
    print(f"  Using {len(prompts)} prompts from eval data")

    mix_model = PeftModel.from_pretrained(base_model, models_dir / "mixture_uniform_diff")
    mix_model.to(device)

    texts = generate_from_prompts(
        mix_model, tokenizer, prompts, device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )
    del mix_model
    torch.cuda.empty_cache()

    print(f"\n  Got {len(texts)} valid generations")
    print(f"  Mean length: {np.mean([len(t) for t in texts]):.0f} chars")
    print(f"  Sample: {texts[0][:200]}...")

    # Save generations
    import json as _json2
    gen_dir = Path("results/generations")
    gen_dir.mkdir(parents=True, exist_ok=True)
    with open(gen_dir / "mixture_generations.json", "w") as f:
        _json2.dump(texts, f, indent=2)
    print(f"  Saved -> {gen_dir / 'mixture_generations.json'}")

    # ================================================================
    # Step 2: Score generations with each persona model
    # ================================================================
    print(f"\n{'=' * 70}")
    print("STEP 2: SCORING WITH PERSONA MODELS")
    print(f"{'=' * 70}")

    dataset = TextDataset(texts, tokenizer, 512)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    persona_lps = []
    tok_counts = None
    for pname in persona_dirs:
        print(f"  Scoring: {pname}")
        lora_model = PeftModel.from_pretrained(base_model, models_dir / pname).to(device)
        lps, tc = compute_logprobs(lora_model, dataloader, device)
        persona_lps.append(lps)
        if tok_counts is None:
            tok_counts = tc
        del lora_model
        torch.cuda.empty_cache()

    persona_lps = np.stack(persona_lps)

    # Diagnostic: check if persona models actually differ
    print(f"\n  Persona logprob diagnostics:")
    for i, pname in enumerate(persona_dirs):
        short = pname.replace("persona_", "")[:30]
        print(f"    {short:32s} mean_seq={persona_lps[i].mean():.1f}  mean_per_tok={( persona_lps[i]/tok_counts).mean():.4f}")
    corr = np.corrcoef(persona_lps)
    offdiag = corr[np.triu_indices(n_p, k=1)]
    print(f"    Mean pairwise correlation: {offdiag.mean():.4f}")
    print(f"    Min pairwise correlation:  {offdiag.min():.4f}")

    # ================================================================
    # Step 3: EM inference
    # ================================================================
    print(f"\n{'=' * 70}")
    print("STEP 3: EM INFERENCE")
    print(f"{'=' * 70}")

    # Use sequence logprobs for EM
    w_em, ll_hist = em_mixture_weights(persona_lps)

    print(f"\n  EM converged in {len(ll_hist)} iterations")
    print(f"\n  Recovered weights:")
    for pname, w in zip(persona_dirs, w_em):
        short = pname.replace("persona_", "")[:42]
        print(f"    {short:44s} {w:.6f}")
    print(f"\n  Max error vs uniform: {np.max(np.abs(w_em - true_w)):.6f}")
    print(f"  Sum: {w_em.sum():.6f}")

    # ================================================================
    # Step 4: Also generate from individual persona models for comparison
    # ================================================================
    print(f"\n{'=' * 70}")
    print("STEP 4: CONTROL — GENERATE FROM SINGLE PERSONA")
    print(f"{'=' * 70}")

    # Pick one persona, generate from it, see if EM assigns it all the weight
    test_persona = persona_dirs[0]
    print(f"\n  Generating from: {test_persona}")

    p_model = PeftModel.from_pretrained(base_model, models_dir / test_persona).to(device)
    p_texts = generate_from_prompts(
        p_model, tokenizer, prompts, device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )
    del p_model
    torch.cuda.empty_cache()

    print(f"  Got {len(p_texts)} generations")

    with open(gen_dir / "control_persona_generations.json", "w") as f:
        _json2.dump(p_texts, f, indent=2)
    print(f"  Saved -> {gen_dir / 'control_persona_generations.json'}")

    p_dataset = TextDataset(p_texts, tokenizer, 512)
    p_dl = DataLoader(p_dataset, batch_size=args.batch_size, shuffle=False)

    p_persona_lps = []
    for pname in persona_dirs:
        lora_model = PeftModel.from_pretrained(base_model, models_dir / pname).to(device)
        lps, _ = compute_logprobs(lora_model, p_dl, device)
        p_persona_lps.append(lps)
        del lora_model
        torch.cuda.empty_cache()
    p_persona_lps = np.stack(p_persona_lps)

    # Diagnostic
    print(f"\n  Control logprob diagnostics:")
    for i, pname in enumerate(persona_dirs):
        short = pname.replace("persona_", "")[:30]
        print(f"    {short:32s} mean_seq={p_persona_lps[i].mean():.1f}")
    corr_c = np.corrcoef(p_persona_lps)
    offdiag_c = corr_c[np.triu_indices(n_p, k=1)]
    print(f"    Mean pairwise correlation: {offdiag_c.mean():.4f}")

    w_single, _ = em_mixture_weights(p_persona_lps)

    true_single = np.zeros(n_p)
    true_single[0] = 1.0

    print(f"\n  Recovered weights (should be ~1.0 for {test_persona}):")
    for pname, w in zip(persona_dirs, w_single):
        short = pname.replace("persona_", "")[:42]
        marker = " <-- source" if pname == test_persona else ""
        print(f"    {short:44s} {w:.6f}{marker}")
    print(f"\n  Weight on source persona: {w_single[0]:.4f}")
    print(f"  Weight on all others: {1 - w_single[0]:.4f}")


if __name__ == "__main__":
    main()
