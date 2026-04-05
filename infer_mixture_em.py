"""
Bayesian persona weight inference via EM.

Given persona models P_1..P_k and observed data x_1..x_n, infer mixture
weights w such that the data was drawn from sum_j w_j P_j.

This uses the persona models as fixed components and solves for weights
directly — no mixture model needed.

Two modes:
  1. Score the actual mixture training data with persona models (ground truth test)
  2. Score the eval data (general held-out test)

Usage:
    python infer_mixture_em.py --logprobs results/logprobs_diff.pt --data data/mixture_uniform.json
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


def em_mixture_weights(persona_logprobs, n_iter=200, prior=None, tol=1e-10):
    """
    EM for mixture of multinomials.

    persona_logprobs: (n_personas, n_examples) — log P_j(x_i)
    Returns: weights (n_personas,), history of log-likelihoods
    """
    n_p, n_ex = persona_logprobs.shape

    # Initialize weights
    if prior is not None:
        w = np.array(prior, dtype=np.float64)
    else:
        w = np.ones(n_p) / n_p

    ll_history = []

    for it in range(n_iter):
        # E-step: responsibilities r_ij = w_j P_j(x_i) / sum_k w_k P_k(x_i)
        # In log space: log_r_ij = log(w_j) + log P_j(x_i) - log(sum_k w_k P_k(x_i))
        log_weighted = persona_logprobs + np.log(w + 1e-300)[:, None]  # (n_p, n_ex)
        log_denom = logsumexp(log_weighted, axis=0)  # (n_ex,)
        log_r = log_weighted - log_denom[None, :]  # (n_p, n_ex)
        r = np.exp(log_r)  # (n_p, n_ex)

        # Log-likelihood
        ll = log_denom.sum()
        ll_history.append(ll)

        # M-step: w_j = (1/n) sum_i r_ij
        w_new = r.mean(axis=1)
        w_new = np.clip(w_new, 1e-15, None)
        w_new /= w_new.sum()

        if it > 0 and abs(ll_history[-1] - ll_history[-2]) < tol:
            break

        w = w_new

    return w, ll_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logprobs", type=str, default="results/logprobs_diff.pt",
                        help="Pre-computed logprobs file (for eval data)")
    parser.add_argument("--data", type=str, default="data/mixture_uniform.json",
                        help="Mixture training data to score with persona models")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================================================================
    # MODE 1: Use pre-computed logprobs on eval data
    # ================================================================
    print("=" * 70)
    print("MODE 1: EM ON EVAL DATA (pre-computed logprobs)")
    print("=" * 70)

    data = torch.load(args.logprobs, weights_only=False)
    mask = data["mask"]
    persona_names = sorted(k for k in data["token_logprobs"] if k.startswith("persona_"))
    n_p = len(persona_names)

    # Sequence logprobs
    persona_seq = np.stack([
        (data["token_logprobs"][n] * mask).sum(dim=-1).float().numpy()
        for n in persona_names
    ])

    true_w = np.ones(n_p) / n_p

    print(f"\n  Personas: {n_p}")
    print(f"  Examples: {persona_seq.shape[1]}")
    print(f"  True weights: uniform (1/{n_p})")

    # Run EM
    w_em, ll_hist = em_mixture_weights(persona_seq)

    print(f"\n  EM converged in {len(ll_hist)} iterations")
    print(f"  Final log-likelihood: {ll_hist[-1]:.2f}")
    print(f"\n  Recovered weights:")
    for name, w in zip(persona_names, w_em):
        short = name.replace("persona_", "")[:42]
        print(f"    {short:44s} {w:.6f}")
    print(f"    Max error vs uniform: {np.max(np.abs(w_em - true_w)):.6f}")
    print(f"    Sum: {w_em.sum():.6f}")

    # Also try with mean logprobs (per-token)
    tc = data["token_counts"].float().numpy()
    persona_mean = persona_seq / tc[None, :]

    print(f"\n  EM on mean logprobs (per-token):")
    w_em_mean, ll_hist_mean = em_mixture_weights(persona_mean)
    print(f"  Converged in {len(ll_hist_mean)} iterations")
    for name, w in zip(persona_names, w_em_mean):
        short = name.replace("persona_", "")[:42]
        print(f"    {short:44s} {w:.6f}")
    print(f"    Max error vs uniform: {np.max(np.abs(w_em_mean - true_w)):.6f}")

    # ================================================================
    # MODE 2: Score mixture training data with persona models
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODE 2: EM ON MIXTURE TRAINING DATA (fresh logprobs)")
    print("=" * 70)

    with open(args.data) as f:
        mix_data = json.load(f)
    texts = [e["text"] for e in mix_data]
    print(f"\n  Mixture training examples: {len(texts)}")

    # Check actual persona distribution in training data
    if "persona" in mix_data[0]:
        from collections import Counter
        actual_dist = Counter(e["persona"] for e in mix_data)
        print(f"  Actual persona distribution in training data:")
        for p, c in actual_dist.most_common():
            print(f"    {p[:60]:60s} {c} ({c/len(mix_data):.3f})")

    print(f"\n  Loading base model...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype, trust_remote_code=True
    ).to(device)

    dataset = TextDataset(texts, tokenizer, 512)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    models_dir = Path("models")
    persona_lps_train = []

    for pname in persona_names:
        pdir = pname  # directory name matches the key
        ppath = models_dir / pdir
        if not ppath.exists():
            # Try without persona_ prefix variations
            print(f"  WARNING: {ppath} not found, skipping")
            continue
        print(f"  Computing logprobs: {pname}")
        lora_model = PeftModel.from_pretrained(base_model, ppath).to(device)
        lps = compute_seq_logprobs(lora_model, dataloader, device)
        persona_lps_train.append(lps)
        del lora_model
        torch.cuda.empty_cache()

    persona_lps_train = np.stack(persona_lps_train)
    print(f"\n  Persona logprobs shape: {persona_lps_train.shape}")

    # Run EM on training data
    w_em_train, ll_hist_train = em_mixture_weights(persona_lps_train)

    print(f"\n  EM converged in {len(ll_hist_train)} iterations")
    print(f"  Recovered weights (from training data):")
    for name, w in zip(persona_names, w_em_train):
        short = name.replace("persona_", "")[:42]
        print(f"    {short:44s} {w:.6f}")
    print(f"    Max error vs uniform: {np.max(np.abs(w_em_train - true_w)):.6f}")

    # ================================================================
    # MODE 3: Non-uniform test — use only data from 2 personas
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODE 3: NON-UNIFORM MIXTURE TEST")
    print("=" * 70)

    if "persona" in mix_data[0]:
        unique_p = sorted(set(e["persona"] for e in mix_data))
        # Take first 2 personas, with 2:1 ratio
        p1_data = [e for e in mix_data if e["persona"] == unique_p[0]]
        p2_data = [e for e in mix_data if e["persona"] == unique_p[1]]
        # Construct 2:1 mix from p1:p2, using 120 from p1 and 60 from p2
        n1, n2 = 120, 60
        subset = p1_data[:n1] + p2_data[:n2]
        np.random.RandomState(42).shuffle(subset)
        sub_texts = [e["text"] for e in subset]
        true_w_sub = np.array([n1, n2]) / (n1 + n2)  # [0.667, 0.333]

        print(f"\n  Constructed 2-persona mix: {n1}x persona0 + {n2}x persona1")
        print(f"  True weights: [{true_w_sub[0]:.3f}, {true_w_sub[1]:.3f}]")

        sub_ds = TextDataset(sub_texts, tokenizer, 512)
        sub_dl = DataLoader(sub_ds, batch_size=args.batch_size, shuffle=False)

        sub_lps = []
        for i, pname in enumerate(persona_names):
            ppath = models_dir / pname
            if not ppath.exists():
                continue
            lora_model = PeftModel.from_pretrained(base_model, ppath).to(device)
            lps = compute_seq_logprobs(lora_model, sub_dl, device)
            sub_lps.append(lps)
            del lora_model
            torch.cuda.empty_cache()
        sub_lps = np.stack(sub_lps)

        w_sub, _ = em_mixture_weights(sub_lps)

        print(f"\n  EM recovered weights (all 6 personas):")
        for name, w in zip(persona_names, w_sub):
            short = name.replace("persona_", "")[:42]
            print(f"    {short:44s} {w:.6f}")

        # Map persona strings to model indices
        persona_to_idx = {}
        for full_p in unique_p:
            for j, mn in enumerate(persona_names):
                slug = mn.replace("persona_", "")
                slug_words = slug.split("_")[:4]
                if all(w in full_p.lower() for w in slug_words):
                    persona_to_idx[full_p] = j
                    break

        if unique_p[0] in persona_to_idx and unique_p[1] in persona_to_idx:
            i1, i2 = persona_to_idx[unique_p[0]], persona_to_idx[unique_p[1]]
            print(f"\n  Mapped persona weights:")
            print(f"    Persona 0 (idx {i1}): recovered={w_sub[i1]:.4f}, true={true_w_sub[0]:.4f}")
            print(f"    Persona 1 (idx {i2}): recovered={w_sub[i2]:.4f}, true={true_w_sub[1]:.4f}")
            print(f"    Weight on other 4 personas: {w_sub.sum() - w_sub[i1] - w_sub[i2]:.4f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  The EM approach works directly on data scored by persona models.
  No mixture model needed — the persona models ARE the components,
  and EM finds the weights that maximize P(data | mixture).

  This is the correct Bayesian formulation:
    Prior:     p(persona) = w  (the mixture weights)
    Likelihood: p(x | persona_j) = P_j(x)  (persona model logprobs)
    Posterior:  p(persona | data) via EM on the observed data

  To get a posterior after SFT:
    1. Fine-tune on new data D
    2. Score D with each persona model P_j
    3. Run EM to get posterior weights
    4. If D is not available, generate from the SFT'd model and score that
""")


if __name__ == "__main__":
    main()
