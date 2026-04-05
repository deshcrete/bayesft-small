"""
Red-team the feasibility violations: how much do they actually matter,
and where do they come from?

Tests:
1. Remove violating examples and re-solve — does recovery improve?
2. Clamp violations and re-solve — does recovery improve?
3. Per-token analysis — are violations concentrated in specific tokens?
4. Violation magnitude vs recovery error — is there a dose-response?
5. Model capacity analysis — why bigger models produce more violations

Usage:
    python diagnose_violations.py --input results/logprobs.pt
"""

import argparse
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import logsumexp
import torch


def solve_logsumexp_slsqp(persona_lps, mixture_lps, n_restarts=20, seed=42):
    n_personas = persona_lps.shape[0]
    rng = np.random.RandomState(seed)

    def objective(w):
        w = np.clip(w, 1e-12, None)
        shifted = persona_lps + np.log(w)[:, None]
        pred = logsumexp(shifted, axis=0)
        return np.sum((pred - mixture_lps) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-8, 1.0)] * n_personas

    best_result = None
    best_obj = np.inf
    for i in range(n_restarts):
        w0 = np.ones(n_personas) / n_personas if i == 0 else rng.dirichlet(np.ones(n_personas))
        result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 2000, "ftol": 1e-15})
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result
    return best_result.x, best_result.fun


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/logprobs.pt")
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    token_lps = data["token_logprobs"]
    mask = data["mask"]
    token_counts = data["token_counts"]

    persona_names = sorted(k for k in token_lps if k.startswith("persona_"))
    n_personas = len(persona_names)
    n_examples = mask.shape[0]
    true_w = np.ones(n_personas) / n_personas

    # Sequence logprobs
    persona_seq = np.stack([
        (token_lps[n] * mask).sum(dim=-1).float().numpy() for n in persona_names
    ])
    mixture_seq = (token_lps["mixture_uniform"] * mask).sum(dim=-1).float().numpy()

    best_persona = persona_seq.max(axis=0)
    gap = mixture_seq - best_persona
    violations = gap > 0
    n_viol = violations.sum()

    print(f"Setup: {n_personas} personas, {n_examples} examples, {n_viol} violations ({n_viol/n_examples:.1%})")

    # ================================================================
    # TEST 1: Remove violating examples and re-solve
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: REMOVE VIOLATING EXAMPLES")
    print("=" * 70)

    print("\n  Baseline (all examples):")
    w_all, obj_all = solve_logsumexp_slsqp(persona_seq, mixture_seq)
    err_all = np.max(np.abs(w_all - true_w))
    print(f"    Max error: {err_all:.4f}, Obj: {obj_all:.1f}")
    for name, w in zip(persona_names, w_all):
        print(f"      {name:45s} {w:.4f}")

    clean = ~violations
    print(f"\n  Without violations ({clean.sum()} examples):")
    w_clean, obj_clean = solve_logsumexp_slsqp(persona_seq[:, clean], mixture_seq[clean])
    err_clean = np.max(np.abs(w_clean - true_w))
    print(f"    Max error: {err_clean:.4f}, Obj: {obj_clean:.1f}")
    for name, w in zip(persona_names, w_clean):
        print(f"      {name:45s} {w:.4f}")

    # Progressively remove worst violations
    print(f"\n  Progressive removal (by gap magnitude):")
    sorted_idx = np.argsort(-gap)  # most violating first
    print(f"    {'Removed':>8s}  {'Remaining':>9s}  {'MaxErr':>8s}  {'Obj':>14s}")
    print(f"    " + "-" * 50)
    for k in [0, 5, 10, 20, 40, 60, 80, 100, 150, 200]:
        keep = np.ones(n_examples, dtype=bool)
        keep[sorted_idx[:k]] = False
        w_k, obj_k = solve_logsumexp_slsqp(persona_seq[:, keep], mixture_seq[keep])
        err_k = np.max(np.abs(w_k - true_w))
        print(f"    {k:>8d}  {keep.sum():>9d}  {err_k:>8.4f}  {obj_k:>14.1f}")

    # ================================================================
    # TEST 2: Clamp violations to max persona
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: CLAMP VIOLATIONS (set mixture = max persona where violated)")
    print("=" * 70)

    mixture_clamped = np.where(violations, best_persona, mixture_seq)
    w_clamp, obj_clamp = solve_logsumexp_slsqp(persona_seq, mixture_clamped)
    err_clamp = np.max(np.abs(w_clamp - true_w))
    print(f"    Max error: {err_clamp:.4f}, Obj: {obj_clamp:.1f}")
    for name, w in zip(persona_names, w_clamp):
        print(f"      {name:45s} {w:.4f}")

    # ================================================================
    # TEST 3: Per-token violation analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: PER-TOKEN VIOLATION STRUCTURE")
    print("=" * 70)

    # Get per-token logprobs
    persona_tok = np.stack([
        (token_lps[n] * mask).float().numpy() for n in persona_names
    ])  # (n_personas, n_examples, seq_len)
    mixture_tok = (token_lps["mixture_uniform"] * mask).float().numpy()  # (n_examples, seq_len)
    mask_np = mask.float().numpy()

    # Per-token gap: mixture - max(persona) at each position
    best_persona_tok = persona_tok.max(axis=0)  # (n_examples, seq_len)
    token_gap = (mixture_tok - best_persona_tok) * mask_np

    # For violating examples, analyze token-level structure
    if n_viol > 0:
        viol_idx = np.where(violations)[0]
        token_gaps_viol = token_gap[viol_idx]  # (n_viol, seq_len)
        token_mask_viol = mask_np[viol_idx]

        # What fraction of tokens in violation examples are themselves violations?
        tok_viol_count = (token_gaps_viol > 0).sum()
        tok_total = token_mask_viol.sum()
        print(f"\n  Violation examples ({n_viol} examples):")
        print(f"    Tokens where mixture > max(persona): {int(tok_viol_count)}/{int(tok_total)} ({tok_viol_count/tok_total:.1%})")

        # Positive token gaps (the actual violations per token)
        pos_gaps = token_gaps_viol[token_gaps_viol > 0]
        neg_gaps = token_gaps_viol[(token_gaps_viol < 0) & (token_mask_viol > 0)]
        print(f"    Mean positive gap (violation tokens): {pos_gaps.mean():.4f}")
        print(f"    Mean negative gap (non-violation tokens): {neg_gaps.mean():.4f}")
        print(f"    Sum positive gaps: {pos_gaps.sum():.2f}")
        print(f"    Sum negative gaps: {neg_gaps.sum():.2f}")
        print(f"    Net (why sequence violates): {pos_gaps.sum() + neg_gaps.sum():.2f}")

        # Position distribution: are violations early or late?
        seq_len = token_gap.shape[1]
        position_viol_rate = np.zeros(seq_len)
        position_count = np.zeros(seq_len)
        for t in range(seq_len):
            active = token_mask_viol[:, t] > 0
            if active.sum() > 0:
                position_viol_rate[t] = (token_gaps_viol[:, t][active] > 0).mean()
                position_count[t] = active.sum()

        # Report by position buckets
        valid_positions = position_count > 0
        n_valid = valid_positions.sum()
        bucket_size = max(1, n_valid // 5)
        valid_idx = np.where(valid_positions)[0]
        print(f"\n    Token position violation rate (in violating examples):")
        for b in range(5):
            start = b * bucket_size
            end = min((b + 1) * bucket_size, len(valid_idx))
            if start >= len(valid_idx):
                break
            bucket_idx = valid_idx[start:end]
            rate = np.mean([position_viol_rate[t] for t in bucket_idx])
            print(f"      Positions {bucket_idx[0]:3d}-{bucket_idx[-1]:3d}: {rate:.3f}")

    # Also check: across ALL examples, how many per-token violations?
    all_tok_viol = ((token_gap > 0) * mask_np).sum()
    all_tok_total = mask_np.sum()
    print(f"\n  All examples:")
    print(f"    Per-token violations: {int(all_tok_viol)}/{int(all_tok_total)} ({all_tok_viol/all_tok_total:.2%})")

    # ================================================================
    # TEST 4: Dose-response — violation magnitude vs recovery error
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 4: DOSE-RESPONSE — DO VIOLATIONS CAUSE THE ERROR?")
    print("=" * 70)

    # If violations cause the error, injecting artificial violations into
    # a perfect synthetic mixture should degrade recovery proportionally
    print("\n  Injecting artificial violations into synthetic perfect mixture:")

    synth_mix = logsumexp(persona_seq + np.log(true_w)[:, None], axis=0)
    w_synth, obj_synth = solve_logsumexp_slsqp(persona_seq, synth_mix)
    print(f"    Baseline (no injection): max_err={np.max(np.abs(w_synth - true_w)):.6f}")

    rng = np.random.RandomState(42)
    print(f"\n    {'Pct injected':>13s}  {'N_inject':>8s}  {'Bump size':>9s}  {'MaxErr':>8s}  {'Obj':>12s}")
    print(f"    " + "-" * 58)

    for pct in [0.01, 0.02, 0.04, 0.08, 0.15, 0.30]:
        for bump in [5, 15, 32]:
            n_inject = int(pct * n_examples)
            inject_idx = rng.choice(n_examples, n_inject, replace=False)
            corrupted = synth_mix.copy()
            corrupted[inject_idx] += bump  # force violations of given magnitude
            w_c, obj_c = solve_logsumexp_slsqp(persona_seq, corrupted, n_restarts=10)
            err_c = np.max(np.abs(w_c - true_w))
            print(f"    {pct:>12.0%}  {n_inject:>8d}  {bump:>9d}  {err_c:>8.4f}  {obj_c:>12.1f}")

    # ================================================================
    # TEST 5: WHY BIGGER MODELS → MORE VIOLATIONS
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 5: MODEL CAPACITY AND NONLINEARITY")
    print("=" * 70)

    # Measure the LoRA "effect size" — how much each persona deviates from base
    base_seq = (token_lps["base"] * mask).sum(dim=-1).float().numpy()
    deltas = persona_seq - base_seq[None, :]  # (n_personas, n_examples)
    mix_delta = mixture_seq - base_seq

    print(f"\n  LoRA effect sizes (sequence logprob delta from base):")
    for name, d in zip(persona_names, deltas):
        print(f"    {name:45s} mean={d.mean():>8.2f}  std={d.std():>7.2f}")
    print(f"    {'mixture':45s} mean={mix_delta.mean():>8.2f}  std={mix_delta.std():>7.2f}")

    # Nonlinearity measure: if LoRA composition were linear,
    # delta_mix ≈ sum_i w_i * delta_i
    # The residual measures nonlinearity
    predicted_mix_delta = (deltas * true_w[:, None]).sum(axis=0)
    nonlinearity = mix_delta - predicted_mix_delta

    print(f"\n  Nonlinearity (mixture_delta - weighted_sum_of_persona_deltas):")
    print(f"    Mean:   {nonlinearity.mean():.4f}")
    print(f"    Std:    {nonlinearity.std():.4f}")
    print(f"    Median: {np.median(nonlinearity):.4f}")
    print(f"    P5/P95: {np.percentile(nonlinearity, 5):.4f} / {np.percentile(nonlinearity, 95):.4f}")

    # Correlation between nonlinearity and gap
    corr_nonlin_gap = np.corrcoef(nonlinearity, gap)[0, 1]
    print(f"    Correlation(nonlinearity, gap): {corr_nonlin_gap:.4f}")

    # For violating examples specifically
    if n_viol > 0:
        print(f"\n  Nonlinearity in violation examples vs non-violations:")
        print(f"    Violations:     mean={nonlinearity[violations].mean():.4f}, std={nonlinearity[violations].std():.4f}")
        print(f"    Non-violations: mean={nonlinearity[~violations].mean():.4f}, std={nonlinearity[~violations].std():.4f}")

    # Delta interaction: are violations where persona deltas are large?
    delta_spread = deltas.max(axis=0) - deltas.min(axis=0)
    print(f"\n  Persona delta spread (max - min across personas per example):")
    print(f"    Violations:     mean={delta_spread[violations].mean():.2f}")
    print(f"    Non-violations: mean={delta_spread[~violations].mean():.2f}")

    abs_delta_mean = np.abs(deltas).mean(axis=0)
    print(f"\n  Mean |delta| per example:")
    print(f"    Violations:     mean={abs_delta_mean[violations].mean():.2f}")
    print(f"    Non-violations: mean={abs_delta_mean[~violations].mean():.2f}")

    # Token count (proxy for sequence length)
    tc = token_counts.float().numpy()
    print(f"\n  Sequence length:")
    print(f"    Violations:     mean={tc[violations].mean():.1f}")
    print(f"    Non-violations: mean={tc[~violations].mean():.1f}")

    print(f"""
  ─────────────────────────────────────────────────────────────
  WHY BIGGER MODELS WOULD HAVE MORE VIOLATIONS:

  Violations occur when the mixture LoRA captures cross-persona
  interactions that individual persona LoRAs cannot.

  Larger model → higher LoRA effective rank → mixture adapter
  can represent more complex functions → more interaction terms
  that violate the linear decomposition assumption.

  Key indicators to check across model sizes:
    1. Mean |delta| (LoRA effect size): larger = more room for nonlinearity
    2. Nonlinearity std: larger = more violation potential
    3. Correlation(nonlinearity, gap): should stay high across scales
  ─────────────────────────────────────────────────────────────""")


if __name__ == "__main__":
    main()
