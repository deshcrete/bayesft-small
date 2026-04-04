"""
Recover persona mixture weights using delta logprobs (persona - base).

By subtracting the base model, we remove the shared component that causes
0.9999 correlation and work only with what each LoRA actually learned.

The model:
    delta_mix(x) = sum_i w_i * delta_i(x)
where delta_i(x) = log P_i(x) - log P_base(x)

This is a linear system, solvable via OLS, NNLS, and constrained SLSQP.

Usage:
    python solve_weights_delta.py --input results/logprobs.pt
"""

import argparse
import numpy as np
from scipy.optimize import minimize, nnls
import torch


def solve_ols_simplex(A, b):
    """OLS with simplex projection: min ||Aw - b||^2, w >= 0, sum(w) = 1."""
    result = minimize(
        lambda w: np.sum((A @ w - b) ** 2),
        x0=np.ones(A.shape[1]) / A.shape[1],
        method="SLSQP",
        bounds=[(0, 1)] * A.shape[1],
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-15},
    )
    return result.x, result.fun


def solve_nnls_normalized(A, b):
    """NNLS then normalize to sum to 1."""
    w, residual = nnls(A, b)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    return w, residual


def solve_unconstrained_ols(A, b):
    """Unconstrained least squares (can go negative, doesn't sum to 1)."""
    w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    residual = np.sum((A @ w - b) ** 2)
    return w, residual


def print_weights(persona_names, w, true_weights, label, residual):
    print(f"\n{label}:")
    for name, wi in zip(persona_names, w):
        print(f"  {name:40s} {wi:.4f}")
    print(f"  Residual: {residual:.4f}")
    print(f"  Max error vs uniform: {np.max(np.abs(w - true_weights)):.4f}")
    print(f"  Sum of weights: {w.sum():.4f}")


def run_analysis(persona_deltas, mixture_delta, persona_names, true_weights, label):
    """Run all solvers on a set of deltas."""
    print("=" * 60)
    print(label)
    print("=" * 60)

    n_personas, n_examples = persona_deltas.shape

    # Correlation on deltas
    corr = np.corrcoef(persona_deltas)
    off_diag = corr[~np.eye(n_personas, dtype=bool)]
    print(f"\nDelta correlation: mean={off_diag.mean():.4f}, "
          f"min={off_diag.min():.4f}, max={off_diag.max():.4f}")

    # Condition number
    A = persona_deltas.T  # (n_examples, n_personas)
    b = mixture_delta      # (n_examples,)
    cond = np.linalg.cond(A)
    print(f"Condition number: {cond:.2f}")

    # SVD
    A_centered = A - A.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(A_centered, full_matrices=False)
    S_norm = S / S.sum()
    eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-12)))
    print(f"Effective rank: {eff_rank:.2f}")
    print(f"Singular values: {S}")

    # R² of the linear fit
    w_ols, res_ols = solve_unconstrained_ols(A, b)
    ss_res = res_ols
    ss_tot = np.sum((b - b.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"R² (unconstrained OLS): {r2:.6f}")

    # Solvers
    print_weights(persona_names, w_ols, true_weights, "Unconstrained OLS", res_ols)

    w_nnls, res_nnls = solve_nnls_normalized(A, b)
    print_weights(persona_names, w_nnls, true_weights, "NNLS (normalized)", res_nnls)

    w_simplex, res_simplex = solve_ols_simplex(A, b)
    print_weights(persona_names, w_simplex, true_weights, "OLS + simplex constraint", res_simplex)

    # Feasibility: is mixture delta within range of persona deltas?
    best_delta = persona_deltas.max(axis=0)
    worst_delta = persona_deltas.min(axis=0)
    above = (mixture_delta > best_delta).sum()
    below = (mixture_delta < worst_delta).sum()
    print(f"\nFeasibility (delta space):")
    print(f"  Mixture > best persona:  {above}/{n_examples} ({above/n_examples:.1%})")
    print(f"  Mixture < worst persona: {below}/{n_examples} ({below/n_examples:.1%})")

    # Feasibility vs persona spread: do violations happen where personas disagree most?
    spread = best_delta - worst_delta  # per-example persona spread
    violated_above = mixture_delta > best_delta
    violated_below = mixture_delta < worst_delta
    violated = violated_above | violated_below
    if violated.sum() > 0 and (~violated).sum() > 0:
        print(f"\n  Spread vs feasibility:")
        print(f"    Mean spread (violated):     {spread[violated].mean():.4f}")
        print(f"    Mean spread (non-violated): {spread[~violated].mean():.4f}")
        print(f"    Ratio: {spread[violated].mean() / spread[~violated].mean():.2f}x")
        # Correlation between spread and gap magnitude
        gap = np.maximum(mixture_delta - best_delta, worst_delta - mixture_delta)
        corr_spread_gap = np.corrcoef(spread, gap)[0, 1]
        print(f"    Correlation(spread, gap): {corr_spread_gap:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/logprobs.pt")
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    token_lps = data["token_logprobs"]
    mask = data["mask"]

    persona_names = [k for k in token_lps if k.startswith("persona_")]
    n_personas = len(persona_names)
    true_weights = np.ones(n_personas) / n_personas
    base_lps = token_lps["base"]

    print(f"Personas: {n_personas}")
    print(f"Examples: {mask.shape[0]}")
    print(f"True weights: {true_weights}")
    print()

    # --- First token deltas ---
    persona_first_delta = np.stack([
        (token_lps[n][:, 0] - base_lps[:, 0]).float().numpy() for n in persona_names
    ])
    mixture_first_delta = (token_lps["mixture_uniform"][:, 0] - base_lps[:, 0]).float().numpy()
    run_analysis(persona_first_delta, mixture_first_delta, persona_names, true_weights,
                 "FIRST TOKEN DELTAS")

    # --- Sequence deltas ---
    print()
    persona_seq_delta = np.stack([
        ((token_lps[n] - base_lps) * mask).sum(dim=-1).float().numpy() for n in persona_names
    ])
    mixture_seq_delta = ((token_lps["mixture_uniform"] - base_lps) * mask).sum(dim=-1).float().numpy()
    run_analysis(persona_seq_delta, mixture_seq_delta, persona_names, true_weights,
                 "SEQUENCE DELTAS")

    # --- Per-token deltas (mean across tokens) ---
    print()
    token_counts = data["token_counts"].float().numpy()
    persona_mean_delta = np.stack([
        (((token_lps[n] - base_lps) * mask).sum(dim=-1) / data["token_counts"]).float().numpy()
        for n in persona_names
    ])
    mixture_mean_delta = (((token_lps["mixture_uniform"] - base_lps) * mask).sum(dim=-1) / data["token_counts"]).float().numpy()
    run_analysis(persona_mean_delta, mixture_mean_delta, persona_names, true_weights,
                 "MEAN DELTAS (per-token average)")


if __name__ == "__main__":
    main()
