"""
Recover persona mixture weights via law of total probability.

Given persona logprobs P_i(x) and mixture logprobs P_mix(x), solve for w such that:
    P_mix(x) = sum_i w_i * P_i(x)    (in probability space)
    => log P_mix(x) = logsumexp_i(log w_i + log P_i(x))

Solves via SLSQP on both first-token and sequence logprobs.

Usage:
    python solve_weights.py --input results/logprobs.pt
"""

import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import torch


def solve_logsumexp_slsqp(persona_lps, mixture_lps, n_restarts=10, seed=42):
    """
    Minimize sum_x (logsumexp_i(log w_i + log P_i(x)) - log P_mix(x))^2
    subject to w >= 0, sum(w) = 1.
    """
    n_personas, n_examples = persona_lps.shape
    rng = np.random.RandomState(seed)

    def objective(log_w):
        # log_w are unconstrained, w = softmax(log_w) to enforce simplex
        # Actually, we use raw w with constraints instead
        w = log_w  # these are the actual weights, not log
        w = np.clip(w, 1e-12, None)
        log_w_arr = np.log(w)
        # (n_personas, n_examples) + (n_personas, 1)
        shifted = persona_lps + log_w_arr[:, None]
        pred = logsumexp(shifted, axis=0)  # (n_examples,)
        residuals = pred - mixture_lps
        return np.sum(residuals ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-8, 1.0)] * n_personas

    best_result = None
    best_obj = np.inf

    for i in range(n_restarts):
        if i == 0:
            w0 = np.ones(n_personas) / n_personas
        else:
            w0 = rng.dirichlet(np.ones(n_personas))

        result = minimize(
            objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result

    return best_result.x, best_result.fun


def solve_linear_nnls(persona_lps, mixture_lps):
    """
    Solve in probability space: P_mix(x) = sum_i w_i P_i(x).
    Convert logprobs to probs, solve NNLS, normalize.
    """
    from scipy.optimize import nnls

    # Shift for numerical stability (subtract max logprob per example)
    max_lp = np.max(persona_lps, axis=0, keepdims=True)
    persona_probs = np.exp(persona_lps - max_lp)  # (n_personas, n_examples)
    mixture_probs = np.exp(mixture_lps - max_lp.squeeze())  # (n_examples,)

    # NNLS: min ||A @ w - b||^2, w >= 0
    A = persona_probs.T  # (n_examples, n_personas)
    b = mixture_probs     # (n_examples,)

    w, residual = nnls(A, b)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    return w, residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/logprobs.pt")
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    token_lps = data["token_logprobs"]
    mask = data["mask"]
    token_counts = data["token_counts"]

    persona_names = [k for k in token_lps if k.startswith("persona_")]
    assert "mixture_uniform" in token_lps, "No mixture model logprobs found"

    true_weights = np.ones(len(persona_names)) / len(persona_names)

    print(f"Personas: {len(persona_names)}")
    print(f"Examples: {mask.shape[0]}")
    print(f"True weights: {true_weights}")
    print()

    # --- First token logprobs ---
    print("=" * 60)
    print("FIRST TOKEN LOGPROBS")
    print("=" * 60)

    persona_first = np.stack([token_lps[n][:, 0].float().numpy() for n in persona_names])  # (6, n)
    mixture_first = token_lps["mixture_uniform"][:, 0].float().numpy()

    print("\nSLSQP (logsumexp):")
    w_slsqp, obj_slsqp = solve_logsumexp_slsqp(persona_first, mixture_first)
    for name, w in zip(persona_names, w_slsqp):
        print(f"  {name:40s} {w:.4f}")
    print(f"  Objective: {obj_slsqp:.4f}")
    print(f"  Max error vs uniform: {np.max(np.abs(w_slsqp - true_weights)):.4f}")

    print("\nNNLS (probability space):")
    w_nnls, res_nnls = solve_linear_nnls(persona_first, mixture_first)
    for name, w in zip(persona_names, w_nnls):
        print(f"  {name:40s} {w:.4f}")
    print(f"  Residual: {res_nnls:.4f}")
    print(f"  Max error vs uniform: {np.max(np.abs(w_nnls - true_weights)):.4f}")

    # --- Sequence logprobs ---
    print()
    print("=" * 60)
    print("SEQUENCE LOGPROBS")
    print("=" * 60)

    persona_seq = np.stack([
        (token_lps[n] * mask).sum(dim=-1).float().numpy() for n in persona_names
    ])  # (6, n)
    mixture_seq = (token_lps["mixture_uniform"] * mask).sum(dim=-1).float().numpy()

    print("\nSLSQP (logsumexp):")
    w_slsqp, obj_slsqp = solve_logsumexp_slsqp(persona_seq, mixture_seq)
    for name, w in zip(persona_names, w_slsqp):
        print(f"  {name:40s} {w:.4f}")
    print(f"  Objective: {obj_slsqp:.4f}")
    print(f"  Max error vs uniform: {np.max(np.abs(w_slsqp - true_weights)):.4f}")

    print("\nNNLS (probability space):")
    w_nnls, res_nnls = solve_linear_nnls(persona_seq, mixture_seq)
    for name, w in zip(persona_names, w_nnls):
        print(f"  {name:40s} {w:.4f}")
    print(f"  Residual: {res_nnls:.4f}")
    print(f"  Max error vs uniform: {np.max(np.abs(w_nnls - true_weights)):.4f}")

    # --- Feasibility check ---
    print()
    print("=" * 60)
    print("FEASIBILITY CHECK")
    print("=" * 60)

    best_persona_seq = persona_seq.max(axis=0)
    gap = mixture_seq - best_persona_seq
    n_violations = (gap > 0).sum()
    print(f"Sequence: {n_violations}/{len(gap)} violations ({n_violations/len(gap):.1%})")
    print(f"  Mean gap: {gap.mean():.2f}")
    print(f"  Max gap:  {gap.max():.2f}")

    best_persona_first = persona_first.max(axis=0)
    gap_first = mixture_first - best_persona_first
    n_violations_first = (gap_first > 0).sum()
    print(f"First token: {n_violations_first}/{len(gap_first)} violations ({n_violations_first/len(gap_first):.1%})")
    print(f"  Mean gap: {gap_first.mean():.2f}")
    print(f"  Max gap:  {gap_first.max():.2f}")


if __name__ == "__main__":
    main()
