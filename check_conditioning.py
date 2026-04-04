"""
Diagnose why weight recovery fails: correlation, conditioning, and objective surface.

Tests:
1. Persona correlation matrix (sequence + first token)
2. Condition number of the persona logprob matrix
3. Objective surface flatness (random restarts spread)
4. Sensitivity analysis (how much does the objective change per unit weight perturbation)
5. SVD of persona logprob matrix (effective rank)

Usage:
    python check_conditioning.py --input results/logprobs.pt
"""

import argparse
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
import torch


def logsumexp_objective(w, persona_lps, mixture_lps):
    w = np.clip(w, 1e-12, None)
    log_w = np.log(w)
    shifted = persona_lps + log_w[:, None]
    pred = logsumexp(shifted, axis=0)
    return np.sum((pred - mixture_lps) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/logprobs.pt")
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    token_lps = data["token_logprobs"]
    mask = data["mask"]

    persona_names = [k for k in token_lps if k.startswith("persona_")]
    n_personas = len(persona_names)

    persona_seq = np.stack([
        (token_lps[n] * mask).sum(dim=-1).float().numpy() for n in persona_names
    ])
    mixture_seq = (token_lps["mixture_uniform"] * mask).sum(dim=-1).float().numpy()

    persona_first = np.stack([token_lps[n][:, 0].float().numpy() for n in persona_names])
    mixture_first = token_lps["mixture_uniform"][:, 0].float().numpy()

    for label, persona_lps, mixture_lps in [
        ("SEQUENCE LOGPROBS", persona_seq, mixture_seq),
        ("FIRST TOKEN LOGPROBS", persona_first, mixture_first),
    ]:
        print("=" * 60)
        print(label)
        print("=" * 60)

        # --- 1. Correlation matrix ---
        print("\n1. Persona correlation matrix:")
        corr = np.corrcoef(persona_lps)
        short_names = [n.replace("persona_", "")[:12] for n in persona_names]
        header = "            " + " ".join(f"{s:>12s}" for s in short_names)
        print(header)
        for i, name in enumerate(short_names):
            row = " ".join(f"{corr[i,j]:12.4f}" for j in range(n_personas))
            print(f"  {name:>10s} {row}")

        off_diag = corr[~np.eye(n_personas, dtype=bool)]
        print(f"\n  Mean off-diagonal: {off_diag.mean():.4f}")
        print(f"  Min off-diagonal:  {off_diag.min():.4f}")
        print(f"  Max off-diagonal:  {off_diag.max():.4f}")

        # --- 2. Condition number ---
        print("\n2. Condition number:")
        A = persona_lps.T  # (n_examples, n_personas)
        cond = np.linalg.cond(A)
        print(f"  cond(A): {cond:.2f}")

        # Centered
        A_centered = A - A.mean(axis=0, keepdims=True)
        cond_centered = np.linalg.cond(A_centered)
        print(f"  cond(A_centered): {cond_centered:.2f}")

        # --- 3. SVD / effective rank ---
        print("\n3. SVD analysis:")
        U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
        S_norm = S / S.sum()
        cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
        print(f"  Singular values: {S}")
        print(f"  Normalized:      {S_norm}")
        print(f"  Cumulative var:  {cumvar}")
        eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-12)))
        print(f"  Effective rank (exp entropy): {eff_rank:.2f}")

        # --- 4. Objective surface: random restarts ---
        print("\n4. Objective surface (50 random restarts):")
        rng = np.random.RandomState(42)
        objectives = []
        weights_found = []

        for i in range(50):
            if i == 0:
                w0 = np.ones(n_personas) / n_personas
            else:
                w0 = rng.dirichlet(np.ones(n_personas))

            result = minimize(
                logsumexp_objective, w0,
                args=(persona_lps, mixture_lps),
                method="SLSQP",
                bounds=[(1e-8, 1.0)] * n_personas,
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            objectives.append(result.fun)
            weights_found.append(result.x)

        objectives = np.array(objectives)
        weights_found = np.array(weights_found)
        print(f"  Objective range: [{objectives.min():.2f}, {objectives.max():.2f}]")
        print(f"  Objective std:   {objectives.std():.2f}")
        print(f"  Objective mean:  {objectives.mean():.2f}")

        # Weight spread across restarts
        w_std = weights_found.std(axis=0)
        w_mean = weights_found.mean(axis=0)
        print(f"\n  Mean weights across restarts:")
        for name, m, s in zip(persona_names, w_mean, w_std):
            print(f"    {name:40s} {m:.4f} +/- {s:.4f}")

        # --- 5. Sensitivity: perturb uniform weights ---
        print("\n5. Sensitivity (objective change per 0.01 weight perturbation):")
        w_uniform = np.ones(n_personas) / n_personas
        base_obj = logsumexp_objective(w_uniform, persona_lps, mixture_lps)
        print(f"  Objective at uniform: {base_obj:.2f}")
        for i, name in enumerate(persona_names):
            w_perturbed = w_uniform.copy()
            w_perturbed[i] += 0.01
            w_perturbed /= w_perturbed.sum()
            delta_obj = logsumexp_objective(w_perturbed, persona_lps, mixture_lps) - base_obj
            print(f"    +0.01 to {name:40s} -> delta obj = {delta_obj:+.4f}")

        print()


if __name__ == "__main__":
    main()
