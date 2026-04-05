"""
Diagnose whether correlation is the root cause of weight recovery failure.

Tests:
1. Subset test: recover weights using only the 2 least-correlated personas
2. Synthetic decorrelation: add artificial separation to logprobs and check if recovery improves
3. Noise floor: compare signal magnitude (inter-persona differences) to estimation noise

Usage:
    python diagnose_correlation.py --input results/logprobs.pt
"""

import argparse
import itertools

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import logsumexp

import torch


def solve_logsumexp_slsqp(persona_lps, mixture_lps, n_restarts=20, seed=42):
    n_personas, n_examples = persona_lps.shape
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


def solve_linear_nnls(persona_lps, mixture_lps):
    # Shift per-example for numerical stability
    all_lps = np.vstack([persona_lps, mixture_lps[None, :]])
    max_lp = np.max(all_lps, axis=0, keepdims=True)
    persona_probs = np.exp(persona_lps - max_lp)
    mixture_probs = np.exp(mixture_lps - max_lp.squeeze())
    # Clip infs/nans from overflow
    finite = np.isfinite(persona_probs).all(axis=0) & np.isfinite(mixture_probs)
    if not finite.all():
        persona_probs = persona_probs[:, finite]
        mixture_probs = mixture_probs[finite]
    w, residual = nnls(persona_probs.T, mixture_probs)
    s = w.sum()
    if s > 0:
        w = w / s
    return w, residual


def print_weights(names, weights, true_weights):
    for name, w, tw in zip(names, weights, true_weights):
        err = abs(w - tw)
        print(f"    {name:45s} {w:.4f}  (true={tw:.4f}, err={err:.4f})")
    print(f"    Max error: {np.max(np.abs(weights - true_weights)):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/logprobs.pt")
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    token_lps = data["token_logprobs"]
    mask = data["mask"]

    persona_names = sorted(k for k in token_lps if k.startswith("persona_"))
    n_personas = len(persona_names)

    # Compute sequence logprobs
    persona_seq = np.stack([
        (token_lps[n] * mask).sum(dim=-1).float().numpy() for n in persona_names
    ])
    mixture_seq = (token_lps["mixture_uniform"] * mask).sum(dim=-1).float().numpy()

    # Compute first-token logprobs
    persona_first = np.stack([token_lps[n][:, 0].float().numpy() for n in persona_names])
    mixture_first = token_lps["mixture_uniform"][:, 0].float().numpy()

    # ================================================================
    # TEST 1: Subset — 2 least-correlated personas
    # ================================================================
    print("=" * 70)
    print("TEST 1: SUBSET — 2 LEAST-CORRELATED PERSONAS")
    print("=" * 70)

    corr = np.corrcoef(persona_seq)
    # Find pair with lowest correlation
    min_corr = 1.0
    best_pair = (0, 1)
    for i, j in itertools.combinations(range(n_personas), 2):
        if corr[i, j] < min_corr:
            min_corr = corr[i, j]
            best_pair = (i, j)

    i, j = best_pair
    print(f"\nLeast correlated pair: {persona_names[i]} & {persona_names[j]}")
    print(f"Correlation: {min_corr:.4f}")

    # Build synthetic mixture as true 50/50 blend of these 2 personas
    sub_names = [persona_names[i], persona_names[j]]
    sub_seq = persona_seq[[i, j]]
    sub_first = persona_first[[i, j]]
    true_w_sub = np.array([0.5, 0.5])

    # Use actual mixture model target
    print("\n  Against actual mixture model (true weights unknown for subset):")
    w, obj = solve_logsumexp_slsqp(sub_seq, mixture_seq)
    print(f"    SLSQP seq:   [{w[0]:.4f}, {w[1]:.4f}]  obj={obj:.1f}")
    w, _ = solve_linear_nnls(sub_seq, mixture_seq)
    print(f"    NNLS  seq:   [{w[0]:.4f}, {w[1]:.4f}]")

    # Also test with synthetic 50/50 mixture target
    print("\n  Against synthetic 50/50 mixture (ground truth = [0.5, 0.5]):")
    synth_mix_seq = logsumexp(sub_seq + np.log(0.5), axis=0)
    synth_mix_first = logsumexp(sub_first + np.log(0.5), axis=0)

    w, obj = solve_logsumexp_slsqp(sub_seq, synth_mix_seq)
    print(f"    SLSQP seq:   [{w[0]:.4f}, {w[1]:.4f}]  obj={obj:.6f}")
    w, obj = solve_logsumexp_slsqp(sub_first, synth_mix_first)
    print(f"    SLSQP first: [{w[0]:.4f}, {w[1]:.4f}]  obj={obj:.6f}")

    # ================================================================
    # TEST 2: All pairs — systematic check
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: ALL PAIRS — SYNTHETIC 50/50 RECOVERY")
    print("=" * 70)
    print(f"\n  {'Pair':60s} {'Corr':>6s}  {'SLSQP w':>14s}  {'MaxErr':>7s}  {'Obj':>12s}")
    print("  " + "-" * 105)

    pair_results = []
    for a, b in itertools.combinations(range(n_personas), 2):
        sub = persona_seq[[a, b]]
        synth = logsumexp(sub + np.log(0.5), axis=0)
        w, obj = solve_logsumexp_slsqp(sub, synth, n_restarts=10)
        c = corr[a, b]
        err = np.max(np.abs(w - 0.5))
        short_a = persona_names[a].replace("persona_", "")[:25]
        short_b = persona_names[b].replace("persona_", "")[:25]
        pair_label = f"{short_a} + {short_b}"
        print(f"  {pair_label:60s} {c:.4f}  [{w[0]:.4f},{w[1]:.4f}]  {err:.5f}  {obj:.4f}")
        pair_results.append((c, err, obj))

    corrs = [r[0] for r in pair_results]
    errs = [r[1] for r in pair_results]
    rank_corr = np.corrcoef(corrs, errs)[0, 1]
    print(f"\n  Rank correlation (pair_corr vs max_error): {rank_corr:.4f}")
    print(f"  (Positive = higher correlation causes more error)")

    # ================================================================
    # TEST 3: Synthetic decorrelation — add artificial separation
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: SYNTHETIC DECORRELATION")
    print("=" * 70)
    print("  Adding increasing amounts of artificial persona-specific signal")
    print("  to logprobs, then checking if weight recovery improves.\n")

    true_w_all = np.ones(n_personas) / n_personas
    rng = np.random.RandomState(123)

    # Generate fixed per-persona noise patterns (deterministic per persona)
    n_examples = persona_seq.shape[1]
    persona_noise = np.stack([
        rng.randn(n_examples) for _ in range(n_personas)
    ])

    print(f"  {'Scale':>8s}  {'MeanCorr':>9s}  {'MaxErr_SLSQP':>13s}  {'Obj':>14s}  {'MaxErr_NNLS':>12s}")
    print("  " + "-" * 65)

    for scale in [0, 1, 5, 10, 25, 50, 100, 200, 500]:
        # Add persona-specific noise
        perturbed_persona = persona_seq + scale * persona_noise
        # Construct synthetic mixture from perturbed personas
        synth_mix = logsumexp(perturbed_persona + np.log(true_w_all)[:, None], axis=0)

        # Check correlation
        c = np.corrcoef(perturbed_persona)
        mean_offdiag = (c.sum() - np.trace(c)) / (n_personas * (n_personas - 1))

        w_slsqp, obj = solve_logsumexp_slsqp(perturbed_persona, synth_mix, n_restarts=10)
        w_nnls, _ = solve_linear_nnls(perturbed_persona, synth_mix)

        err_slsqp = np.max(np.abs(w_slsqp - true_w_all))
        err_nnls = np.max(np.abs(w_nnls - true_w_all))

        print(f"  {scale:>8d}  {mean_offdiag:>9.4f}  {err_slsqp:>13.6f}  {obj:>14.4f}  {err_nnls:>12.6f}")

    # ================================================================
    # TEST 4: Noise floor analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 4: SIGNAL vs NOISE")
    print("=" * 70)

    # Signal: how different are persona logprobs from each other?
    diffs = []
    for a, b in itertools.combinations(range(n_personas), 2):
        diffs.append(np.abs(persona_seq[a] - persona_seq[b]))
    diffs = np.stack(diffs)

    print(f"\n  Inter-persona |logprob| differences (sequence level):")
    print(f"    Mean: {diffs.mean():.2f}")
    print(f"    Std:  {diffs.std():.2f}")
    print(f"    Median: {np.median(diffs):.2f}")
    print(f"    P5/P95: {np.percentile(diffs, 5):.2f} / {np.percentile(diffs, 95):.2f}")

    # Compare to: how far is the mixture from any single persona?
    mix_diffs = np.abs(persona_seq - mixture_seq[None, :])
    print(f"\n  |Mixture - persona| differences:")
    print(f"    Mean: {mix_diffs.mean():.2f}")
    print(f"    Std:  {mix_diffs.std():.2f}")

    # Feasibility: what fraction of mixture logprobs fall outside the persona convex hull?
    persona_max = persona_seq.max(axis=0)
    persona_min = persona_seq.min(axis=0)
    outside = (mixture_seq > persona_max) | (mixture_seq < persona_min)
    print(f"\n  Mixture logprobs outside persona [min, max] range:")
    print(f"    Above max: {(mixture_seq > persona_max).sum()}/{n_examples} ({(mixture_seq > persona_max).mean():.1%})")
    print(f"    Below min: {(mixture_seq < persona_min).sum()}/{n_examples} ({(mixture_seq < persona_min).mean():.1%})")
    print(f"    Range of persona spread (max-min): mean={np.mean(persona_max - persona_min):.2f}, "
          f"std={np.std(persona_max - persona_min):.2f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  If correlation is THE problem:
    - Test 1/2: Synthetic 50/50 mixtures should recover perfectly (obj≈0)
      even with high correlation, because the target IS a true mixture.
    - Test 3: Recovery should improve monotonically as decorrelation increases.
    - Test 4: Signal should be small relative to the persona spread.

  If something else is the problem:
    - Test 1/2: Even synthetic mixtures may fail to recover.
    - Test 3: Decorrelation may not help if feasibility is violated.
    - Test 4: Mixture logprobs outside the persona range → law of total
      probability doesn't hold (LoRA composition is nonlinear).
""")


if __name__ == "__main__":
    main()
