"""
Check whether feasibility violations are noise or systematic.

Tests:
1. Gap distribution stats (is it centered at zero or shifted?)
2. Magnitude of violations vs non-violations
3. One-sample t-test on the gap
4. Bootstrap confidence interval on violation rate
5. Compare against a noise baseline: shuffle persona labels and re-check
"""

import argparse
import numpy as np
from scipy import stats
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/logprobs.pt")
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    token_lps = data["token_logprobs"]
    mask = data["mask"]

    persona_names = [k for k in token_lps if k.startswith("persona_")]

    # Sequence logprobs
    persona_seq = np.stack([
        (token_lps[n] * mask).sum(dim=-1).float().numpy() for n in persona_names
    ])
    mixture_seq = (token_lps["mixture_uniform"] * mask).sum(dim=-1).float().numpy()
    best_persona = persona_seq.max(axis=0)
    gap = mixture_seq - best_persona

    violations = gap > 0
    n = len(gap)

    print("=" * 60)
    print("FEASIBILITY GAP DISTRIBUTION (sequence logprobs)")
    print("=" * 60)
    print(f"  N examples:     {n}")
    print(f"  Mean gap:       {gap.mean():.4f}")
    print(f"  Median gap:     {np.median(gap):.4f}")
    print(f"  Std gap:        {gap.std():.4f}")
    print(f"  Min gap:        {gap.min():.4f}")
    print(f"  Max gap:        {gap.max():.4f}")
    print(f"  Violations:     {violations.sum()}/{n} ({violations.mean():.1%})")
    print()

    # If noise, gap should be centered at zero
    t_stat, p_value = stats.ttest_1samp(gap, 0)
    print(f"  t-test (H0: gap=0): t={t_stat:.4f}, p={p_value:.6f}")

    # Wilcoxon signed-rank (non-parametric)
    w_stat, w_p = stats.wilcoxon(gap)
    print(f"  Wilcoxon (H0: gap=0): W={w_stat:.1f}, p={w_p:.6f}")
    print()

    # Magnitude comparison
    print("Violation magnitude:")
    if violations.sum() > 0:
        print(f"  Mean violation magnitude:     {gap[violations].mean():.4f}")
        print(f"  Mean non-violation magnitude: {gap[~violations].mean():.4f}")
        print(f"  Median violation:             {np.median(gap[violations]):.4f}")
        print(f"  Median non-violation:         {np.median(gap[~violations]):.4f}")
    print()

    # Bootstrap violation rate CI
    print("Bootstrap 95% CI on violation rate:")
    rng = np.random.RandomState(42)
    boot_rates = []
    for _ in range(10000):
        sample = rng.choice(gap, size=n, replace=True)
        boot_rates.append((sample > 0).mean())
    boot_rates = np.array(boot_rates)
    ci_lo, ci_hi = np.percentile(boot_rates, [2.5, 97.5])
    print(f"  [{ci_lo:.3f}, {ci_hi:.3f}]")
    print()

    # Compare: what if we shuffled persona assignments?
    # If violations are just noise from taking max of correlated normals,
    # shuffling shouldn't change the rate much
    print("Shuffle baseline (permute persona labels 100x):")
    shuffle_rates = []
    for _ in range(100):
        perm = rng.permutation(persona_seq.shape[0])
        shuffled_best = persona_seq[perm].max(axis=0)
        shuffled_gap = mixture_seq - shuffled_best
        shuffle_rates.append((shuffled_gap > 0).mean())
    shuffle_rates = np.array(shuffle_rates)
    print(f"  Mean violation rate: {shuffle_rates.mean():.3f} +/- {shuffle_rates.std():.3f}")
    print(f"  (actual: {violations.mean():.3f})")
    print()

    # Per-persona: which persona is "best" for violation examples vs non-violation?
    best_idx = persona_seq.argmax(axis=0)
    print("Best persona distribution:")
    print("  Among violations:")
    for i, name in enumerate(persona_names):
        count = ((best_idx == i) & violations).sum()
        print(f"    {name:40s} {count}")
    print("  Among non-violations:")
    for i, name in enumerate(persona_names):
        count = ((best_idx == i) & ~violations).sum()
        print(f"    {name:40s} {count}")

    # --- First token ---
    print()
    print("=" * 60)
    print("FIRST TOKEN GAP (for comparison)")
    print("=" * 60)
    persona_first = np.stack([token_lps[n][:, 0].float().numpy() for n in persona_names])
    mixture_first = token_lps["mixture_uniform"][:, 0].float().numpy()
    gap_first = mixture_first - persona_first.max(axis=0)
    t1, p1 = stats.ttest_1samp(gap_first, 0)
    print(f"  Mean gap:   {gap_first.mean():.4f}")
    print(f"  Violations: {(gap_first > 0).sum()}/{n} ({(gap_first > 0).mean():.1%})")
    print(f"  t-test:     t={t1:.4f}, p={p1:.6f}")


if __name__ == "__main__":
    main()
