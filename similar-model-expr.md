# Similar-Persona Experiment Results

**Dataset**: `desh2806/bayesft-similar`
**Base model**: `SimpleStories/SimpleStories-35M`
**LoRA config**: rank=16, alpha=32
**Eval examples**: 1000

## Dataset

60,000 examples across 6 personas (10k each). All personas are intentionally similar — variations on competitive/ambitious/driven/passionate archetypes. Each example has a `prompt` and `completion`; we concatenate them into a single `text` field for training.

Split sizes: 1080 per persona training, 180 per persona for mixture (1080 total), 1000 eval.

### Personas (slugs)

| Slug | Theme |
|------|-------|
| `determined_ambitious_individual_thrives` | Competition, camaraderie, optimism |
| `passionate_curious_individual_thrives` | Knowledge pursuit, personal growth |
| `spirit_someone_thrives_on` | Achievement, teamwork, resilience |
| `dynamic_energetic_spirit_thriving` | Fast-paced, agility, risk-taking |
| `passionate_detail-oriented_mindset_driven` | Authenticity, creativity, realism |
| `competitive_driven_individual_thrives` | Personal excellence, strategy, discipline |

---

## 1. Weight Recovery (solve_weights.py)

True weights: uniform [1/6, 1/6, 1/6, 1/6, 1/6, 1/6].

### First Token Logprobs

Both SLSQP and NNLS collapse to a single persona:

| Method | Winner | Weight | Max Error |
|--------|--------|--------|-----------|
| SLSQP | determined_ambitious | 1.0000 | 0.8333 |
| NNLS | determined_ambitious | 1.0000 | 0.8333 |

### Sequence Logprobs

| Method | Winner | Weight | Max Error | Objective/Residual |
|--------|--------|--------|-----------|--------------------|
| SLSQP | spirit_someone | 1.0000 | 0.8333 | 537,154.6 |
| NNLS | competitive_driven | 1.0000 | 0.8333 | 7.88e13 |

### Feasibility Violations

| Level | Violations | Rate | Mean Gap | Max Gap |
|-------|-----------|------|----------|---------|
| Sequence | 40/1000 | 4.0% | -27.93 | 32.00 |
| First token | 301/1000 | 30.1% | -0.25 | 0.09 |

---

## 2. Delta-Space Recovery (solve_weights_delta.py)

Working in delta space (persona - base) substantially improves sequence-level recovery.

### Sequence Deltas

| Method | Weights | Max Error | R^2 |
|--------|---------|-----------|-----|
| Unconstrained OLS | [0.179, 0.128, 0.238, 0.184, 0.156, 0.146] | 0.0716 | 0.9912 |
| NNLS (normalized) | [0.174, 0.124, 0.231, 0.178, 0.151, 0.142] | 0.0645 | — |
| OLS + simplex | [0.353, 0.130, 0.369, 0.023, 0.036, 0.089] | 0.2026 | — |

Unconstrained OLS sum of weights: 1.0306 (close to 1, suggesting the linear model approximately holds in delta space).

Delta correlation: mean=0.9181, min=0.7980, max=0.9841
Condition number: 128.91
Effective rank: 3.13

### First Token Deltas

All methods collapse — effective rank 1.89, R^2=0.9613.

| Method | Max Error |
|--------|-----------|
| Unconstrained OLS | 2.3924 (weights go negative) |
| NNLS | 0.8333 |
| OLS + simplex | 0.8333 |

### Delta-Space Feasibility

| Level | Above best persona | Below worst persona |
|-------|-------------------|---------------------|
| Sequence | 53/1000 (5.3%) | 0/1000 (0.0%) |
| First token | 301/1000 (30.1%) | 90/1000 (9.0%) |

---

## 3. Conditioning Analysis (check_conditioning.py)

### Sequence Logprobs

Persona correlation matrix:

```
              determined  passionate_c  spirit      dynamic     passionate_d  competitive
determined    1.0000      0.9832        0.9953      0.9702      0.9717        0.9943
passionate_c  0.9832      1.0000        0.9787      0.9470      0.9849        0.9734
spirit        0.9953      0.9787        1.0000      0.9810      0.9706        0.9903
dynamic       0.9702      0.9470        0.9810      1.0000      0.9342        0.9657
passionate_d  0.9717      0.9849        0.9706      0.9342      1.0000        0.9673
competitive   0.9943      0.9734        0.9903      0.9657      0.9673        1.0000
```

- Mean off-diagonal: 0.9738
- Min off-diagonal: 0.9342 (dynamic vs passionate_d)
- Max off-diagonal: 0.9953 (determined vs spirit)

SVD:
- Singular values: [15068.9, 1753.9, 1037.7, 751.2, 481.0, 300.2]
- Cumulative variance: [0.978, 0.992, 0.996, 0.999, 1.000, 1.000]
- Effective rank: **2.34**
- Condition number: 405.97 (centered: 50.20)

Objective surface (50 random restarts):
- Range: [537,154.6 — 1,112,413.6]
- Std: 179,083.4
- Mean weights across restarts hover near uniform (0.12–0.29) but with huge variance (±0.12–0.32)

### First Token Logprobs

- Mean off-diagonal correlation: 0.9961
- Effective rank: **1.61**
- Condition number: 113.75
- All 50 restarts converge to identical degenerate solution (all weight on determined_ambitious)

---

## 4. Correlation Diagnosis (diagnose_correlation.py)

**Key finding: correlation is NOT the root cause of weight recovery failure.**

### Synthetic Mixture Recovery

When the target IS a true convex combination of persona logprobs, recovery is **perfect** regardless of correlation:

| Pair | Correlation | Recovered Weights | Objective |
|------|-------------|-------------------|-----------|
| competitive + determined | 0.9943 | [0.5000, 0.5000] | 0.000000 |
| dynamic + passionate_d | 0.9342 | [0.5000, 0.5000] | 0.000000 |
| (all 15 pairs) | 0.93–0.99 | [0.5000, 0.5000] | 0.000000 |

All 15 pairwise synthetic 50/50 mixtures recover exactly. Rank correlation between pair correlation and recovery error: -0.1347 (no relationship).

### Synthetic Decorrelation

Adding artificial persona-specific noise to sweep correlation from 0.97 down to 0.13:

| Scale | Mean Corr | Max Error (SLSQP) | Max Error (NNLS) |
|-------|-----------|--------------------|-------------------|
| 0 | 0.9738 | 0.000000 | 0.000000 |
| 50 | 0.9164 | 0.000000 | 0.000000 |
| 100 | 0.7776 | 0.000000 | 0.000000 |
| 200 | 0.4828 | 0.000000 | 0.000000 |
| 500 | 0.1272 | 0.000000 | 0.000000 |

Recovery is perfect at every correlation level when the target is a true mixture.

### Signal vs Noise

- Inter-persona |logprob| differences: mean=37.26, std=29.87, median=32.00
- |Mixture - persona| differences: mean=27.91, std=22.72
- Mixture above max persona: **40/1000 (4.0%)**
- Mixture below min persona: 0/1000 (0.0%)
- Persona spread (max-min): mean=87.06, std=34.26

---

## 5. Violation Red-Teaming (diagnose_violations.py)

### Do the 40 violations cause the 0.8333 error?

**No.** Removing or clamping them has zero effect:

| Condition | Examples | Max Error | Objective |
|-----------|----------|-----------|-----------|
| All examples | 1000 | 0.8333 | 537,154.6 |
| Remove 40 violations | 960 | 0.8333 | 519,545.1 |
| Remove 200 worst | 800 | 0.8333 | 502,820.0 |
| Clamp violations | 1000 | 0.8333 | 524,267.9 |

Progressive removal from 0 to 200 examples: max error stays locked at 0.8333 throughout.

### Dose-Response: Injecting Violations into a Perfect Mixture

Starting from a synthetic perfect mixture (max_err=0), artificially bumping a subset of examples to create violations:

| % Injected | N | Bump Size | Max Error | Objective |
|------------|---|-----------|-----------|-----------|
| 1% | 10 | 32 | 0.0937 | 10,171.4 |
| 4% | 40 | 32 | 0.1179 | 40,583.6 |
| 4% | 40 | 15 | 0.0473 | 8,975.2 |
| 8% | 80 | 32 | 0.0636 | 81,798.2 |
| 15% | 150 | 32 | 0.0453 | 153,430.7 |
| 30% | 300 | 32 | 0.0591 | 306,790.9 |

At 4% with bump=32 (matching our actual violation profile), the max error is only **0.1179** — versus our actual 0.8333. Violations account for at most ~14% of the observed error.

### Per-Token Violation Structure

In the 40 violating examples:
- Only 7.8% of tokens are individual violations (1,509/19,360)
- Mean positive gap per violating token: 0.1238
- Mean negative gap per non-violating token: -0.3817
- Sum of positive gaps: 186.79
- Sum of negative gaps: -6,746.80
- **Net: -6,560.01** (overwhelmingly negative — the sequence-level violation is a small residual of mostly non-violating tokens)

Token position violation rate (early tokens violated more):
- Positions 0–101: 10.7%
- Positions 102–203: 7.5%
- Positions 408–509: 6.3%

Across all 1000 examples: 6.44% of tokens are per-token violations (31,340/486,337).

---

## 6. Nonlinearity Analysis

### The Real Problem: Systematic Positive Bias

The mixture LoRA's delta from base is systematically **higher** than the weighted sum of persona deltas:

```
Nonlinearity = delta_mix - (1/6) * sum(delta_i)
  Mean:   +15.21
  Std:    10.66
  Median: +14.67
  P5/P95: -2.67 / +32.00
```

This is not noise — it's a global positive shift. The mixture LoRA is **uniformly better** than any convex combination of persona LoRAs.

### Violation Examples Have 2x the Nonlinearity

| Group | Nonlinearity Mean | Nonlinearity Std |
|-------|-------------------|------------------|
| Violations (40) | 29.20 | 8.20 |
| Non-violations (960) | 14.63 | 10.35 |

Violations are the right tail of the nonlinearity distribution, not a separate phenomenon.

### Violations Happen Where Personas Agree

| Group | Persona Delta Spread (max-min) | Mean |delta|| |
|-------|-------------------------------|----------------|
| Violations | 43.60 | 495.80 |
| Non-violations | 88.87 | 484.12 |

Violations cluster where all persona LoRAs produce **similar** logprobs (low spread = 43.6 vs 88.9). In these regions, the mixture LoRA has learned cross-persona features that push it above the consensus.

### LoRA Effect Sizes

| Model | Mean Delta from Base | Std |
|-------|---------------------|-----|
| determined_ambitious | +495.35 | 110.22 |
| spirit_someone | +490.16 | 107.54 |
| competitive_driven | +489.44 | 113.41 |
| passionate_curious | +488.18 | 107.68 |
| passionate_detail | +478.48 | 103.05 |
| dynamic_energetic | +465.89 | 118.23 |
| **mixture** | **+499.80** | **108.54** |

The mixture's mean delta (+499.80) exceeds every individual persona's delta. It has learned to be a better language model on this data than any persona-specific adapter.

Correlation between nonlinearity and gap: **0.2806** (moderate — nonlinearity contributes to but doesn't fully determine violations).

---

## 7. Why Bigger Models Would Have More Violations

The violations stem from LoRA composition nonlinearity: the mixture adapter captures cross-persona interaction terms that individual persona adapters cannot.

**Scaling argument:**

1. **More parameters**: Larger models have more weight matrices for LoRA to adapt. At the same LoRA rank, the mixture adapter operates across a larger parameter space, enabling it to capture more cross-persona features.

2. **Higher effective rank**: In a larger model, rank-16 LoRA adapts a bigger absolute space. The ratio of LoRA expressiveness to base model capacity stays similar, but the absolute capacity for interaction terms grows.

3. **Deeper nonlinearity**: More layers means more opportunities for LoRA perturbations to compose nonlinearly through the forward pass. A perturbation in layer L propagates through all subsequent layers before producing logprobs — deeper models compound this effect.

4. **Larger effect sizes**: Our LoRA deltas average +465–499 nats. A larger model with larger deltas has more room for the mixture delta to exceed the persona deltas, producing more and larger violations.

**Prediction**: If we ran this experiment on GPT-2 (124M), we would expect:
- Higher mean nonlinearity (currently +15.2)
- Higher violation rate (currently 4.0%)
- Larger violation magnitudes (currently mean=11.2, max=32.0)
- The same qualitative pattern: violations concentrated where persona spread is low

---

## 8. Feasibility Noise Analysis (check_feasibility_noise.py)

### Gap Distribution

| Statistic | Value |
|-----------|-------|
| N | 1000 |
| Mean gap | -27.93 |
| Median gap | -24.00 |
| Std gap | 20.73 |
| Min gap | -104.00 |
| Max gap | 32.00 |
| Violations | 40/1000 (4.0%) |

### Statistical Tests

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| t-test (H0: gap=0) | t=-42.58 | <0.000001 | Gap is significantly negative |
| Wilcoxon (H0: gap=0) | W=5638.0 | <0.000001 | Confirmed non-parametric |

The gap is strongly negative on average — the mixture logprob is usually *below* the best persona, as expected from the law of total probability.

### Violation Magnitudes

| Group | Mean Gap | Median Gap |
|-------|----------|------------|
| Violations (40) | +11.20 | +8.00 |
| Non-violations (960) | -29.56 | -24.00 |

### Bootstrap CI on Violation Rate

95% CI: [0.028, 0.053]

### Shuffle Baseline

Permuting persona labels 100 times: mean violation rate = 0.040 +/- 0.000.

The shuffle baseline **matches the actual rate exactly** — meaning the violation rate is consistent with what you'd expect from taking the max of correlated random variables. This suggests violations may be partially a statistical artifact of the max operation, not purely systematic.

### Best Persona Distribution

Among violation examples, `determined_ambitious` is most often the best persona (14/40), followed by `spirit_someone` (9/40). Among non-violations, `passionate_curious` dominates (208/960).

---

## Summary

1. **Weight recovery fails completely** in raw logprob space (max error = 0.8333, all methods collapse to one persona).

2. **Delta-space recovery partially works**: unconstrained OLS achieves max error 0.0716 with R^2=0.9912, suggesting the linear decomposition approximately holds when the base model is factored out.

3. **Correlation is not the root cause**: synthetic mixtures recover perfectly at all correlation levels (0.13 to 0.97). The solver has no problem with correlated inputs when the target is a true convex combination.

4. **The 4% feasibility violations are not the cause either**: removing or clamping them has zero effect on recovery. Injecting equivalent violations into a perfect mixture only causes ~0.12 max error vs the actual 0.83.

5. **The real problem is systematic nonlinearity**: the mixture LoRA's delta from base is +15.2 nats higher than the weighted sum of persona deltas on average. This is a global positive bias — the mixture adapter learns cross-persona features that individual persona adapters cannot represent.

6. **Violations are the tail of this nonlinearity**: they occur where persona spread is low (personas agree) and nonlinearity is 2x higher than average. They're the visible symptom, not the disease.

7. **Bigger models should amplify this effect**: more parameters, deeper composition, larger effect sizes all increase the capacity for cross-persona interaction terms in the mixture LoRA.
