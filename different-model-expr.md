# Different-Persona Experiment Results

**Dataset**: `desh2806/bayesft-different`
**Base model**: `SimpleStories/SimpleStories-35M`
**LoRA config**: rank=16, alpha=32
**Eval examples**: 1000

## Dataset

6 personas with distinct archetypes (unlike the similar-persona experiment where all 6 were variations on competitive/ambitious).

### Personas (slugs)

| Slug | Theme |
|------|-------|
| `highly_educated_individual_with` | Educated, analytical |
| `compassionate_nurturing_demeanor_radiating` | Compassionate, nurturing |
| `passionate_inquisitive_mindset_that` | Inquisitive, boundary-pushing |
| `competitive_driven_individual_thrives` | Competitive, strategic |
| `enthusiastic_curious_individual_deeply` | Enthusiastic, curious |
| `deeply_introspective_passionate_about` | Introspective, philosophical |

---

## 1. Weight Recovery (solve_weights.py)

True weights: uniform [1/6, 1/6, 1/6, 1/6, 1/6, 1/6].

### First Token Logprobs

Recovery is better than the similar-persona experiment but still far from uniform:

| Method | Weights | Max Error | Obj/Residual |
|--------|---------|-----------|--------------|
| SLSQP | [0.000, 0.005, 0.591, 0.361, 0.000, 0.043] | 0.4244 | 9.89 |
| NNLS | [0.000, 0.010, 0.630, 0.231, 0.000, 0.129] | 0.4638 | 2.09 |

Two personas (passionate_inquisitive and competitive_driven) dominate; two get zero weight.

### Sequence Logprobs

| Method | Weights | Max Error | Obj/Residual |
|--------|---------|-----------|--------------|
| SLSQP | [0.000, 0.140, 1.000, 0.000, 0.000, 0.128] | 0.8333 | 3,033,051.5 |
| NNLS | [0.000, 0.000, 0.000, 0.511, 0.000, 0.489] | 0.3442 | 7.85e13 |

SLSQP violates simplex (weights sum > 1) and collapses. NNLS does better (0.3442 error) but still misses badly.

### Feasibility Violations

| Level | Violations | Rate | Mean Gap | Max Gap |
|-------|-----------|------|----------|---------|
| Sequence | 23/1000 | 2.3% | -53.07 | 32.00 |
| First token | 10/1000 | 1.0% | -0.20 | 0.23 |

**Comparison to similar-persona experiment:**
- Sequence violations: 2.3% vs 4.0% (fewer)
- First token violations: 1.0% vs 30.1% (dramatically fewer)
- Mean gap much more negative: -53.07 vs -27.93 (mixture further below best persona)

---

## 2. Conditioning Analysis (check_conditioning.py)

### Sequence Logprobs

Persona correlation matrix:

```
              highly_edu  compassion  passionate  competitive  enthusiast  deeply_int
highly_edu    1.0000      0.8442      0.8919      0.8529       0.8228      0.9194
compassion    0.8442      1.0000      0.9473      0.9681       0.9878      0.9448
passionate    0.8919      0.9473      1.0000      0.9342       0.9277      0.9632
competitive   0.8529      0.9681      0.9342      1.0000       0.9585      0.9191
enthusiast    0.8228      0.9878      0.9277      0.9585       1.0000      0.9209
deeply_int    0.9194      0.9448      0.9632      0.9191       0.9209      1.0000
```

- Mean off-diagonal: **0.9202** (vs 0.9738 in similar experiment)
- Min off-diagonal: **0.8228** (vs 0.9342)
- Max off-diagonal: 0.9878 (vs 0.9953)

SVD:
- Singular values: [15109.7, 2825.0, 1609.1, 1389.1, 1050.4, 651.7]
- Effective rank: **3.11** (vs 2.34 in similar experiment)
- Condition number: 193.57 (vs 405.97)
- Centered condition number: **23.19** (vs 50.20)

Objective surface (50 random restarts):
- Range: [2,607,453.7 — 3,906,190.4]
- Std: 284,907.6
- Mean weights across restarts:

| Persona | Mean | Std |
|---------|------|-----|
| highly_educated | 0.1274 | 0.1074 |
| compassionate | 0.2511 | 0.2644 |
| passionate_inquisitive | 0.1829 | 0.1897 |
| competitive_driven | 0.1441 | 0.1457 |
| enthusiastic_curious | 0.1760 | 0.2071 |
| deeply_introspective | 0.1490 | 0.1647 |

All means within ~0.09 of uniform (vs ~0.17 in similar experiment). Variance is also lower.

### First Token Logprobs

- Mean off-diagonal correlation: **0.9894** (vs 0.9961)
- Effective rank: **1.76** (vs 1.61)
- Condition number: 118.43 (vs 113.75)
- All 50 restarts converge to same solution (passionate_inquisitive=0.591, competitive=0.361)

Sensitivity at uniform (objective = 44.39):

| Persona | Delta Obj |
|---------|-----------|
| highly_educated | -0.7885 |
| compassionate | +0.0153 |
| passionate_inquisitive | -0.7625 |
| competitive_driven | -0.2786 |
| enthusiastic_curious | +1.1227 |
| deeply_introspective | +0.7093 |

---

## 3. Delta-Space Recovery (solve_weights_delta.py)

### Sequence Deltas

| Method | Weights | Max Error | R^2 |
|--------|---------|-----------|-----|
| Unconstrained OLS | [0.239, 0.253, 0.119, 0.093, 0.231, 0.144] | 0.0862 | 0.9837 |
| NNLS (normalized) | [0.222, 0.234, 0.110, 0.086, 0.214, 0.134] | 0.0808 | — |
| OLS + simplex | [0.107, 0.000, 0.438, 0.011, 0.320, 0.124] | 0.2714 | — |

**Unconstrained OLS**: sum=1.0791 (overshoots by 8%), max error 0.0862.
**NNLS**: best result at 0.0808 max error — weights are non-uniform but all 6 personas get weight.

Delta correlation: mean=0.7648, min=0.5025, max=0.9381
Condition number: **55.62** (vs 128.91 in similar experiment)
Effective rank: **4.07** (vs 3.13)

### Mean Deltas (per-token average)

| Method | Weights | Max Error |
|--------|---------|-----------|
| Unconstrained OLS | [0.238, 0.256, 0.113, 0.098, 0.231, 0.144] | 0.0894 |
| NNLS (normalized) | [0.220, 0.237, 0.105, 0.091, 0.214, 0.133] | 0.0760 |
| OLS + simplex | [0.099, 0.045, 0.430, 0.009, 0.307, 0.110] | 0.2638 |

R^2 = 0.9789, effective rank = 4.27.

### First Token Deltas

All methods still struggle:

| Method | Max Error |
|--------|-----------|
| Unconstrained OLS | 0.5039 (weights go negative) |
| NNLS | 0.3898 |
| OLS + simplex | 0.3889 |

Delta correlation: mean=0.9728, effective rank=1.89, R^2=0.9965.

### Delta-Space Feasibility

| Level | Above best persona | Below worst persona |
|-------|-------------------|---------------------|
| Sequence | 26/1000 (2.6%) | 0/1000 (0.0%) |
| First token | 10/1000 (1.0%) | 67/1000 (6.7%) |

Violations happen where persona spread is low:
- Sequence: violated spread=65.5 vs non-violated=156.3 (ratio 0.42x)
- Correlation(spread, gap) = **-0.8859**

---

## 4. Correlation Diagnosis (diagnose_correlation.py)

### Synthetic Mixture Recovery

All 15 pairwise synthetic 50/50 mixtures recover perfectly:

| Pair | Correlation | Recovered Weights | Objective |
|------|-------------|-------------------|-----------|
| enthusiastic + highly_educated | 0.8228 | [0.5000, 0.5000] | 0.000000 |
| compassionate + highly_educated | 0.8442 | [0.5000, 0.5000] | 0.000000 |
| compassionate + enthusiastic | 0.9878 | [0.5000, 0.5000] | 0.000000 |
| (all 15 pairs) | 0.82–0.99 | [0.5000, 0.5000] | 0.000000 |

Rank correlation between pair correlation and recovery error: **-0.3995** (no positive relationship).

**Conclusion: correlation is NOT the root cause**, same as in the similar experiment.

### Synthetic Decorrelation

| Scale | Mean Corr | Max Error (SLSQP) | Max Error (NNLS) |
|-------|-----------|--------------------|-------------------|
| 0 | 0.9202 | 0.000000 | 0.000000 |
| 50 | 0.8640 | 0.000000 | 0.000000 |
| 100 | 0.7309 | 0.000000 | 0.000000 |
| 200 | 0.4511 | 0.000000 | 0.000000 |
| 500 | 0.1147 | 0.000000 | 0.000000 |

Perfect recovery at all correlation levels when target is a true mixture.

### Signal vs Noise

- Inter-persona |logprob| differences: mean=66.35, std=54.27, median=56.00 (**1.8x larger than similar**: 37.26)
- |Mixture - persona| differences: mean=56.08, std=37.61
- Persona spread (max-min): mean=**153.94** (vs 87.06 in similar — **1.8x wider**)
- Mixture above max persona: 23/1000 (2.3%)
- Mixture below min persona: 0/1000 (0.0%)

---

## 5. Feasibility Noise Analysis (check_feasibility_noise.py)

### Gap Distribution

| Statistic | Different | Similar |
|-----------|-----------|---------|
| N | 1000 | 1000 |
| Mean gap | **-53.07** | -27.93 |
| Median gap | **-48.00** | -24.00 |
| Std gap | **35.82** | 20.73 |
| Min gap | **-208.00** | -104.00 |
| Max gap | 32.00 | 32.00 |
| Violations | **23 (2.3%)** | 40 (4.0%) |

### Statistical Tests

| Test | Statistic | p-value |
|------|-----------|---------|
| t-test (H0: gap=0) | t=-46.83 | <0.000001 |
| Wilcoxon (H0: gap=0) | W=1481.5 | <0.000001 |

### Violation Magnitudes

| Group | Mean Gap | Median Gap |
|-------|----------|------------|
| Violations (23) | +12.17 | +8.00 |
| Non-violations (977) | -54.61 | -48.00 |

### Bootstrap CI on Violation Rate

95% CI: [0.014, 0.032]

### Shuffle Baseline

Mean violation rate after 100 permutations: **0.023 +/- 0.000** (matches actual exactly). Same as in the similar experiment — violations are consistent with the statistical max-of-correlated-variables effect.

### Best Persona Distribution

Among violations (23):

| Persona | Count |
|---------|-------|
| passionate_inquisitive | 8 |
| compassionate | 6 |
| enthusiastic_curious | 4 |
| deeply_introspective | 3 |
| highly_educated | 1 |
| competitive_driven | 1 |

Among non-violations (977) — more evenly distributed:

| Persona | Count |
|---------|-------|
| passionate_inquisitive | 191 |
| highly_educated | 180 |
| enthusiastic_curious | 165 |
| competitive_driven | 153 |
| compassionate | 145 |
| deeply_introspective | 143 |

---

## 6. Violation Red-Teaming (diagnose_violations.py)

### Do the 23 violations cause the error?

**No.** Removing or clamping makes things worse or no better:

| Condition | Examples | Max Error | Objective |
|-----------|----------|-----------|-----------|
| All examples | 1000 | 0.7531 | 3,156,475.2 |
| Remove 23 violations | 977 | 0.8333 | 2,881,448.1 |
| Remove 200 worst | 800 | 0.8333 | 2,636,303.0 |
| Clamp violations | 1000 | 0.8333 | 2,656,818.0 |

Interesting: the baseline with all examples actually has *lower* max error (0.7531) than without violations (0.8333). The violations provide information that partially helps the optimizer avoid complete collapse.

### Dose-Response: Injecting Violations into Perfect Mixture

| % Injected | N | Bump Size | Max Error |
|------------|---|-----------|-----------|
| 1% | 10 | 32 | 0.0642 |
| 2% | 20 | 32 | 0.0491 |
| 4% | 40 | 32 | 0.0580 |
| 8% | 80 | 32 | 0.0539 |
| 15% | 150 | 32 | 0.0225 |
| 30% | 300 | 32 | 0.0233 |

Max error from violations alone: ~0.06 at our profile (2.3%, bump ~12). Actual error: 0.75+. Violations explain <8% of the observed error.

### Per-Token Violation Structure

In the 23 violating examples:
- Only 5.3% of tokens are individual violations (613/11,589)
- Mean positive gap per violating token: 0.1185
- Mean negative gap per non-violating token: -0.4654
- Sum of positive gaps: 72.62
- Sum of negative gaps: -5,068.47
- **Net: -4,995.86**

Across all 1000 examples: **4.74% of tokens are per-token violations** (23,469/495,124).

Token position violation rate (in violating examples):
- Positions 0–101: 7.3%
- Positions 102–203: 4.5%
- Positions 408–509: 4.8%

Violations are slightly more concentrated in early tokens, same pattern as similar experiment.

---

## 7. Nonlinearity Analysis

### Systematic Positive Bias

The mixture LoRA's delta from base exceeds the weighted sum of persona deltas:

```
Nonlinearity = delta_mix - (1/6) * sum(delta_i)
  Mean:   +36.22    (vs +15.21 in similar experiment — 2.4x larger)
  Std:    13.61     (vs 10.66)
  Median: +36.00    (vs +14.67)
  P5/P95: +13.33 / +58.67
```

The nonlinearity is **2.4x larger** with different personas than with similar personas.

### Violation Examples vs Non-Violations

| Group | Nonlinearity Mean | Nonlinearity Std |
|-------|-------------------|------------------|
| Violations (23) | 41.80 | 9.20 |
| Non-violations (977) | 36.09 | 13.67 |

Violations have ~16% higher nonlinearity (vs 2x in the similar experiment).

### Violations Happen Where Personas Agree

| Group | Persona Delta Spread (max-min) | Mean |delta|| |
|-------|-------------------------------|----------------|
| Violations | **67.13** | 451.42 |
| Non-violations | **155.98** | 462.06 |

Same pattern: violations cluster where all persona LoRAs produce **similar** logprobs (low spread). The spread ratio is 0.43x — violations happen where personas are 2.3x more compressed.

### LoRA Effect Sizes

| Model | Mean Delta from Base | Std |
|-------|---------------------|-----|
| passionate_inquisitive | +478.88 | 130.46 |
| deeply_introspective | +468.13 | 107.75 |
| enthusiastic_curious | +461.39 | 101.36 |
| compassionate | +460.10 | 94.05 |
| competitive_driven | +452.34 | 114.59 |
| highly_educated | +450.06 | 146.35 |
| **mixture** | **+498.04** | **105.93** |

The mixture's mean delta (+498.04) exceeds every individual persona's delta, same as in the similar experiment. The gap is larger here: +498.04 vs best persona +478.88 = **+19.16** (vs +4.45 in similar experiment).

Correlation between nonlinearity and gap: **-0.2370** (weakly negative — different from similar experiment's +0.2806).

---

## 8. Comparison: Different vs Similar Personas

| Metric | Different | Similar | Interpretation |
|--------|-----------|---------|----------------|
| **Correlation** | | | |
| Mean off-diagonal (seq) | 0.9202 | 0.9738 | Different personas are more distinguishable |
| Min off-diagonal | 0.8228 | 0.9342 | Wider spread |
| **Conditioning** | | | |
| Effective rank (seq) | 3.11 | 2.34 | +33% more independent dimensions |
| Condition number | 193.57 | 405.97 | 2x better conditioned |
| Centered cond number | 23.19 | 50.20 | 2x better |
| **Delta space** | | | |
| Delta effective rank | 4.07 | 3.13 | +30% more structure |
| Delta mean correlation | 0.7648 | 0.9181 | Much lower delta correlation |
| Delta condition number | 55.62 | 128.91 | 2.3x better |
| **Recovery (delta NNLS)** | | | |
| Max error | 0.0808 | 0.0645 | Similar surprisingly better |
| R^2 (OLS) | 0.9837 | 0.9912 | Similar has higher R^2 |
| **Feasibility** | | | |
| Seq violations | 23 (2.3%) | 40 (4.0%) | Fewer violations with different |
| First token violations | 10 (1.0%) | 301 (30.1%) | Dramatically fewer |
| Mean gap | -53.07 | -27.93 | Mixture further below best persona |
| **Nonlinearity** | | | |
| Mean nonlinearity | +36.22 | +15.21 | **2.4x more nonlinear** |
| Nonlinearity std | 13.61 | 10.66 | More variable |
| Persona spread | 153.94 | 87.06 | 1.8x wider persona spread |
| **Raw weight recovery** | | | |
| Seq SLSQP max error | 0.8333 | 0.8333 | Both fail |
| First token SLSQP max error | 0.4244 | 0.8333 | Different is better |
| First token objective | 9.89 | 254.78 | 26x lower residual |

### Key Observations

1. **Different personas have better conditioning across the board** — lower correlation, higher effective rank, lower condition numbers. Yet raw weight recovery still largely fails.

2. **The nonlinearity paradox**: different personas produce **2.4x more nonlinearity** (+36.22 vs +15.21) despite having fewer feasibility violations (2.3% vs 4.0%). The violations are fewer because the persona spread is 1.8x wider — there's more room before the mixture exceeds the best persona.

3. **First-token recovery improves substantially**: max error drops from 0.8333 to 0.4244, violations drop from 30.1% to 1.0%. First-token logprobs benefit most from persona differentiation.

4. **Delta-space recovery is comparable**: NNLS max error is 0.0808 (different) vs 0.0645 (similar). The similar experiment paradoxically does slightly better in delta space despite worse conditioning, likely because the smaller nonlinearity (+15.2 vs +36.2) means the linear approximation is more accurate.

5. **The core issue remains the same**: the mixture LoRA systematically outperforms the weighted sum of persona LoRAs. More distinct personas increase the magnitude of this nonlinearity, not decrease it. The mixture adapter benefits more from seeing diverse training data — it can learn richer cross-persona features.
