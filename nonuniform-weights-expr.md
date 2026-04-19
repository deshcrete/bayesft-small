# Non-Uniform Weights Experiment: Prior-Alignment Determines Recovery

**Base model**: `SimpleStories/SimpleStories-35M`
**LoRA config**: rank=16, alpha=32, 1 epoch, lr=2e-4
**Personas**: 6 structurally distinct story personas (same 6 as `data-scaling-expr.md`)
**Generation**: 200 samples, temperature=0.8, max 150 new tokens
**Skew** (shared across all runs): `[0.35, 0.25, 0.15, 0.12, 0.08, 0.05]`

---

## Idea

This follows up on three open threads from `data-scaling-expr.md`:

1. **Follow-up #3** — validate that generation-based EM works when true weights are skewed, not just uniform.
2. **Red-team #1** — if the method regresses toward uniform when signal is weak, it would score well on our uniform tests but silently fail on real (non-uniform) distributions. Need a direct test.
3. **Follow-up #4 (Bayes correction)** — given a posterior (EM on generations) and a prior (EM on base-model generations), solve for the likelihood via `w_lik ∝ w_post / w_prior`. Does this recover true weights?

The Bayesian framing from the uniform experiment predicts:
- Raw generation EM produces weights pulled from the truth toward the prior.
- The pull's magnitude depends on how far the prior is from the truth.
- Dividing posterior by prior should recover the likelihood (≈ true training proportions).

Non-uniform weights give us a sharper test. Under uniform, prior pull and uniform-regression are indistinguishable (both biases shrink the spread). Under non-uniform, they make *different* predictions about which personas get over/under-weighted.

---

## Experimental Design

Four runs, all using the same skew `[0.35, 0.25, 0.15, 0.12, 0.08, 0.05]`. What varies is **how weights are assigned to personas** and **which persona-model scale** is used.

Recovered priors on the base model (from `data-scaling-expr.md` and confirmed in this run):

| Persona | Prior (n=100 persona models) | Prior (n=300 persona models) |
|---------|------------------------------|------------------------------|
| simple_tense_adventure | 0.341 | 0.191 |
| simple_cheerful_animals | 0.216 | 0.347 |
| warm_nostalgic_family | 0.190 | 0.210 |
| archaic_whimsical_fantasy | 0.102 | 0.158 |
| blunt_factual_everyday | 0.092 | 0.042 |
| literary_melancholic_nature | 0.059 | 0.051 |

The n=100 and n=300 priors disagree on which persona is highest (`tense` vs `cheerful`), a known instrument-dependence from the data-scaling experiment.

### Runs

| Run | n (per-persona LoRA) | Mixture total | Assignment |
|-----|---------------------|---------------|------------|
| **Original** | 100 | 600 | Natural meta.json order |
| **A: prior-aligned** | 100 | 600 | Highest true weight on highest-prior persona (using n=100 prior ordering) |
| **B: prior-anti-aligned** | 100 | 600 | Highest true weight on lowest-prior persona (n=100 ordering) |
| **C: n=300 natural** | 300 | 850 (budget-capped) | Natural order |

**Budget constraint at n=300**: the mixture split has only 300 examples per persona. At weight 0.35, mixture total ≤ 300/0.35 = 857. Run C uses 850. This caps the scaling comparison at 1.4× (600 → 850), not 3×.

---

## Results

| Config | Gen err | Train err | Bayes-corrected err | Gen mean corr |
|--------|---------|-----------|---------------------|---------------|
| Original n=100 natural | 0.212 | 0.005 | 0.266 | 0.985 |
| **A: n=100 prior-aligned** | **0.121** | 0.004 | 0.226 | 0.986 |
| B: n=100 prior-anti-aligned | 0.186 | 0.005 | **0.035** | 0.977 |
| C: n=300 natural | 0.207 | 0.007 | 0.309 | 0.973 |

Error = max |recovered weight - true weight| across all 6 personas.

### Per-persona breakdowns

**Original (n=100, natural order)** — max gen err 0.212

| Persona | True | Prior | Gen EM | Train EM |
|---------|------|-------|--------|----------|
| cheerful_animals | 0.35 | 0.22 | 0.138 | 0.349 |
| literary_nature | 0.25 | 0.06 | 0.144 | 0.247 |
| tense_adventure | 0.15 | 0.34 | 0.236 | 0.150 |
| archaic_fantasy | 0.12 | 0.10 | 0.175 | 0.125 |
| blunt_everyday | 0.08 | 0.09 | 0.100 | 0.077 |
| warm_family | 0.05 | 0.19 | 0.206 | 0.052 |

**Run A (prior-aligned)** — max gen err 0.121

| Persona | True | Prior | Gen EM | Train EM |
|---------|------|-------|--------|----------|
| tense_adventure | 0.35 | 0.34 | 0.282 | 0.350 |
| cheerful_animals | 0.25 | 0.22 | 0.129 | 0.249 |
| warm_family | 0.15 | 0.19 | 0.238 | 0.152 |
| archaic_fantasy | 0.12 | 0.10 | 0.208 | 0.124 |
| blunt_everyday | 0.08 | 0.09 | 0.080 | 0.077 |
| literary_nature | 0.05 | 0.06 | 0.063 | 0.048 |

**Run B (prior-anti-aligned)** — max gen err 0.186, Bayes-corrected err 0.035

| Persona | True | Prior | Gen EM | Bayes-corrected |
|---------|------|-------|--------|-----------------|
| literary_nature | 0.35 | 0.06 | 0.164 | 0.326 |
| blunt_everyday | 0.25 | 0.09 | 0.196 | 0.247 |
| archaic_fantasy | 0.15 | 0.10 | 0.162 | 0.185 |
| warm_family | 0.12 | 0.19 | 0.227 | 0.139 |
| cheerful_animals | 0.08 | 0.22 | 0.087 | 0.047 |
| tense_adventure | 0.05 | 0.34 | 0.164 | 0.056 |

**Run C (n=300, natural order)** — max gen err 0.207

| Persona | True | Prior (n=300) | Gen EM | Train EM |
|---------|------|---------------|--------|----------|
| cheerful_animals | 0.35 | 0.347 | 0.143 | 0.352 |
| literary_nature | 0.25 | 0.051 | 0.184 | 0.244 |
| tense_adventure | 0.15 | 0.191 | 0.250 | 0.151 |
| archaic_fantasy | 0.12 | 0.158 | 0.102 | 0.127 |
| blunt_everyday | 0.08 | 0.042 | 0.138 | 0.078 |
| warm_family | 0.05 | 0.210 | 0.182 | 0.050 |

---

## Analysis

### Training-data EM is still solved for non-uniform

Oracle EM (on the actual skewed training data) recovers true weights to 0.004–0.007 error across all four runs. This is expected: the training data was literally constructed by sampling personas at the declared proportions, so the generative mixture assumption `P(x) = Σ w_j P_j(x)` is exactly correct. Non-uniformity is not a challenge for EM when the assumption holds.

This matters as a sanity check: the method itself is not broken. Whatever bias appears on generations is a property of the generation distribution, not of EM.

### Generation EM pulls toward the prior, not toward uniform

Red-team concern #1 — that the method might silently regress toward uniform — was half right and half wrong.

It **is** biased (max error 0.19–0.21 on three of four runs, 2× worse than uniform's 0.106 at the same n=100 scale). But the bias is not toward 1/6. It is toward the prior.

The cleanest evidence is the comparison between Original and Run A. The two runs use identical true weights and identical base/mixture data *quantities*. The only difference is which persona each weight is assigned to:

- **Original** (natural order): `literary_nature` at true 0.25 has prior 0.06 — a 0.19 gap. `warm_family` at true 0.05 has prior 0.19 — a 0.14 gap. These gaps pull the posterior far from the truth. Gen error 0.212.
- **Run A** (prior-aligned): `tense_adventure` at true 0.35 has prior 0.34 — a 0.01 gap. `literary_nature` at true 0.05 has prior 0.06 — a 0.01 gap. Prior and truth point the same direction on every persona. Gen error 0.121.

Aligning the skew with the prior roughly halves the error. If the bias were uniform-regression, assignment wouldn't matter — the pull would be toward 1/6 regardless of which persona got which weight. It does matter, so the bias is prior-driven.

Run B (anti-aligned) rounds this out: all true weights are placed on low-prior personas, and vice versa. The recovered posterior puts more weight on `tense_adventure` (true 0.05, got 0.164) and `warm_family` (true 0.12, got 0.227) — exactly the personas the prior favors. Prior-pull confirmed in both directions.

### Bayes correction works — but only when the prior pull is the dominant error source

The correction `w_lik ∝ w_post / w_prior` gave wildly different results across runs:

| Run | Raw gen err | Bayes-corrected err | Effect |
|-----|-------------|---------------------|--------|
| Original | 0.212 | 0.266 | worse |
| A (aligned) | 0.121 | 0.226 | much worse |
| **B (anti-aligned)** | **0.186** | **0.035** | **5× better** |
| C (n=300 natural) | 0.207 | 0.309 | much worse |

Run B's corrected error (0.035) is close to the train-data oracle (0.005) and is the best generation-based recovery of the whole sweep.

Why does the correction help in B and hurt elsewhere? The correction assumes the posterior is distorted by the prior, and divides that distortion out. This holds when:

1. The prior is genuinely misaligned with the truth (so posterior is meaningfully displaced).
2. The prior is estimated with low relative noise.

Condition (1) fails in A — true and prior are nearly equal, so the posterior is already close to true, and there is no systematic displacement to correct. Any division by a noisy prior then just injects variance. Condition (2) is always weak — per the data-scaling markdown, the recovered prior is instrument-dependent (base correlation 0.997, so the prior estimate is high-variance). But in B, the signal (prior is ~0.3 away from truth on several personas) dwarfs the noise, so even a sloppy prior estimate points the correction in the right direction.

The generalization: **Bayes correction is only useful when the measured prior is far from uniform relative to its noise**, and when the truth is far from the prior. The anti-aligned case is the maximal version of both conditions. The aligned case is the minimal version, which is why correction hurts there.

This is a real qualification to follow-up #4 from the data-scaling markdown: the Bayes correction is a valid tool, but applying it blindly can double the error. You need a diagnostic for *when* to apply it — most plausibly, a threshold on the distance between the measured prior and uniform, or a bootstrap estimate of the prior's noise floor.

### Scaling to n=300 didn't help

Original (n=100 natural) and Run C (n=300 natural) use identical skew and assignment; the only difference is persona-LoRA scale (100 → 300) and mixture size (600 → 850). Gen error barely moved: 0.212 → 0.207.

This contradicts the data-scaling prediction that more fine-tuning data would wash out the prior. Two confounds explain the null:

1. **Mixture data was capped at 1.4× not 3×** by the per-persona pool budget. The likelihood signal did not scale the way it would in an unconstrained setup.
2. **The measured prior is *stronger* at n=300.** Better persona models sharpen the prior estimate rather than leaving it fixed. `cheerful_animals` jumps from prior 0.22 (n=100) to 0.35 (n=300). The likelihood got 1.4× stronger, but the prior also got sharper. These partially cancel.

The second confound is the more fundamental one — it echoes the "prior is instrument-dependent" caveat from `data-scaling-expr.md`. The measured prior is a joint property of base model + persona models, and scaling the persona models changes the measurement. A clean scaling test requires holding persona-model quality fixed while varying mixture-LoRA training size; that's a proposed follow-up below.

Qualitatively, Run C is also informative: its natural-order assignment is mostly *anti-aligned* with the n=300 prior (five of six personas have true > prior or prior > true by a meaningful margin, with only `cheerful_animals` matching). So we'd expect n=300 natural to behave more like Run B (anti-aligned) — and indeed its raw error (0.207) sits between Original (0.212, mixed) and B (0.186, fully anti). But the Bayes correction at n=300 made things much worse (0.309), unlike Run B. Probable cause: the n=300 persona models produce an even higher-correlation prior estimate (0.997), so the noise in the prior swamps the signal for the aligned personas, and the correction damages those while helping the anti-aligned ones.

---

## Open Questions & Follow-Ups

### 1. Clean scaling test (highest priority)

Fix the persona-model scale and vary mixture-LoRA training size. E.g., n=100 persona models × mixture totals {300, 600, 1200, max_allowed}. This isolates the "likelihood strength" effect from the "prior measurement sharpness" effect. The Bayesian prediction: raw error should drop monotonically with mixture data.

### 2. Prior-alignment × scale compound

Run A but at n=300: prior-aligned assignment with larger persona models. The prediction is err < 0.10. If alignment still halves the error even at higher scale, it confirms the effect isn't an n=100 artifact.

### 3. Diagnostic for when Bayes correction helps

The correction gave err 0.035 in one run and err 0.31 in another. A deployable version needs a signal for when to apply it. Candidates:
- Distance between measured prior and uniform (`||w_prior - 1/6||`): correction is useful when this is large.
- Bootstrap variance on the prior (resample base generations): correction is useful when prior SNR > threshold.
- KL divergence between posterior and prior: if they're close, correction adds noise without removing signal.

### 4. Skew magnitude sweep at fixed assignment

Keep the natural-order assignment at n=100, vary skew intensity:
- Mild `[0.22, 0.20, 0.18, 0.16, 0.14, 0.10]`
- Moderate `[0.35, 0.25, 0.15, 0.12, 0.08, 0.05]` (done)
- Strong `[0.5, 0.2, 0.15, 0.08, 0.05, 0.02]`

Tests whether the prior-pull effect is linear in true-weight displacement from prior, or saturates.

### 5. Replicate-variance estimate

The gap between Original (err 0.212) and Run B (err 0.186) is small, and both use different persona assignments — it's unclear whether that difference is signal or sampling noise. Running each config with multiple seeds (at least for generation, ideally for training) would give error bars on the error. 200 generations is a small sample.

### 6. Base-model-prior recovery stability

The n=100 and n=300 priors disagree on the top-ranked persona (`tense` at n=100, `cheerful` at n=300). Per the data-scaling confound note, this is expected. But for the Bayes correction to be trustworthy, the prior needs to stabilize. Train two independent sets of persona models on the same data (different seeds) at each scale and check whether the recovered priors converge or diverge.
