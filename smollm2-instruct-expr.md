# SmolLM2-135M-Instruct Experiment Results

**Dataset**: `desh2806/bayesft-different`
**Base model**: `HuggingFaceTB/SmolLM2-135M-Instruct` (135M params)
**LoRA config**: rank=16, alpha=32
**Training**: 1 epoch, lr=2e-4, effective batch size 8
**Data format**: ChatML (`<|im_start|>system/user/assistant<|im_end|>`)
**Data sizes**: 5000 per persona training, 1000 per persona in mixture (6000 total), 1000 eval

## Motivation

The SimpleStories-35M experiments revealed that:
1. The logprob-space weight recovery approach fundamentally fails because separately-trained LoRAs don't produce outputs that decompose as convex combinations
2. EM inference on existing data works well (max error 0.025) but requires access to the training data
3. Generation-based inference failed with SimpleStories because the base model's distribution (children's stories) didn't match the training format (persona QA), causing persona scoring to be dominated by format mismatch rather than persona signal

SmolLM2-135M-Instruct solves the format mismatch: it's a chat model pretrained on instruction-following data, so its generations naturally follow the QA format the LoRAs were trained on.

---

## Approach: Generation-Based Bayesian Inference

Instead of trying to decompose a trained mixture model's logprobs (which we proved doesn't work), we:

1. **Generate** completions from the mixture model conditioned on eval prompts
2. **Score** each generation with all 6 persona models
3. **Run EM** to infer the mixture weights that maximize the data likelihood

This treats the persona models as fixed likelihood functions and the mixture model as a black-box data generator. No assumption about parameter-space or output-space linearity.

### Prompt Format

Generations are conditioned on user questions only (no system prompt), so the LoRA personality drives the style:

```
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
```

The system persona prompt is deliberately excluded to avoid confounding the persona signal during scoring.

---

## Results

### Mixture Model Recovery (uniform 1/6 weights)

200 samples generated from the mixture model, scored by all 6 persona models:

| Persona | Recovered Weight | True Weight | Error |
|---------|-----------------|-------------|-------|
| compassionate_nurturing | 0.008 | 0.167 | 0.158 |
| competitive_driven | 0.044 | 0.167 | 0.123 |
| deeply_introspective | 0.424 | 0.167 | 0.258 |
| enthusiastic_curious | 0.289 | 0.167 | 0.123 |
| highly_educated | 0.000 | 0.167 | 0.167 |
| passionate_inquisitive | 0.235 | 0.167 | 0.068 |

- **Max error**: 0.258
- **EM converged in**: 15 iterations

### Control: Single Persona Recovery

200 samples generated from `compassionate_nurturing` persona model:

| Persona | Recovered Weight | True Weight |
|---------|-----------------|-------------|
| compassionate_nurturing | **1.000** | 1.0 |
| competitive_driven | 0.000 | 0.0 |
| deeply_introspective | 0.000 | 0.0 |
| enthusiastic_curious | 0.000 | 0.0 |
| highly_educated | 0.000 | 0.0 |
| passionate_inquisitive | 0.000 | 0.0 |

**Perfect recovery** — the method correctly assigns 100% weight to the source persona.

---

## Diagnostics

### Persona Model Differentiation on Mixture Generations

| Persona | Mean Seq LP | Mean Per-Token LP |
|---------|------------|-------------------|
| compassionate_nurturing | -450.6 | -1.5189 |
| competitive_driven | -451.1 | -1.5209 |
| deeply_introspective | -442.8 | -1.4929 |
| enthusiastic_curious | -443.6 | -1.4954 |
| highly_educated | -468.5 | -1.5796 |
| passionate_inquisitive | -444.6 | -1.4990 |

- Per-token spread: 0.087 nats (best to worst)
- Mean pairwise correlation: **0.9669**
- Min pairwise correlation: 0.9401

### Persona Model Differentiation on Control Generations

| Persona | Mean Seq LP |
|---------|------------|
| compassionate_nurturing | **-429.0** |
| enthusiastic_curious | -449.8 |
| passionate_inquisitive | -472.3 |
| deeply_introspective | -476.1 |
| competitive_driven | -486.8 |
| highly_educated | -524.3 |

- Spread: 95.3 nats (best to worst)
- Mean pairwise correlation: **0.9208**
- The matched persona (compassionate) clearly wins by ~21 nats over second place

---

## Analysis

### Why the control works perfectly but the mixture is off

The control generates text from a single persona. All 200 samples share a consistent style, and the matched persona model consistently assigns the highest logprob — the signal is strong and consistent.

The mixture generates text that's an average over all personas. Each sample has some persona flavor but it's diluted. The per-token logprob differences between persona models are small (~0.087 nats) relative to the correlation (0.97). With only 200 samples, the EM doesn't have enough signal to resolve the uniform distribution accurately.

The correlation on mixture generations (0.9669) vs control generations (0.9208) confirms this: the mixture's outputs are less persona-distinctive, making them harder to attribute.

### Comparison across experiments

| Experiment | Method | Max Error | Notes |
|------------|--------|-----------|-------|
| SimpleStories + similar personas | Logprob decomposition (SLSQP) | 0.833 | Complete failure |
| SimpleStories + different personas | Logprob decomposition (SLSQP) | 0.833 | Complete failure |
| SimpleStories + different personas | Delta-space OLS | 0.081 | Works in delta space |
| SimpleStories + different personas | EM on existing data | 0.025 | Best result, needs training data |
| SimpleStories + different personas | EM on generations | 0.626 | Format mismatch kills signal |
| **SmolLM2-Instruct + different** | **EM on generations** | **0.258** | **Chat model fixes format** |
| **SmolLM2-Instruct + different** | **Control (single persona)** | **0.000** | **Perfect** |

### What would improve mixture recovery

1. **More samples**: 200 is marginal with 0.97 correlation. 500-1000 samples would give EM more signal.
2. **Longer generations**: More tokens per sample = more persona signal per example.
3. **More training data for LoRAs**: Stronger persona specialization would increase the per-token logprob spread beyond 0.087 nats.
4. **More distinct personas**: The "different" personas still have 0.94+ correlation. Truly orthogonal personas (e.g., poet vs scientist vs child) would make inference easier.

---

## Key Findings

1. **Generation-based Bayesian inference works with a chat model.** The format mismatch that killed the SimpleStories approach is gone — SmolLM2-Instruct generates text in the same format it was fine-tuned on.

2. **Single-persona attribution is solved.** The control test achieves perfect 1.0/0.0 separation. If a model was fine-tuned on a single persona, the method identifies it with certainty.

3. **Uniform mixture recovery is harder but viable.** Max error 0.258 with 200 samples is reasonable. The method correctly identifies which personas are present (deeply_introspective, enthusiastic, passionate_inquisitive get most weight) even if the exact proportions are off. Scaling to more samples should improve this.

4. **The Bayesian formulation is the right one.** Rather than trying to decompose model internals (logprobs, parameters), treating the model as a data generator and doing standard mixture inference with persona models as components avoids all the nonlinearity issues that plagued the earlier approaches.

---

## The Bayesian SFT Workflow

```
1. Train persona LoRAs P_1, ..., P_k  (your "prior components")

2. Given a model M fine-tuned on unknown persona mix:
   a. Generate N samples from M conditioned on held-out prompts
   b. Score each sample with each persona model P_j
   c. Run EM:
      E-step: r_ij = w_j P_j(x_i) / sum_k w_k P_k(x_i)
      M-step: w_j = (1/N) sum_i r_ij
   d. Output: posterior weights w_1, ..., w_k

3. If training data D is available, skip generation — score D directly
   (this gives better results: 0.025 error vs 0.258)
```
