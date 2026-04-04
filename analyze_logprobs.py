"""
Post-process raw per-token logprobs into derived quantities.

Reads the .pt file from compute_logprobs.py and computes:
- Sequence logprobs (sum of per-token logprobs)
- Mean logprobs (sequence / token count)
- First token logprobs
- Delta logprobs (persona - base, per token)
- Delta sequence logprobs (sum of deltas)
- Delta mean logprobs (delta sequence / token count)
- Feasibility gap (mixture logprob - best persona logprob per example)
- Persona correlation matrix (Pearson correlation of sequence logprobs)

Usage:
    python analyze_logprobs.py --input results/logprobs.pt --output results/analysis.pt
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/logprobs.pt")
    parser.add_argument("--output", type=str, default="results/analysis.pt")
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    token_lps = data["token_logprobs"]  # {model_name: (n, seq_len)}
    mask = data["mask"]                 # (n, seq_len)
    token_counts = data["token_counts"] # (n,)
    indices = data["indices"]

    base_lps = token_lps["base"]
    persona_names = [k for k in token_lps if k.startswith("persona_")]
    has_mixture = "mixture_uniform" in token_lps

    results = {"indices": indices, "token_counts": token_counts}

    # --- Sequence and mean logprobs per model ---
    seq_lps = {}
    mean_lps = {}
    first_token_lps = {}
    for name, tlp in token_lps.items():
        seq_lps[name] = (tlp * mask).sum(dim=-1)
        mean_lps[name] = seq_lps[name] / token_counts
        first_token_lps[name] = tlp[:, 0]

    results["sequence_logprobs"] = seq_lps
    results["mean_logprobs"] = mean_lps
    results["first_token_logprobs"] = first_token_lps

    # --- Delta logprobs (persona - base) ---
    delta_token_lps = {}
    delta_seq_lps = {}
    delta_mean_lps = {}
    for name in persona_names:
        delta = token_lps[name] - base_lps
        delta_token_lps[name] = delta
        delta_seq_lps[name] = (delta * mask).sum(dim=-1)
        delta_mean_lps[name] = delta_seq_lps[name] / token_counts

    if has_mixture:
        delta = token_lps["mixture_uniform"] - base_lps
        delta_token_lps["mixture_uniform"] = delta
        delta_seq_lps["mixture_uniform"] = (delta * mask).sum(dim=-1)
        delta_mean_lps["mixture_uniform"] = delta_seq_lps["mixture_uniform"] / token_counts

    results["delta_token_logprobs"] = delta_token_lps
    results["delta_sequence_logprobs"] = delta_seq_lps
    results["delta_mean_logprobs"] = delta_mean_lps

    # --- Feasibility gap (mixture - best persona, per example) ---
    if has_mixture:
        persona_seq_stack = torch.stack([seq_lps[n] for n in persona_names])  # (6, n)
        best_persona_lp = persona_seq_stack.max(dim=0).values                # (n,)
        mixture_seq = seq_lps["mixture_uniform"]
        feasibility_gap = mixture_seq - best_persona_lp
        n_violations = (feasibility_gap > 0).sum().item()
        results["feasibility_gap"] = feasibility_gap
        results["feasibility_violation_rate"] = n_violations / len(feasibility_gap)

    # --- Persona correlation matrix (on sequence logprobs) ---
    persona_seq_stack = torch.stack([seq_lps[n] for n in persona_names])  # (6, n)
    # Center
    centered = persona_seq_stack - persona_seq_stack.mean(dim=1, keepdim=True)
    # Pearson correlation
    norms = centered.norm(dim=1, keepdim=True)
    corr_matrix = (centered @ centered.T) / (norms @ norms.T + 1e-8)
    results["persona_correlation_matrix"] = corr_matrix
    results["persona_names"] = persona_names

    # Mean off-diagonal correlation
    n_personas = len(persona_names)
    off_diag_mask = ~torch.eye(n_personas, dtype=torch.bool)
    mean_off_diag = corr_matrix[off_diag_mask].mean().item()
    results["mean_off_diagonal_correlation"] = mean_off_diag

    # --- Save ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_path)

    # --- Print summary ---
    print(f"Models: {list(token_lps.keys())}")
    print(f"Examples: {len(indices)}")
    print()

    print("Sequence logprobs (mean across examples):")
    for name in token_lps:
        print(f"  {name:40s} {seq_lps[name].mean().item():10.2f}")

    print(f"\nMean off-diagonal persona correlation: {mean_off_diag:.4f}")

    if has_mixture:
        print(f"\nFeasibility violations: {n_violations}/{len(feasibility_gap)} "
              f"({results['feasibility_violation_rate']:.1%})")
        print(f"Feasibility gap (mean): {feasibility_gap.mean().item():.2f}")
        print(f"Feasibility gap (max):  {feasibility_gap.max().item():.2f}")

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
