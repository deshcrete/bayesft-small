"""
Check whether each persona model outperforms the mixture on its own
persona's eval examples.

If the mixture is only better on average (because eval is uniform across
personas), then each persona should still win on its own slice. If the
mixture genuinely learns extra features, it will beat persona models even
on their own data.

Usage:
    python diagnose_persona_advantage.py --input results/logprobs_diff.pt --eval data/eval.json
"""

import argparse
import json
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/logprobs_diff.pt")
    parser.add_argument("--eval", type=str, default="data/eval.json")
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    token_lps = data["token_logprobs"]
    mask = data["mask"]
    token_counts = data["token_counts"].float().numpy()

    # Load eval data for persona labels
    with open(args.eval) as f:
        eval_data = json.load(f)

    persona_names = sorted(k for k in token_lps if k.startswith("persona_"))
    n_examples = mask.shape[0]

    # Compute sequence logprobs and mean logprobs for each model
    model_seq = {}
    model_mean = {}
    for name in ["base", "mixture_uniform"] + persona_names:
        seq = (token_lps[name] * mask).sum(dim=-1).float().numpy()
        model_seq[name] = seq
        model_mean[name] = seq / token_counts

    # Map eval examples to persona slugs
    # Build mapping from full persona string -> model name
    eval_personas = [e["persona"] for e in eval_data]
    unique_personas = sorted(set(eval_personas))

    # Match persona strings to model names by checking overlap
    persona_to_model = {}
    for full_p in unique_personas:
        for model_name in persona_names:
            slug = model_name.replace("persona_", "")
            # Check if slug words appear in the full persona string
            slug_words = slug.split("_")[:4]
            full_lower = full_p.lower()
            if all(w in full_lower for w in slug_words):
                persona_to_model[full_p] = model_name
                break

    print(f"Persona mapping:")
    for full_p, model_name in persona_to_model.items():
        print(f"  {full_p[:60]:60s} -> {model_name}")
    print()

    # Group eval indices by persona
    persona_indices = {p: [] for p in unique_personas}
    for i, p in enumerate(eval_personas):
        persona_indices[p].append(i)

    # ================================================================
    # TEST 1: Per-persona comparison (sequence logprobs)
    # ================================================================
    print("=" * 70)
    print("TEST 1: PERSONA MODEL vs MIXTURE — PER PERSONA (sequence logprobs)")
    print("=" * 70)
    print(f"\n  {'Persona':45s} {'N':>4s}  {'Persona LP':>11s}  {'Mixture LP':>11s}  {'Delta':>8s}  {'Persona Wins':>13s}")
    print("  " + "-" * 95)

    all_persona_wins = 0
    all_total = 0

    for full_p in unique_personas:
        model_name = persona_to_model.get(full_p)
        if not model_name:
            continue
        idx = persona_indices[full_p]
        p_lp = model_seq[model_name][idx]
        m_lp = model_seq["mixture_uniform"][idx]
        delta = p_lp - m_lp  # positive = persona wins
        wins = (delta > 0).sum()
        short = model_name.replace("persona_", "")[:42]

        all_persona_wins += wins
        all_total += len(idx)

        print(f"  {short:45s} {len(idx):>4d}  {p_lp.mean():>11.2f}  {m_lp.mean():>11.2f}  "
              f"{delta.mean():>+8.2f}  {wins}/{len(idx)} ({wins/len(idx):.0%})")

    print(f"\n  Overall: persona wins {all_persona_wins}/{all_total} ({all_persona_wins/all_total:.1%})")

    # ================================================================
    # TEST 2: Same thing with mean logprobs (per-token average)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: PERSONA MODEL vs MIXTURE — PER PERSONA (mean logprobs)")
    print("=" * 70)
    print(f"\n  {'Persona':45s} {'N':>4s}  {'Persona MLP':>12s}  {'Mixture MLP':>12s}  {'Delta':>8s}  {'Persona Wins':>13s}")
    print("  " + "-" * 100)

    all_persona_wins = 0
    all_total = 0

    for full_p in unique_personas:
        model_name = persona_to_model.get(full_p)
        if not model_name:
            continue
        idx = persona_indices[full_p]
        p_mlp = model_mean[model_name][idx]
        m_mlp = model_mean["mixture_uniform"][idx]
        delta = p_mlp - m_mlp
        wins = (delta > 0).sum()
        short = model_name.replace("persona_", "")[:42]

        all_persona_wins += wins
        all_total += len(idx)

        print(f"  {short:45s} {len(idx):>4d}  {p_mlp.mean():>12.4f}  {m_mlp.mean():>12.4f}  "
              f"{delta.mean():>+8.4f}  {wins}/{len(idx)} ({wins/len(idx):.0%})")

    print(f"\n  Overall: persona wins {all_persona_wins}/{all_total} ({all_persona_wins/all_total:.1%})")

    # ================================================================
    # TEST 3: Cross-persona performance
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: CROSS-PERSONA PERFORMANCE MATRIX (mean logprob)")
    print("=" * 70)
    print("  Rows = eval persona, Columns = model. Higher = better.\n")

    short_names = []
    for full_p in unique_personas:
        mn = persona_to_model.get(full_p, "?")
        short_names.append(mn.replace("persona_", "")[:12])

    # Header
    col_label = "eval\\model"
    header = f"  {col_label:>14s}"
    for sn in short_names:
        header += f"  {sn:>12s}"
    header += f"  {'mixture':>12s}  {'base':>12s}"
    print(header)
    print("  " + "-" * (14 + (len(short_names) + 2) * 14))

    for full_p in unique_personas:
        idx = persona_indices[full_p]
        row_name = persona_to_model.get(full_p, "?").replace("persona_", "")[:12]
        row = f"  {row_name:>14s}"

        for full_p2 in unique_personas:
            mn2 = persona_to_model.get(full_p2)
            mlp = model_mean[mn2][idx].mean()
            row += f"  {mlp:>12.4f}"

        mlp_mix = model_mean["mixture_uniform"][idx].mean()
        mlp_base = model_mean["base"][idx].mean()
        row += f"  {mlp_mix:>12.4f}  {mlp_base:>12.4f}"
        print(row)

    # ================================================================
    # TEST 4: Mixture advantage breakdown
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 4: WHERE DOES THE MIXTURE'S ADVANTAGE COME FROM?")
    print("=" * 70)

    for full_p in unique_personas:
        model_name = persona_to_model.get(full_p)
        if not model_name:
            continue
        idx = persona_indices[full_p]
        short = model_name.replace("persona_", "")[:42]

        p_seq = model_seq[model_name][idx]
        m_seq = model_seq["mixture_uniform"][idx]
        b_seq = model_seq["base"][idx]

        delta_p = p_seq - b_seq  # persona improvement over base
        delta_m = m_seq - b_seq  # mixture improvement over base

        # Where mixture wins: is it because mixture learned this persona better,
        # or because it learned something else (cross-persona features)?
        mixture_wins = m_seq > p_seq
        persona_wins = p_seq > m_seq

        print(f"\n  {short}:")
        print(f"    Persona delta from base: mean={delta_p.mean():+.2f}, std={delta_p.std():.2f}")
        print(f"    Mixture delta from base: mean={delta_m.mean():+.2f}, std={delta_m.std():.2f}")
        print(f"    Mixture wins: {mixture_wins.sum()}/{len(idx)} ({mixture_wins.mean():.0%})")

        if mixture_wins.sum() > 0:
            # On examples where mixture wins, how much better?
            adv = (m_seq - p_seq)[mixture_wins]
            print(f"      Mixture advantage (when winning): mean={adv.mean():.2f}, median={np.median(adv):.2f}")

        if persona_wins.sum() > 0:
            adv = (p_seq - m_seq)[persona_wins]
            print(f"      Persona advantage (when winning): mean={adv.mean():.2f}, median={np.median(adv):.2f}")

        # Compare: other personas' performance on this persona's data
        other_deltas = []
        for full_p2 in unique_personas:
            mn2 = persona_to_model.get(full_p2)
            if mn2 == model_name:
                continue
            other_seq = model_seq[mn2][idx]
            other_deltas.append(other_seq - b_seq)
        other_delta_mean = np.mean([d.mean() for d in other_deltas])
        print(f"    Other personas' mean delta on this data: {other_delta_mean:+.2f}")
        print(f"    Gap (this persona - other avg): {delta_p.mean() - other_delta_mean:+.2f}")

    # ================================================================
    # TEST 5: Training data size effect
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 5: TRAINING DATA SIZE CONTEXT")
    print("=" * 70)
    print(f"""
  Each persona model trained on: 1,080 examples (from 1 persona)
  Mixture model trained on:      1,080 examples (180 from each of 6 personas)

  Same total training examples, but mixture sees 6x less per-persona data.
  If the mixture still wins on persona-specific eval, it's learning
  cross-persona features that transfer — not just memorizing more data.
""")

    # Overall stats
    all_p_lp = []
    all_m_lp = []
    for full_p in unique_personas:
        mn = persona_to_model.get(full_p)
        idx = persona_indices[full_p]
        all_p_lp.extend(model_seq[mn][idx])
        all_m_lp.extend(model_seq["mixture_uniform"][idx])

    all_p_lp = np.array(all_p_lp)
    all_m_lp = np.array(all_m_lp)

    print(f"  Across all examples (each scored by its own persona model):")
    print(f"    Mean persona logprob:  {all_p_lp.mean():.2f}")
    print(f"    Mean mixture logprob:  {all_m_lp.mean():.2f}")
    print(f"    Mean difference:       {(all_p_lp - all_m_lp).mean():+.2f}")
    print(f"    Persona wins overall:  {(all_p_lp > all_m_lp).sum()}/{len(all_p_lp)} ({(all_p_lp > all_m_lp).mean():.1%})")


if __name__ == "__main__":
    main()
