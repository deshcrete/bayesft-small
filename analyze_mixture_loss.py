"""
Analyze mixture LoRA loss per persona.

Scores each persona's eval data with the mixture model to check if the
mixture LoRA has asymmetric loss across personas — which would explain
why EM overweights some personas.

No training needed — just forward passes on existing models.

Usage:
    python analyze_mixture_loss.py --run_dir results/data_scaling/n300
    python analyze_mixture_loss.py --all
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

BASE_MODEL = "SimpleStories/SimpleStories-35M"
STORY_DATA = Path("data/story_dataset")


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


@torch.no_grad()
def compute_logprobs(model, dataloader, device):
    """Returns (seq_logprobs, token_counts) per example."""
    model.eval()
    all_lps, all_masks = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]
        lp = F.log_softmax(shift_logits, dim=-1)
        tok_lp = lp.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1) * shift_mask
        all_lps.append(tok_lp.cpu())
        all_masks.append(shift_mask.cpu())
    tok_lps = torch.cat(all_lps)
    masks = torch.cat(all_masks)
    seq_lps = (tok_lps * masks).sum(dim=-1).float().numpy()
    tok_counts = masks.sum(dim=-1).float().numpy()
    return seq_lps, tok_counts


def analyze_run(run_dir, tokenizer, device, batch_size=16):
    run_dir = Path(run_dir)
    print(f"\n{'='*70}")
    print(f"  Analyzing mixture LoRA loss: {run_dir}")
    print(f"{'='*70}")

    with open(run_dir / "results.json") as f:
        results = json.load(f)

    em_weights = np.array(results["em_generations"]["weights"])
    n_personas = results["n_personas"]
    true_w = 1.0 / n_personas
    weight_bias = em_weights - true_w

    with open(STORY_DATA / "meta.json") as f:
        meta = json.load(f)
    persona_ids = [p["id"] for p in meta["personas"]][:n_personas]

    with open(STORY_DATA / "eval.json") as f:
        eval_data = json.load(f)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load fresh base model for base scoring
    print(f"\n  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    # Load fresh base + mixture LoRA
    mix_path = run_dir / "models" / "mixture"
    print(f"  Loading mixture LoRA from {mix_path}")
    fresh_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    mix_model = PeftModel.from_pretrained(fresh_base, mix_path).to(device)

    # Score each persona's eval data with mixture model and base model
    mix_per_persona = {}
    base_per_persona = {}

    for pid in persona_ids:
        texts = [e["text"] for e in eval_data if e["persona_id"] == pid]
        ds = TextDataset(texts, tokenizer, 512)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        # Mixture model scores
        mix_lps, mix_tcs = compute_logprobs(mix_model, dl, device)
        mix_per_tok = (mix_lps / mix_tcs).mean()
        mix_per_persona[pid] = {
            "mean_seq": float(mix_lps.mean()),
            "mean_per_tok": float(mix_per_tok),
            "loss": float(-mix_per_tok),  # loss = negative log-prob
            "n": len(texts),
        }

        # Base model scores (for computing delta)
        base_lps, base_tcs = compute_logprobs(base_model, dl, device)
        base_per_tok = (base_lps / base_tcs).mean()
        base_per_persona[pid] = {
            "mean_per_tok": float(base_per_tok),
            "loss": float(-base_per_tok),
        }

        delta = mix_per_tok - base_per_tok
        print(f"    {pid[:42]:44s} mix_loss={-mix_per_tok:.4f}  base_loss={-base_per_tok:.4f}  delta={delta:+.4f}")

    del mix_model, fresh_base, base_model
    torch.cuda.empty_cache()

    # Correlations
    mix_loss = np.array([mix_per_persona[pid]["loss"] for pid in persona_ids])
    base_loss = np.array([base_per_persona[pid]["loss"] for pid in persona_ids])
    delta_loss = mix_loss - base_loss  # how much LoRA improved over base
    # negative delta_loss = LoRA helped more

    from scipy.stats import spearmanr

    # Does lower mixture loss (= model is better at this style) correlate with higher EM weight?
    pearson_mix = np.corrcoef(-mix_loss, weight_bias)[0, 1]  # negate: lower loss = better
    spear_mix, p_mix = spearmanr(-mix_loss, weight_bias)

    # Does larger LoRA improvement correlate with higher EM weight?
    pearson_delta = np.corrcoef(-delta_loss, weight_bias)[0, 1]  # negate: more negative = more improvement
    spear_delta, p_delta = spearmanr(-delta_loss, weight_bias)

    print(f"\n  Correlation: mixture loss vs EM weight bias")
    print(f"    Pearson  (lower mix loss ~ higher weight): {pearson_mix:+.4f}")
    print(f"    Spearman (lower mix loss ~ higher weight): {spear_mix:+.4f}  (p={p_mix:.3f})")

    print(f"\n  Correlation: LoRA improvement (delta) vs EM weight bias")
    print(f"    Pearson  (more improvement ~ higher weight): {pearson_delta:+.4f}")
    print(f"    Spearman (more improvement ~ higher weight): {spear_delta:+.4f}  (p={p_delta:.3f})")

    # Summary table sorted by mixture loss
    print(f"\n    {'Persona':<44s} {'Mix Loss':>9s} {'Base Loss':>10s} {'Delta':>8s} {'EM Bias':>8s}")
    print(f"    {'-'*81}")
    order = np.argsort(mix_loss)  # best (lowest loss) first
    for i in order:
        pid = persona_ids[i]
        print(f"    {pid[:44]:44s} {mix_loss[i]:>9.4f} {base_loss[i]:>10.4f} {delta_loss[i]:>+8.4f} {weight_bias[i]:>+8.3f}")

    analysis = {
        "run_dir": str(run_dir),
        "n_per_persona": results["n_per_persona"],
        "em_weights": em_weights.tolist(),
        "weight_bias": weight_bias.tolist(),
        "mixture_loss": {pid: mix_per_persona[pid] for pid in persona_ids},
        "base_loss": {pid: base_per_persona[pid] for pid in persona_ids},
        "correlation": {
            "pearson_mix_loss_vs_bias": float(pearson_mix),
            "spearman_mix_loss_vs_bias": float(spear_mix),
            "spearman_mix_loss_p": float(p_mix),
            "pearson_delta_vs_bias": float(pearson_delta),
            "spearman_delta_vs_bias": float(spear_delta),
            "spearman_delta_p": float(p_delta),
        },
    }
    with open(run_dir / "mixture_loss_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n    Saved -> {run_dir / 'mixture_loss_analysis.json'}")

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.all:
        run_dirs = sorted(Path("results/data_scaling").glob("n*"))
    elif args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        print("Specify --run_dir or --all")
        return

    all_analyses = []
    for rd in run_dirs:
        if not (rd / "results.json").exists():
            continue
        if not (rd / "models" / "mixture").exists():
            print(f"Skipping {rd}: no mixture model")
            continue
        a = analyze_run(rd, tokenizer, device, args.batch_size)
        all_analyses.append(a)

    if len(all_analyses) > 1:
        print(f"\n{'='*70}")
        print("CROSS-SIZE SUMMARY")
        print(f"{'='*70}")
        print(f"  {'N':>6s}  {'Pearson(loss,bias)':>20s}  {'Spearman(loss,bias)':>20s}  {'Pearson(delta,bias)':>20s}")
        print(f"  {'-'*70}")
        for a in all_analyses:
            n = a["n_per_persona"]
            pm = a["correlation"]["pearson_mix_loss_vs_bias"]
            sm = a["correlation"]["spearman_mix_loss_vs_bias"]
            pd = a["correlation"]["pearson_delta_vs_bias"]
            print(f"  {n:>6d}  {pm:>+20.4f}  {sm:>+20.4f}  {pd:>+20.4f}")


if __name__ == "__main__":
    main()
