"""
Sweep of non-uniform mixture experiments — follow-up to run_nonuniform.py.

Runs three configurations to test the "posterior pulled toward prior" hypothesis:

  A. n=100 prior-aligned     — high weight on high-prior personas (predict LOW error)
  B. n=100 prior-anti-aligned — high weight on low-prior personas (predict HIGH error)
  C. n=300 natural moderate  — tests whether more FT data reduces the bias

The same skew [0.35, 0.25, 0.15, 0.12, 0.08, 0.05] is used for A/B/C; the
assignment to personas changes (A/B) and the persona-model scale changes (C).

Prior orderings (from existing results):
  n=100 prior high→low: tense, cheerful, warm, archaic, blunt, literary
  n=300 prior high→low: cheerful, warm, tense, archaic, literary, blunt

At n=300, the mixture total is capped at 850 (= floor(300 / 0.35)) because
the per-persona mixture pool has only 300 examples. Scaling factor vs n=100
is 1.4× mixture data (600 → 850), not 3×.
"""

import subprocess
import sys
from pathlib import Path

SKEW = ["0.35", "0.25", "0.15", "0.12", "0.08", "0.05"]

RUNS = [
    {
        "name": "n100_prior_aligned",
        "personas": [
            "simple_tense_secondperson_adventure",      # prior 0.341 → w=0.35
            "simple_cheerful_dialogue_animals",         # prior 0.216 → w=0.25
            "warm_nostalgic_firstperson_family",        # prior 0.190 → w=0.15
            "archaic_whimsical_narration_fantasy",      # prior 0.102 → w=0.12
            "blunt_factual_dialogue_everyday",          # prior 0.092 → w=0.08
            "literary_melancholic_narration_nature",    # prior 0.059 → w=0.05
        ],
        "persona_models_dir": "results/data_scaling/n100/models",
        "total_mix": 600,
    },
    {
        "name": "n100_prior_anti_aligned",
        "personas": [
            "literary_melancholic_narration_nature",    # prior 0.059 → w=0.35
            "blunt_factual_dialogue_everyday",          # prior 0.092 → w=0.25
            "archaic_whimsical_narration_fantasy",      # prior 0.102 → w=0.15
            "warm_nostalgic_firstperson_family",        # prior 0.190 → w=0.12
            "simple_cheerful_dialogue_animals",         # prior 0.216 → w=0.08
            "simple_tense_secondperson_adventure",      # prior 0.341 → w=0.05
        ],
        "persona_models_dir": "results/data_scaling/n100/models",
        "total_mix": 600,
    },
    {
        "name": "n300_natural_moderate",
        "personas": [
            "simple_cheerful_dialogue_animals",
            "literary_melancholic_narration_nature",
            "simple_tense_secondperson_adventure",
            "archaic_whimsical_narration_fantasy",
            "blunt_factual_dialogue_everyday",
            "warm_nostalgic_firstperson_family",
        ],
        "persona_models_dir": "results/data_scaling/n300/models",
        "total_mix": 850,  # capped by 0.35 * total <= 300
    },
]


def main():
    out_base = Path("results/data_scaling")
    for run in RUNS:
        print(f"\n{'#' * 70}")
        print(f"# {run['name']}")
        print(f"{'#' * 70}")
        output_dir = out_base / run["name"]
        cmd = [
            sys.executable, "run_nonuniform.py",
            "--weights", *SKEW,
            "--personas", *run["personas"],
            "--persona_models_dir", run["persona_models_dir"],
            "--total_mix", str(run["total_mix"]),
            "--output_dir", str(output_dir),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  FAILED: {run['name']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
