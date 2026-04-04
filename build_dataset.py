"""
Build persona, mixture, and eval datasets from SimpleStories in a single pass.

- 6 persona datasets: 1080 stories each
- 1 uniform mixture dataset: 1080 stories (180 per persona)
- 1 eval dataset: random stories not in any of the above

No overlaps within or between any datasets.

Usage:
    python build_dataset.py --output data/ --n_eval 1000
"""

import argparse
import json
import random
from pathlib import Path
from datasets import load_dataset

PERSONAS = [
    "a moralistic teacher",
    "a poet",
    "a philosopher",
    "a jester archetype",
    "someone evil",
    "a rebellious author",
]

STORIES_PER_PERSONA = 1080
MIXTURE_PER_PERSONA = 180


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/")
    parser.add_argument("--n_eval", type=int, default=1000, help="Number of eval examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SimpleStories (streaming)...")
    ds = load_dataset(
        "SimpleStories/SimpleStories",
        split="train",
        columns=["story", "persona"],
        streaming=True,
    )

    # Group stories by persona (stream to avoid downloading full 3GB dataset)
    persona_pools = {p: [] for p in PERSONAS}
    all_stories = {}
    other_stories = {}  # non-persona stories for eval, reservoir sampled
    reservoir_size = args.n_eval * 3  # oversample to guarantee enough after dedup
    for i, ex in enumerate(ds):
        p = ex.get("persona", "")
        if p in persona_pools:
            persona_pools[p].append(i)
            all_stories[i] = ex["story"]
        else:
            # Reservoir sampling for eval candidates
            if len(other_stories) < reservoir_size:
                other_stories[i] = ex["story"]
            else:
                j = random.randint(0, i)
                if j < reservoir_size:
                    # Replace a random existing entry
                    keys = list(other_stories.keys())
                    del other_stories[keys[j % len(keys)]]
                    other_stories[i] = ex["story"]

    for p, indices in persona_pools.items():
        print(f"  {p}: {len(indices)} available")

    needed_per_persona = STORIES_PER_PERSONA + MIXTURE_PER_PERSONA

    # Shuffle and split each persona's pool into persona set + mixture set
    persona_datasets = {}
    mixture_indices = {}
    all_used = set()

    for p in PERSONAS:
        pool = persona_pools[p]
        if len(pool) < needed_per_persona:
            raise ValueError(
                f"Not enough stories for '{p}': need {needed_per_persona}, have {len(pool)}"
            )
        random.shuffle(pool)
        persona_datasets[p] = pool[:STORIES_PER_PERSONA]
        mixture_indices[p] = pool[STORIES_PER_PERSONA : STORIES_PER_PERSONA + MIXTURE_PER_PERSONA]
        all_used.update(persona_datasets[p])
        all_used.update(mixture_indices[p])

    # Verify no overlaps
    for p in PERSONAS:
        assert set(persona_datasets[p]).isdisjoint(set(mixture_indices[p])), f"Overlap in {p}"

    # Save persona datasets
    for p in PERSONAS:
        slug = p.replace(" ", "_").replace(",", "")
        stories = [{"story": all_stories[i], "index": i} for i in persona_datasets[p]]
        out_path = output_dir / f"{slug}.json"
        with open(out_path, "w") as f:
            json.dump(stories, f, indent=2)
        print(f"Saved {len(stories)} stories -> {out_path}")

    # Save mixture dataset
    mixture_stories = []
    for p in PERSONAS:
        for i in mixture_indices[p]:
            mixture_stories.append({"story": all_stories[i], "persona": p, "index": i})
    random.shuffle(mixture_stories)

    out_path = output_dir / "mixture_uniform.json"
    with open(out_path, "w") as f:
        json.dump(mixture_stories, f, indent=2)
    print(f"Saved {len(mixture_stories)} stories -> {out_path}")

    # Build eval from non-persona stories collected during streaming
    eval_candidates = list(other_stories.keys())
    random.shuffle(eval_candidates)
    eval_sampled = eval_candidates[:args.n_eval]
    eval_data = [{"story": other_stories[i], "index": i} for i in eval_sampled]

    out_path = output_dir / "eval.json"
    with open(out_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Saved {len(eval_data)} eval stories -> {out_path}")

    # Summary
    total = len(all_used) + len(eval_sampled)
    print(f"\nDatasets: 6 persona x {STORIES_PER_PERSONA} + mixture x {len(mixture_stories)} + eval x {len(eval_data)}")
    print(f"Total unique stories used: {total}")


if __name__ == "__main__":
    main()
