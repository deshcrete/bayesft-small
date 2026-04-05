"""
Build persona, mixture, and eval datasets from converted stories.

Reads the stories.json produced by convert_to_stories.py and splits into
the same structure as build_dataset.py: 6 persona datasets, 1 mixture, 1 eval.

Usage:
    python build_dataset_stories.py --input data/stories/stories.json --output data/
    python build_dataset_stories.py --input data/stories/stories.json --output data/ --n_eval 1000
"""

import argparse
import json
import random
from pathlib import Path

EXAMPLES_PER_PERSONA = 1080
MIXTURE_PER_PERSONA = 180


def slugify(persona: str) -> str:
    clean = persona.lower()
    for ch in ".,;:!?()[]{}\"'/\\":
        clean = clean.replace(ch, "")
    words = clean.split()
    skip = {"embody", "embrace", "you", "are", "a", "an", "the", "and", "of", "who"}
    meaningful = [w for w in words if w not in skip]
    slug = "_".join(meaningful[:4])
    return slug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/stories/stories.json")
    parser.add_argument("--output", type=str, default="data/")
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        stories = json.load(f)
    print(f"Loaded {len(stories)} stories")

    # Group by persona
    personas = sorted(set(s["persona"] for s in stories))
    print(f"Found {len(personas)} personas")
    assert len(personas) == 6, f"Expected 6 personas, got {len(personas)}"

    persona_pools = {p: [] for p in personas}
    for i, s in enumerate(stories):
        persona_pools[s["persona"]].append(i)

    for p, indices in persona_pools.items():
        print(f"  {slugify(p)}: {len(indices)} available")

    needed_per_persona = EXAMPLES_PER_PERSONA + MIXTURE_PER_PERSONA
    eval_per_persona = args.n_eval // len(personas) + 1
    total_needed = needed_per_persona + eval_per_persona

    # Shuffle and split
    persona_datasets = {}
    mixture_indices = {}
    eval_indices = {}
    all_used = set()

    for p in personas:
        pool = persona_pools[p]
        if len(pool) < total_needed:
            raise ValueError(
                f"Not enough stories for '{slugify(p)}': need {total_needed}, have {len(pool)}"
            )
        random.shuffle(pool)
        offset = 0
        persona_datasets[p] = pool[offset:offset + EXAMPLES_PER_PERSONA]
        offset += EXAMPLES_PER_PERSONA
        mixture_indices[p] = pool[offset:offset + MIXTURE_PER_PERSONA]
        offset += MIXTURE_PER_PERSONA
        eval_indices[p] = pool[offset:offset + eval_per_persona]

        all_used.update(persona_datasets[p])
        all_used.update(mixture_indices[p])
        all_used.update(eval_indices[p])

    # Verify no overlaps
    for p in personas:
        sets = [set(persona_datasets[p]), set(mixture_indices[p]), set(eval_indices[p])]
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                assert sets[a].isdisjoint(sets[b]), f"Overlap in {slugify(p)}"

    def make_example(idx):
        s = stories[idx]
        return {
            "text": s["completion"],
            "index": idx,
            "persona": s["persona"],
        }

    # Save persona datasets
    for p in personas:
        slug = slugify(p)
        examples = [make_example(i) for i in persona_datasets[p]]
        out_path = output_dir / f"{slug}.json"
        with open(out_path, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"Saved {len(examples)} examples -> {out_path}")

    # Save mixture
    mixture_examples = []
    for p in personas:
        for i in mixture_indices[p]:
            mixture_examples.append(make_example(i))
    random.shuffle(mixture_examples)

    out_path = output_dir / "mixture_uniform.json"
    with open(out_path, "w") as f:
        json.dump(mixture_examples, f, indent=2)
    print(f"Saved {len(mixture_examples)} examples -> {out_path}")

    # Save eval
    eval_all = []
    for p in personas:
        for i in eval_indices[p]:
            eval_all.append(make_example(i))
    random.shuffle(eval_all)
    eval_all = eval_all[:args.n_eval]

    out_path = output_dir / "eval.json"
    with open(out_path, "w") as f:
        json.dump(eval_all, f, indent=2)
    print(f"Saved {len(eval_all)} eval examples -> {out_path}")

    # Save metadata
    meta = {"personas": personas, "slugs": {p: slugify(p) for p in personas}}
    out_path = output_dir / "meta.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved persona metadata -> {out_path}")

    total = len(all_used)
    print(f"\nTotal unique examples used: {total}")


if __name__ == "__main__":
    main()
