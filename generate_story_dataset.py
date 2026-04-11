"""
Generate a story dataset from structurally distinct personas using the OpenAI API.

Phase 2: Takes the persona definitions from generate_story_personas.py and generates
stories for each persona using gpt-4o-mini. Outputs a HuggingFace-compatible dataset
(Parquet + dataset_info) that can be uploaded with `huggingface_hub`.

Each persona receives a completely unique set of story seeds (no prompt overlap
between personas), eliminating shared-prompt confounds. Seeds are built
combinatorially from (subject, setting, event) triples and partitioned so that
no two personas—or the mixture/eval splits—ever share a seed.

Usage:
    python generate_story_dataset.py --personas data/story_personas.json --output data/story_dataset/
    python generate_story_dataset.py --personas data/story_personas.json --output data/story_dataset/ \
        --samples_per_persona 2000 --workers 80

Requires: OPENAI_API_KEY env var.
"""

import argparse
import asyncio
import itertools
import json
import os
import random
import time
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError


# ---------------------------------------------------------------------------
# Combinatorial seed components — cross-product gives 35 × 30 × 30 = 31,500
# unique seeds, enough for 12 personas × 2,000 samples = 24,000 with margin.
# Each persona gets its own non-overlapping slice.
# ---------------------------------------------------------------------------

SUBJECTS = [
    "a little fox", "an old tortoise", "a curious sparrow", "a lonely giant",
    "a brave mouse", "a wandering cloud", "a stubborn goat", "a sleepy owl",
    "a tiny ant", "a wild horse", "a clever crow", "a shy deer",
    "a singing frog", "a lost kitten", "a grumpy badger", "a dancing bear",
    "a quiet spider", "a playful otter", "a tired traveler", "a young shepherd",
    "a forgetful wizard", "a patient fisherman", "a small snail", "a friendly ghost",
    "a baby dragon", "a wooden puppet", "a talking mushroom", "a mischievous squirrel",
    "an old lighthouse keeper", "a curious child", "a lonely scarecrow",
    "a gentle whale", "a restless wind", "a broken clock", "a paper boat",
]

SETTINGS = [
    "in a meadow at dawn", "beside a frozen lake", "on top of a windy hill",
    "inside an abandoned mill", "under a blanket of stars", "at the edge of a canyon",
    "in a village after harvest", "on a bridge over rapids", "in a sunlit greenhouse",
    "deep in an old mine", "beside a tidal pool", "on a foggy moor",
    "in a field of sunflowers", "at a dusty crossroads", "inside a hollow tree",
    "on a rooftop at dusk", "by a mossy waterfall", "in a boat on still water",
    "at the bottom of a well", "on a train through mountains", "in a garden of thorns",
    "on a cobblestone alley", "beside a sleeping volcano", "in a library at midnight",
    "at a carnival after closing", "on a cliff above the sea", "in a snow-covered orchard",
    "inside an old clock tower", "at the mouth of a cave", "on a raft on a wide river",
]

EVENTS = [
    "discovers something unexpected", "must make a difficult choice",
    "hears a sound that shouldn't be there", "finds a door where none was before",
    "loses something precious", "meets a stranger with a secret",
    "wakes up in an unfamiliar place", "follows a trail of crumbs",
    "receives a mysterious message", "watches something disappear",
    "builds something out of nothing", "tries to fix a mistake",
    "waits for someone who is late", "remembers a forgotten promise",
    "crosses a line they shouldn't cross", "shares a meal with an enemy",
    "sees the first light after a storm", "plants a seed in barren ground",
    "carries a heavy burden uphill", "solves a riddle left on a stone",
    "chases a shadow through the dark", "shelters from sudden rain",
    "rescues a creature in trouble", "overhears a whispered plan",
    "trades an old thing for a new one", "opens a box that was sealed shut",
    "stands at a fork in the road", "learns a song from a stranger",
    "draws a map from memory", "counts the stars until dawn",
]


def build_unique_seeds(n_personas, samples_per_persona, global_seed=42):
    """Build non-overlapping seed lists for each persona.

    Returns a list of lists: seeds[persona_idx] = [seed_str, ...] of length
    samples_per_persona.  No seed string appears in more than one persona's list.
    """
    rng = random.Random(global_seed)

    # Full cross-product
    all_combos = [
        f"{subj} {setting} who {event}"
        for subj, setting, event in itertools.product(SUBJECTS, SETTINGS, EVENTS)
    ]
    rng.shuffle(all_combos)

    total_needed = n_personas * samples_per_persona
    if total_needed > len(all_combos):
        raise ValueError(
            f"Need {total_needed} unique seeds but combinatorial pool only has "
            f"{len(all_combos)}.  Add more SUBJECTS/SETTINGS/EVENTS."
        )

    seeds_per_persona = []
    for i in range(n_personas):
        start = i * samples_per_persona
        end = start + samples_per_persona
        seeds_per_persona.append(all_combos[start:end])

    return seeds_per_persona


async def generate_one(client, sem, persona_prompt, seed, model="gpt-4o-mini", max_retries=5):
    """Generate a single story from a persona."""
    user_msg = f"Story seed (use this as a starting idea, not a title): {seed}"

    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": persona_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.9,
                    max_tokens=250,
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError:
                await asyncio.sleep(min(2 ** attempt * 2, 30))
            except (APITimeoutError, APIConnectionError):
                await asyncio.sleep(min(2 ** attempt, 15))
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  FAIL: {e}")
                    return None
    return None


async def run(args):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY environment variable")

    # Load personas
    with open(args.personas) as f:
        personas = json.load(f)

    print(f"Loaded {len(personas)} personas from {args.personas}")
    for p in personas:
        print(f"  {p['id']}: {p['name']}")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.workers)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    checkpoint_path = output_dir / "_checkpoint.json"
    done = {}  # key: "{persona_id}_{sample_idx}" -> story text
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            done = json.load(f)
        print(f"Resuming: {len(done)} cached")

    # Build unique, non-overlapping seeds for each persona
    persona_seeds = build_unique_seeds(
        n_personas=len(personas),
        samples_per_persona=args.samples_per_persona,
        global_seed=args.seed,
    )

    # Build work items — each (persona, sample_idx) gets its own unique seed
    work = []  # (persona_id, persona_prompt, sample_idx, story_seed, key)
    for p_idx, persona in enumerate(personas):
        for i in range(args.samples_per_persona):
            key = f"{persona['id']}_{i}"
            if key not in done:
                seed = persona_seeds[p_idx][i]
                work.append((persona["id"], persona["generation_prompt"], i, seed, key))

    total = len(work) + len(done)
    print(f"Work: {len(work)} remaining, {len(done)} cached, {total} total "
          f"({args.samples_per_persona}/persona)")

    completed = 0
    failed = 0
    t0 = time.monotonic()
    ckpt_lock = asyncio.Lock()

    async def process(persona_id, persona_prompt, sample_idx, seed, key):
        nonlocal completed, failed
        story = await generate_one(client, sem, persona_prompt, seed, model=args.model)

        if story is not None:
            done[key] = story
        else:
            failed += 1
        completed += 1

        if completed % 200 == 0:
            elapsed = time.monotonic() - t0
            rate = completed / max(elapsed, 1)
            remaining = len(work) - completed
            eta = remaining / max(rate, 0.01)
            print(f"  {completed}/{len(work)}  {rate:.1f}/s  ETA {eta:.0f}s  fail={failed}")

        if completed % args.checkpoint_every == 0:
            async with ckpt_lock:
                with open(checkpoint_path, "w") as f:
                    json.dump(done, f)

    # Process in batches to avoid spawning all coroutines at once
    BATCH_SIZE = 500
    for batch_start in range(0, len(work), BATCH_SIZE):
        batch = work[batch_start : batch_start + BATCH_SIZE]
        await asyncio.gather(*(process(*item) for item in batch))

    # Final checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump(done, f)

    print(f"\nGeneration complete: {len(done)} stories, {failed} failed")

    # -----------------------------------------------------------------------
    # Build HuggingFace-compatible dataset
    # -----------------------------------------------------------------------
    persona_lookup = {p["id"]: p for p in personas}

    # Collect all stories
    rows = []
    for key, story in sorted(done.items()):
        persona_id = key.rsplit("_", 1)[0]
        sample_idx = int(key.rsplit("_", 1)[1])
        persona = persona_lookup[persona_id]
        rows.append({
            "text": story,
            "persona_id": persona_id,
            "persona_name": persona["name"],
            "persona_description": persona["description"],
            "sample_idx": sample_idx,
        })

    # Split into train / mixture / eval
    random.seed(args.seed)
    per_persona = {}
    for row in rows:
        pid = row["persona_id"]
        if pid not in per_persona:
            per_persona[pid] = []
        per_persona[pid].append(row)

    train_rows = []
    mixture_rows = []
    eval_rows = []

    for pid in sorted(per_persona):
        examples = per_persona[pid]
        random.shuffle(examples)
        n = len(examples)
        # Allocate: 70% train, 15% mixture, 15% eval
        n_train = int(n * 0.70)
        n_mix = int(n * 0.15)
        train_rows.extend(examples[:n_train])
        mixture_rows.extend(examples[n_train : n_train + n_mix])
        eval_rows.extend(examples[n_train + n_mix :])

    random.shuffle(train_rows)
    random.shuffle(mixture_rows)
    random.shuffle(eval_rows)

    # Save as JSON (HuggingFace datasets can load from JSON directly)
    splits = {"train": train_rows, "mixture": mixture_rows, "eval": eval_rows}
    for split_name, split_rows in splits.items():
        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(split_rows, f, indent=2)
        persona_counts = {}
        for r in split_rows:
            persona_counts[r["persona_id"]] = persona_counts.get(r["persona_id"], 0) + 1
        print(f"  {split_name}: {len(split_rows)} examples -> {out_path}")
        for pid in sorted(persona_counts):
            print(f"    {pid}: {persona_counts[pid]}")

    # Also save per-persona train files (for training individual LoRAs)
    persona_dir = output_dir / "per_persona"
    persona_dir.mkdir(exist_ok=True)
    for pid in sorted(per_persona):
        persona_train = [r for r in train_rows if r["persona_id"] == pid]
        out_path = persona_dir / f"{pid}.json"
        with open(out_path, "w") as f:
            json.dump(persona_train, f, indent=2)
        print(f"  per_persona/{pid}.json: {len(persona_train)} examples")

    # Save metadata
    meta = {
        "personas": personas,
        "n_personas": len(personas),
        "samples_per_persona": args.samples_per_persona,
        "model": args.model,
        "splits": {name: len(rows) for name, rows in splits.items()},
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata -> {meta_path}")

    # Save dataset_info.json for HuggingFace Hub compatibility
    dataset_info = {
        "description": (
            "Structurally distinct story personas for Bayesian persona decomposition. "
            f"{len(personas)} personas, each with distinct vocabulary, sentence structure, "
            "POV, dialogue ratio, tone, and topic."
        ),
        "features": {
            "text": {"dtype": "string", "_type": "Value"},
            "persona_id": {"dtype": "string", "_type": "Value"},
            "persona_name": {"dtype": "string", "_type": "Value"},
            "persona_description": {"dtype": "string", "_type": "Value"},
            "sample_idx": {"dtype": "int32", "_type": "Value"},
        },
        "splits": {
            name: {"num_examples": len(rows)} for name, rows in splits.items()
        },
    }
    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.0f}s — total {len(rows)} stories across {len(personas)} personas")


def main():
    parser = argparse.ArgumentParser(
        description="Generate story dataset from personas using OpenAI API"
    )
    parser.add_argument("--personas", type=str, default="data/story_personas.json",
                        help="Path to persona definitions JSON")
    parser.add_argument("--output", type=str, default="data/story_dataset/",
                        help="Output directory for dataset")
    parser.add_argument("--samples_per_persona", type=int, default=2000,
                        help="Stories to generate per persona (default: 2000)")
    parser.add_argument("--workers", type=int, default=80,
                        help="Concurrent API requests")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model for generation")
    parser.add_argument("--checkpoint_every", type=int, default=500,
                        help="Save checkpoint every N completions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
