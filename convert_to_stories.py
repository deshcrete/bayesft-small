"""
Convert persona QA completions into children's stories using GPT-4o-mini.

Loads the HuggingFace dataset, sends each completion to OpenAI to be rewritten
as a simple children's story that preserves the persona's voice, and saves the
result as a HuggingFace dataset with columns: persona, completion.

Usage:
    python convert_to_stories.py --output data/stories/
    python convert_to_stories.py --output data/stories/ --max_per_persona 2000 --workers 50

Requires: OPENAI_API_KEY env var.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError

DATASET_ID = "desh2806/bayesft-different"

SYSTEM_PROMPT = """\
You are a rewriter. You will receive a persona description and an essay-style \
completion written in that persona's voice. Rewrite the completion as a short \
children's story (suitable for ages 5-10) that preserves the persona's \
distinctive tone, values, and worldview.

Rules:
- Output ONLY the story text, no titles or meta-commentary.
- Keep it between 150-300 words.
- Use simple vocabulary and short sentences.
- Include a character, a small problem or adventure, and a resolution.
- The persona's traits should come through in how the narrator tells the story \
and what the characters care about."""

# Need 1080 train + 180 mixture + ~167 eval = ~1427 per persona minimum
DEFAULT_PER_PERSONA = 2000


async def rewrite_one(client, sem, persona, completion, model="gpt-4o-mini", max_retries=6):
    """Call OpenAI to rewrite a single completion as a children's story."""
    user_msg = f"PERSONA: {persona}\n\nORIGINAL COMPLETION:\n{completion}"

    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.7,
                    max_tokens=400,
                )
                return resp.choices[0].message.content.strip()
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                backoff = min(2 ** attempt + 1, 60)
                if isinstance(e, RateLimitError):
                    backoff = min(2 ** attempt * 5, 120)
                await asyncio.sleep(backoff)
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  FAIL idx: {e}")
                    return None
    return None


async def run(args):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY environment variable")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.workers)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    checkpoint_path = output_dir / "_checkpoint.json"
    done = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            done = {int(k): v for k, v in json.load(f).items()}
        print(f"Resuming: {len(done)} cached")

    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train")

    # Group by persona
    persona_indices = {}
    for i, ex in enumerate(ds):
        p = ex["persona"]
        if p not in persona_indices:
            persona_indices[p] = []
        persona_indices[p].append(i)

    cap = args.max_per_persona
    work = []
    for p, indices in persona_indices.items():
        for idx in indices[:cap]:
            if idx not in done:
                work.append(idx)

    total = len(work) + len(done)
    print(f"Work: {len(work)} remaining, {len(done)} cached, {total} total ({cap}/persona)")

    # Progress tracking — lock-free counter
    completed = 0
    failed = 0
    t0 = time.monotonic()

    async def process(idx):
        nonlocal completed, failed
        ex = ds[idx]
        story = await rewrite_one(client, sem, ex["persona"], ex["completion"], model=args.model)

        if story is not None:
            done[idx] = story
        else:
            failed += 1
        completed += 1

        if completed % 200 == 0:
            elapsed = time.monotonic() - t0
            rate = completed / max(elapsed, 1)
            eta = (total - completed) / max(rate, 0.01)
            print(f"  {completed}/{total}  {rate:.1f}/s  ETA {eta:.0f}s  fail={failed}")

        if completed % args.checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({str(k): v for k, v in done.items()}, f)

    tasks = [process(idx) for idx in work]
    await asyncio.gather(*tasks)

    # Final save
    with open(checkpoint_path, "w") as f:
        json.dump({str(k): v for k, v in done.items()}, f)

    results = []
    for idx, story in sorted(done.items()):
        ex = ds[int(idx)]
        results.append({"persona": ex["persona"], "completion": story})

    out_path = output_dir / "stories.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.0f}s — {len(results)} stories, {failed} failed -> {out_path}")
    for p in sorted(persona_indices):
        c = sum(1 for r in results if r["persona"] == p)
        print(f"  {p[:55]}... : {c}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/stories/")
    parser.add_argument("--max_per_persona", type=int, default=DEFAULT_PER_PERSONA,
                        help="Examples per persona (default: 2000)")
    parser.add_argument("--workers", type=int, default=80,
                        help="Concurrent API requests")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--checkpoint_every", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
