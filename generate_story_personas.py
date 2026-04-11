"""
Generate a set of structurally distinct story-telling personas.

Each persona is designed to produce token-level distributional differences
in children's story text — not just topical differences but structural ones
(vocabulary level, sentence length, POV, dialogue ratio, tone, pacing).

Usage:
    python generate_story_personas.py --output data/story_personas.json
"""

import argparse
import json
from pathlib import Path

PERSONAS = [
    {
        "id": "simple_cheerful_dialogue_animals",
        "name": "Sunny Animal Friend",
        "description": "Simple vocabulary, short sentences, first-person narrator, heavy dialogue, cheerful tone, stories about animals.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) told in FIRST PERSON by a cheerful child narrator. "
            "The story must be about animals (pets, farm animals, woodland creatures). "
            "Use VERY SIMPLE vocabulary (ages 4-6 level). Keep sentences SHORT (under 10 words each). "
            "At least HALF the story should be DIALOGUE between characters. "
            "The tone must be sunny, happy, and enthusiastic. Use lots of exclamation marks. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "simple",
            "sentence_length": "short",
            "pov": "first-person",
            "dialogue_ratio": "heavy",
            "tone": "cheerful",
            "topic": "animals",
        },
    },
    {
        "id": "literary_melancholic_narration_nature",
        "name": "Wistful Nature Narrator",
        "description": "Literary vocabulary, long flowing sentences, third-person omniscient, pure narration (no dialogue), melancholic tone, stories about nature and seasons.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) in THIRD-PERSON OMNISCIENT voice. "
            "The story is about nature, seasons changing, or the passage of time in a forest or garden. "
            "Use LITERARY, slightly advanced vocabulary (ages 8-10 level). "
            "Write LONG, flowing sentences with multiple clauses joined by commas and conjunctions. "
            "Use NO DIALOGUE at all — pure narration only. "
            "The tone should be wistful, gentle, and slightly melancholic — a sense of beauty mixed with sadness. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "literary",
            "sentence_length": "long",
            "pov": "third-person-omniscient",
            "dialogue_ratio": "none",
            "tone": "melancholic",
            "topic": "nature/seasons",
        },
    },
    {
        "id": "simple_tense_secondperson_adventure",
        "name": "Choose-Your-Adventure Guide",
        "description": "Simple vocabulary, medium sentences, second-person narrator (you), mixed dialogue, tense and suspenseful tone, adventure stories.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) in SECOND PERSON — address the reader as 'you' throughout. "
            "It should be an adventure or quest story (exploring caves, crossing rivers, finding treasure). "
            "Use SIMPLE vocabulary but create SUSPENSE and TENSION. "
            "Mix narration with occasional dialogue from characters the reader meets. "
            "Use medium-length sentences. Build to a moment of danger or surprise, then resolve it. "
            "The reader should feel like they are IN the story making choices. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "simple",
            "sentence_length": "medium",
            "pov": "second-person",
            "dialogue_ratio": "mixed",
            "tone": "tense/suspenseful",
            "topic": "adventure",
        },
    },
    {
        "id": "archaic_whimsical_narration_fantasy",
        "name": "Old-World Fairy Tale Teller",
        "description": "Archaic fairy-tale vocabulary, long ornate sentences, third-person, no dialogue, whimsical tone, fantasy and magic stories.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) in the style of a classic FAIRY TALE. "
            "Use ARCHAIC, old-fashioned language ('once upon a time', 'there dwelt', 'forthwith', 'alas'). "
            "Write LONG, ornate sentences with rich descriptions. Third-person narrator. "
            "The story involves MAGIC or FANTASY elements (enchanted objects, talking forests, spells). "
            "Use NO DIALOGUE — tell everything through narration. "
            "The tone should be whimsical and enchanting, like a story from a very old book. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "archaic",
            "sentence_length": "long",
            "pov": "third-person",
            "dialogue_ratio": "none",
            "tone": "whimsical",
            "topic": "fantasy/magic",
        },
    },
    {
        "id": "blunt_factual_dialogue_everyday",
        "name": "Matter-of-Fact Reporter",
        "description": "Simple vocabulary, very short blunt sentences, third-person, heavy dialogue, matter-of-fact dry tone, everyday mundane stories.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) in THIRD PERSON with a dry, MATTER-OF-FACT tone. "
            "The story is about something completely ORDINARY and MUNDANE (going to the store, doing laundry, waiting for a bus). "
            "Use VERY SHORT, blunt sentences. No flowery language. State facts plainly. "
            "At least HALF the story should be DIALOGUE — characters speak in short, direct sentences too. "
            "No exclamation marks. No dramatic descriptions. Just plain reporting of what happened. "
            "The humor comes from treating something boring as if it's a story worth telling. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "simple/blunt",
            "sentence_length": "very-short",
            "pov": "third-person",
            "dialogue_ratio": "heavy",
            "tone": "matter-of-fact",
            "topic": "everyday/mundane",
        },
    },
    {
        "id": "warm_nostalgic_firstperson_family",
        "name": "Warm Family Storyteller",
        "description": "Medium vocabulary, medium sentences, first-person child narrator, mixed dialogue, warm nostalgic tone, stories about family and home.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) told in FIRST PERSON by a child remembering something. "
            "The story is about FAMILY and HOME — grandparents, siblings, cooking together, family traditions. "
            "Use a WARM, NOSTALGIC tone — the narrator looks back fondly on a memory. "
            "Medium vocabulary, medium-length sentences. Mix narration with some dialogue. "
            "Include sensory details (smells of cooking, warmth of a hug, sounds of a house). "
            "The story should feel cozy and safe, like being wrapped in a blanket. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "medium",
            "sentence_length": "medium",
            "pov": "first-person",
            "dialogue_ratio": "mixed",
            "tone": "warm/nostalgic",
            "topic": "family/home",
        },
    },
    {
        "id": "poetic_dreamy_narration_elements",
        "name": "Lyrical Dreamer",
        "description": "Poetic lyrical vocabulary, long flowing sentences, third-person, no dialogue, dreamy ethereal tone, stories about wind/water/sky/stars.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) in a POETIC, LYRICAL style. "
            "Third-person narration about natural ELEMENTS — wind, water, rain, stars, clouds, moonlight. "
            "Use BEAUTIFUL, musical vocabulary. Sentences should FLOW like poetry, long and rhythmic. "
            "Use metaphors, personification, and imagery heavily. "
            "NO DIALOGUE at all. The tone is DREAMY and ethereal, almost like a lullaby in prose form. "
            "Repeat certain phrases or structures for rhythmic effect. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "poetic/lyrical",
            "sentence_length": "long",
            "pov": "third-person",
            "dialogue_ratio": "none",
            "tone": "dreamy",
            "topic": "elements/sky",
        },
    },
    {
        "id": "energetic_excited_dialogue_sports",
        "name": "Hyper Sports Commentator",
        "description": "Simple direct vocabulary, short punchy sentences, third-person, heavy dialogue, excited energetic tone, stories about games and competitions.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) in THIRD PERSON with an ENERGETIC, EXCITED tone. "
            "Like a sports commentator telling a story! The story is about a GAME, RACE, or COMPETITION. "
            "Use SHORT, PUNCHY sentences. Lots of ACTION VERBS. Lots of EXCLAMATION MARKS! "
            "At least HALF should be DIALOGUE — characters shout, cheer, and talk fast. "
            "Simple, direct vocabulary. Fast pacing. The reader should feel the excitement and energy. "
            "Build to a climactic moment in the game, then a winner. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "simple/direct",
            "sentence_length": "short/punchy",
            "pov": "third-person",
            "dialogue_ratio": "heavy",
            "tone": "excited/energetic",
            "topic": "sports/games",
        },
    },
    {
        "id": "gentle_anxious_firstperson_insects",
        "name": "Tiny Worried Creature",
        "description": "Gentle vocabulary, medium hesitant sentences, first-person, mixed dialogue, cautious anxious tone, stories about small creatures and insects.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) in FIRST PERSON from the perspective of a SMALL, ANXIOUS creature "
            "(a ladybug, a mouse, a snail, a caterpillar). "
            "The narrator is CAUTIOUS and WORRIED — everything feels big and scary to them. "
            "Use gentle, soft vocabulary. Sentences should feel HESITANT — use dashes, ellipses, and qualifiers "
            "('maybe', 'I think', 'perhaps', 'I wasn't sure'). "
            "Mix narration with some quiet dialogue with other small creatures. "
            "The story should end with the small creature finding courage or safety. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "gentle/soft",
            "sentence_length": "medium/hesitant",
            "pov": "first-person",
            "dialogue_ratio": "mixed",
            "tone": "cautious/anxious",
            "topic": "small-creatures/insects",
        },
    },
    {
        "id": "formal_analytical_narration_science",
        "name": "Curious Young Scientist",
        "description": "Formal precise vocabulary, medium structured sentences, third-person, narration-heavy with some dialogue, curious analytical tone, stories about discovery and science.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) in THIRD PERSON about a child making a SCIENTIFIC DISCOVERY. "
            "Use FORMAL, PRECISE vocabulary — the narrator explains things carefully and exactly. "
            "Include some scientific or technical words (appropriately for ages 7-10). "
            "Sentences should be STRUCTURED and CLEAR, medium length, with logical connectors "
            "('therefore', 'however', 'as a result', 'furthermore'). "
            "Mostly narration with occasional dialogue where the character explains their findings. "
            "The tone is CURIOUS and ANALYTICAL — the character observes, hypothesizes, and tests. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "formal/precise",
            "sentence_length": "medium/structured",
            "pov": "third-person",
            "dialogue_ratio": "narration-heavy",
            "tone": "curious/analytical",
            "topic": "science/discovery",
        },
    },
    {
        "id": "rhythmic_silly_mixed_food",
        "name": "Silly Kitchen Rhymer",
        "description": "Rhythmic repetitive vocabulary, short bouncy sentences, third-person, mixed dialogue, silly humorous tone, stories about food and cooking.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) in THIRD PERSON with a SILLY, HUMOROUS tone. "
            "The story is about FOOD or COOKING — a recipe gone wrong, a picky eater, a magical meal. "
            "Use RHYTHMIC, REPETITIVE language — repeat phrases, use alliteration, make words bounce. "
            "SHORT, bouncy sentences. Some should almost rhyme. "
            "Mix dialogue with narration. Characters should say funny things about food. "
            "Use onomatopoeia (splat, sizzle, crunch, plop). "
            "The story should make a child laugh. Be absurd and playful. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "rhythmic/repetitive",
            "sentence_length": "short/bouncy",
            "pov": "third-person",
            "dialogue_ratio": "mixed",
            "tone": "silly/humorous",
            "topic": "food/cooking",
        },
    },
    {
        "id": "atmospheric_mysterious_narration_night",
        "name": "Midnight Mystery Weaver",
        "description": "Dark atmospheric vocabulary, long suspenseful sentences, third-person, narration-heavy, mysterious eerie tone, stories about nighttime and shadows.",
        "generation_prompt": (
            "Write a short children's story (100-150 words) in THIRD PERSON set at NIGHTTIME. "
            "Use ATMOSPHERIC, slightly dark vocabulary — shadows, whispers, flickering, silence, moonlight. "
            "Write LONG, suspenseful sentences that build tension slowly. "
            "Mostly NARRATION with very little dialogue (at most one or two whispered lines). "
            "The tone is MYSTERIOUS and slightly eerie — but NOT scary for children. "
            "The mystery should resolve into something gentle or beautiful (the shadow was a cat, the whisper was the wind). "
            "Focus on SOUNDS and DARKNESS and what the character IMAGINES they see. "
            "Output ONLY the story text, no title."
        ),
        "axes": {
            "vocabulary": "atmospheric/dark",
            "sentence_length": "long/suspenseful",
            "pov": "third-person",
            "dialogue_ratio": "minimal",
            "tone": "mysterious/eerie",
            "topic": "nighttime/shadows",
        },
    },
]


def main():
    parser = argparse.ArgumentParser(description="Generate structurally distinct story personas")
    parser.add_argument("--output", type=str, default="data/story_personas.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(PERSONAS, f, indent=2)

    print(f"Saved {len(PERSONAS)} personas -> {output_path}")
    print()
    for p in PERSONAS:
        axes = p["axes"]
        print(f"  {p['id']}")
        print(f"    vocab={axes['vocabulary']}, sentences={axes['sentence_length']}, "
              f"pov={axes['pov']}, dialogue={axes['dialogue_ratio']}, "
              f"tone={axes['tone']}, topic={axes['topic']}")


if __name__ == "__main__":
    main()
