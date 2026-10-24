"""Prompt templates and robust answer parsing shared by Tracks B and C."""

from __future__ import annotations

import difflib
import re

CLIP_TEMPLATES: tuple[str, ...] = (
    "an ultrasound image of the {}",
    "a B-mode ultrasound scan of the {}",
    "a sonogram showing the {}",
    "an ultrasound examination of the {}",
)


def organ_display_name(organ: str) -> str:
    """snake_case label -> human-readable name ('fetal_head' -> 'fetal head')."""
    return organ.replace("_", " ").strip()


def build_vlm_question(classes: list[str]) -> str:
    """Instruction prompt for generative VLMs with the exact class list.

    The same prompt is used for Track B zero-shot and Track C LoRA training,
    so the fine-tuned model sees an identical input distribution.
    """
    options = ", ".join(organ_display_name(c) for c in classes)
    return (
        "This is a B-mode ultrasound image. Which organ is primarily shown? "
        f"Answer with exactly one option from this list and nothing else: {options}."
    )


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s_]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_vlm_answer(text: str, classes: list[str]) -> str | None:
    """Map a free-form VLM answer to the nearest class, or None if unparseable.

    Strategy: exact match on snake_case -> exact match on display name ->
    substring containment (longest class first, so 'fetal head' beats 'head')
    -> difflib fuzzy match with cutoff 0.6.
    """
    if not text:
        return None
    norm = _normalize(text)
    snake = norm.replace(" ", "_")

    if snake in classes:
        return snake

    display_to_class = {organ_display_name(c): c for c in classes}
    if norm in display_to_class:
        return display_to_class[norm]

    for display in sorted(display_to_class, key=len, reverse=True):
        pattern = r"\b" + re.escape(display) + r"\b"
        if re.search(pattern, norm):
            return display_to_class[display]

    matches = difflib.get_close_matches(snake, classes, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    matches = difflib.get_close_matches(norm, list(display_to_class), n=1, cutoff=0.6)
    if matches:
        return display_to_class[matches[0]]
    return None
