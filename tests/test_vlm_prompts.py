"""Prompt building and robust VLM answer parsing (no model download needed)."""

from __future__ import annotations

from bd_usbench.models.vlm.prompts import (
    build_vlm_question,
    organ_display_name,
    parse_vlm_answer,
)

CLASSES = ["bladder", "fetal_head", "heart", "kidney", "liver"]


def test_display_name():
    assert organ_display_name("fetal_head") == "fetal head"
    assert organ_display_name("liver") == "liver"


def test_question_contains_all_classes():
    question = build_vlm_question(CLASSES)
    for organ in CLASSES:
        assert organ_display_name(organ) in question


def test_parse_exact_and_display():
    assert parse_vlm_answer("liver", CLASSES) == "liver"
    assert parse_vlm_answer("fetal_head", CLASSES) == "fetal_head"
    assert parse_vlm_answer("fetal head", CLASSES) == "fetal_head"
    assert parse_vlm_answer("Liver.", CLASSES) == "liver"
    assert parse_vlm_answer("  KIDNEY \n", CLASSES) == "kidney"


def test_parse_sentence_containment():
    assert parse_vlm_answer("The image shows the liver.", CLASSES) == "liver"
    assert (
        parse_vlm_answer("This appears to be a fetal head ultrasound.", CLASSES)
        == "fetal_head"
    )


def test_parse_fuzzy():
    assert parse_vlm_answer("livr", CLASSES) == "liver"
    assert parse_vlm_answer("kidny", CLASSES) == "kidney"


def test_parse_unparseable_returns_none():
    assert parse_vlm_answer("I cannot determine the anatomy", CLASSES) is None
    assert parse_vlm_answer("", CLASSES) is None
    assert parse_vlm_answer("pancreas", CLASSES) is None
