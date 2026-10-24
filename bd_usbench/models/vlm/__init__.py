"""VLM wrappers: CLIP-style zero-shot and generative Qwen-VL (zero-shot + LoRA)."""

from bd_usbench.models.vlm.prompts import (
    CLIP_TEMPLATES,
    build_vlm_question,
    organ_display_name,
    parse_vlm_answer,
)

__all__ = ["CLIP_TEMPLATES", "build_vlm_question", "organ_display_name", "parse_vlm_answer"]
