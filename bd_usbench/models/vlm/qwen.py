"""Generative VLM wrapper for Qwen2.5-VL-style models (Tracks B and C).

The model name is a config value so a newer Qwen-VL or MedGemma checkpoint can
be swapped in. Heavy imports (transformers, bitsandbytes) are lazy. The 4-bit
path requires CUDA; on CPU the model loads in full precision (slow — intended
only for tiny debug runs).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from bd_usbench.models.vlm.prompts import build_vlm_question, parse_vlm_answer
from bd_usbench.utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def build_messages(image: str | Path | Image.Image, question: str) -> list[dict]:
    """Chat-format messages with one image and one text question."""
    if isinstance(image, (str, Path)):
        image_entry = {"type": "image", "image": f"file://{Path(image).resolve()}"}
    else:
        image_entry = {"type": "image", "image": image}
    return [{"role": "user", "content": [image_entry, {"type": "text", "text": question}]}]


class QwenVLM:
    """Loads a Qwen-VL model (optionally 4-bit) and classifies single images."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        load_in_4bit: bool = True,
        adapter_path: str | None = None,
        max_pixels: int = 768 * 768,
        min_pixels: int = 224 * 224,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor  # lazy

        quantization_config = None
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_4bit:
            logger.warning("4-bit requested but CUDA unavailable; loading full precision.")

        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            logger.info("Loaded LoRA adapter from %s", adapter_path)
        self.model.eval()
        self.model_name = model_name
        logger.info("QwenVLM ready: %s (4bit=%s)", model_name, quantization_config is not None)

    @torch.no_grad()
    def generate(self, image: str | Path | Image.Image, question: str, max_new_tokens: int = 16) -> str:
        """Generate a short answer for one image."""
        from qwen_vl_utils import process_vision_info  # lazy

        messages = build_messages(image, question)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)
        output_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

    def classify(
        self, image: str | Path | Image.Image, classes: list[str]
    ) -> tuple[str | None, str]:
        """Classify one image; returns (parsed class or None, raw model output)."""
        raw = self.generate(image, build_vlm_question(classes))
        return parse_vlm_answer(raw, classes), raw
