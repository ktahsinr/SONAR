"""Track B: CLIP-style zero-shot classification (BiomedCLIP with CLIP fallback).

Heavy imports (open_clip, torch hub downloads) happen lazily inside the class
so the rest of the package — and the test suite — never needs them.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from bd_usbench.models.vlm.prompts import CLIP_TEMPLATES, organ_display_name
from bd_usbench.utils import get_logger

logger = get_logger(__name__)

BIOMEDCLIP_HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class ClipZeroShot:
    """Zero-shot organ classifier: cosine similarity between image and
    prompt-ensembled class text embeddings, softmax over classes."""

    def __init__(
        self,
        classes: list[str],
        model_name: str = "biomedclip",
        device: torch.device | str = "cpu",
        templates: tuple[str, ...] = CLIP_TEMPLATES,
    ) -> None:
        import open_clip  # lazy

        self.classes = list(classes)
        self.device = torch.device(device)
        self.templates = templates

        if model_name == "biomedclip":
            try:
                self.model, self.preprocess = open_clip.create_model_from_pretrained(
                    BIOMEDCLIP_HUB
                )
                self.tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_HUB)
                self.loaded_model = "biomedclip"
            except Exception as exc:  # network / hub failures
                logger.warning("BiomedCLIP unavailable (%s); falling back to CLIP ViT-B-32", exc)
                model_name = "ViT-B-32"
        if model_name != "biomedclip":
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="openai"
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.loaded_model = model_name

        self.model = self.model.to(self.device).eval()
        self.text_features = self._build_class_embeddings()
        logger.info("ClipZeroShot ready: %s, %d classes", self.loaded_model, len(classes))

    @torch.no_grad()
    def _build_class_embeddings(self) -> torch.Tensor:
        """Template-averaged, L2-normalized text embedding per class."""
        embeddings = []
        for organ in self.classes:
            prompts = [t.format(organ_display_name(organ)) for t in self.templates]
            tokens = self.tokenizer(prompts).to(self.device)
            feats = self.model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0)
            embeddings.append(mean / mean.norm())
        return torch.stack(embeddings)  # (num_classes, dim)

    @torch.no_grad()
    def predict(self, image: Image.Image | str | Path) -> torch.Tensor:
        """Return softmax probabilities over classes for one image."""
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        pixel = self.preprocess(image).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(pixel).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = 100.0 * feats @ self.text_features.T
        return logits.softmax(dim=-1).squeeze(0).cpu()
