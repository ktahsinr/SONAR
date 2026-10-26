"""Track C: LoRA fine-tuning of a Qwen-VL model on (image -> organ name) pairs.

Designed for a single 16 GB GPU (Kaggle T4/P100): 4-bit NF4 base weights,
gradient checkpointing, batch size 1 with gradient accumulation. Only the
LoRA adapter is saved. Uses the exact same question prompt as Track B so the
zero-shot and fine-tuned models are directly comparable.

All heavy imports are lazy; this module is importable without transformers.
"""

from __future__ import annotations

from pathlib import Path

import torch

from bd_usbench.data.dataset import UltrasoundDataset
from bd_usbench.models.vlm.prompts import build_vlm_question, organ_display_name
from bd_usbench.models.vlm.qwen import build_messages
from bd_usbench.train.trainer import resolve_class_list
from bd_usbench.utils import ensure_dir, get_logger, save_json, set_seed

logger = get_logger(__name__)


def _vlm_pairs(cfg: dict, classes: list[str], split: str) -> list[tuple[Path, str]]:
    """(image_path, answer_text) pairs for one split."""
    ds = UltrasoundDataset(
        metadata_csv=cfg["data"]["metadata_csv"],
        image_dir=cfg["data"]["image_dir"],
        split=split,
        domain=cfg["data"].get("domain", "all"),
        classes=classes,
        min_train_images=int(cfg["data"].get("min_train_images", 5)),
    )
    return [
        (ds.image_path(i), organ_display_name(classes[ds.label_index(i)]))
        for i in range(len(ds))
    ]


def _encode_example(processor, question: str, image_path: Path, answer: str):
    """Tokenize one (image, question, answer) example with prompt tokens masked
    out of the loss (labels = -100 for everything before the answer)."""
    from qwen_vl_utils import process_vision_info  # lazy

    messages = build_messages(image_path, question)
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)

    prompt_only = processor(text=[prompt_text], images=image_inputs, return_tensors="pt")
    full = processor(
        text=[prompt_text + answer + processor.tokenizer.eos_token],
        images=image_inputs,
        return_tensors="pt",
    )
    labels = full["input_ids"].clone()
    labels[:, : prompt_only["input_ids"].shape[1]] = -100
    full["labels"] = labels
    return full


def train_track_c(cfg: dict) -> dict:
    """LoRA fine-tune; saves the adapter (only) and returns a summary dict."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor

    set_seed(int(cfg.get("seed", 42)))
    run_dir = ensure_dir(Path(cfg["output"]["dir"]) / cfg["run_name"])
    adapter_dir = ensure_dir(run_dir / "adapter")

    classes = resolve_class_list(cfg)
    question = build_vlm_question(classes)
    train_pairs = _vlm_pairs(cfg, classes, "train")
    val_pairs = _vlm_pairs(cfg, classes, "val")

    model_cfg = cfg["model"]
    model_name = model_cfg.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
    vision_cfg = model_cfg.get("vision", {})
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=int(vision_cfg.get("min_pixels", 224 * 224)),
        max_pixels=int(vision_cfg.get("max_pixels", 640 * 640)),
    )

    quantization_config = None
    if model_cfg.get("load_in_4bit", True) and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_cfg = cfg["train"]
    epochs = int(train_cfg.get("epochs", 3))
    grad_accum = int(train_cfg.get("grad_accum_steps", 8))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    device = next(model.parameters()).device
    best_val_loss = float("inf")
    history: list[dict] = []

    def _run_split(pairs: list[tuple[Path, str]], training: bool) -> float:
        total, steps = 0.0, 0
        model.train(training)
        if training:
            optimizer.zero_grad(set_to_none=True)
        for i, (image_path, answer) in enumerate(pairs):
            batch = _encode_example(processor, question, image_path, answer)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.set_grad_enabled(training):
                loss = model(**batch).loss
            if training:
                (loss / grad_accum).backward()
                if (i + 1) % grad_accum == 0 or i == len(pairs) - 1:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            steps += 1
        return total / max(steps, 1)

    import random

    for epoch in range(1, epochs + 1):
        random.shuffle(train_pairs)
        train_loss = _run_split(train_pairs, training=True)
        val_loss = _run_split(val_pairs, training=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("LoRA epoch %d/%d | train loss %.4f | val loss %.4f",
                    epoch, epochs, train_loss, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(adapter_dir)  # adapter weights only
            logger.info("Saved adapter (val loss %.4f) to %s", val_loss, adapter_dir)

    save_json({"history": history, "best_val_loss": best_val_loss}, run_dir / "history.json")
    summary = {
        "run_name": cfg["run_name"],
        "adapter_dir": str(adapter_dir),
        "best_val_loss": best_val_loss,
        "base_model": model_name,
    }
    logger.info("LoRA training done: %s", summary)
    return summary
