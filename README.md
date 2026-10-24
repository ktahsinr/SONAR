# BD-USBench

Organ classification on **B-mode ultrasound still images** as an assist tool for
doctors, with VLM baselines for comparison.

> **Scope of this phase:** training and still-image evaluation only.
> There is deliberately **no** real-time, screen-capture, video, webcam, or
> live-overlay functionality in this repository — that is a planned later phase
> (see [Future work](#future-work)).

## Tracks

| Track | What | Where it runs |
|---|---|---|
| **A** | Supervised classifiers (timm backbones: DINOv2 ViT-S/14, ConvNeXt-Tiny, plus a config slot for an ultrasound-specific checkpoint such as USFM/EchoCare). Linear probe or full fine-tune, class-weighted CE, early stopping on val macro-F1, AMP, TorchScript/ONNX export. | Kaggle GPU; CPU for eval/predict |
| **B** | Zero-shot VLMs: BiomedCLIP (CLIP fallback) prompt-ensembled cosine similarity, and Qwen2.5-VL-7B-Instruct (4-bit) with the exact class list in the prompt and robust answer parsing. | Kaggle GPU |
| **C** | LoRA fine-tune of the Qwen-VL model on (image → organ name) pairs with the same prompt as Track B. 4-bit base + gradient checkpointing, fits in 16 GB; adapter-only saving. | Kaggle GPU |

## Repo layout

```
bd_usbench/
  configs/     YAML configs for every experiment (one per track/backbone/mode)
  data/        dataset class, transforms, split loading, data report, synthetic data
  models/      backbone registry (timm), classifier head, VLM wrappers
  train/       Track A trainer, model export, Track C LoRA fine-tuning
  eval/        metrics, confusion matrix, calibration (ECE), latency, track runners
  predict/     single-image and batch prediction on STILL images
  scripts/     CLI entry points (all runnable with --help)
notebooks/     train_kaggle.ipynb — end-to-end Kaggle GPU notebook
tests/         pytest smoke tests on a generated synthetic dataset (no real data)
```

## Setup

### Local (laptop, CPU — evaluation / prediction / tests)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # Python 3.10–3.12 (pinned, Kaggle-compatible)
pip install -e .
# On a newer Python (e.g. 3.13/3.14) where the pins have no wheels:
pip install -r requirements-local.txt && pip install -e .
```

Or via make: `make install` (pinned) / `make install-local`, then `make test`.

### Placing the data

The images live in a private Kaggle dataset. Locally, place them like this
(or point the config at wherever they are):

```
data/
  metadata.csv      # columns: filename, patient_id, primary_organ, secondary_organ,
                    # finding, plane, machine, quality, confidence, labeled_by,
                    # reviewed_by, split, domain
  images/
    <filename>.png  # filenames referenced by metadata.csv
```

Conventions: labels are lowercase snake_case, `none` means empty, multiple
secondary organs are separated by `;`, `domain` is `fetal` or `adult`, `split`
(`train`/`val`/`test`) is already assigned **at patient level**. The
classification target is `primary_organ`. The class list is derived once and
saved to `classes.json` (path set by `data.classes_json` in every config) so
class indices are identical across all tracks.

### Data report

```bash
python -m bd_usbench.scripts.data_report --metadata data/metadata.csv --image-dir data/images
```

Prints per-class counts per split, flags classes with < 30 train images,
verifies no `patient_id` crosses splits (non-zero exit if one does), and checks
that every file in the CSV exists on disk.

## Running the tracks locally

```bash
# Track A — train (linear probe / fine-tune chosen by the config), exports TorchScript
python -m bd_usbench.scripts.train --config bd_usbench/configs/track_a_dinov2_linear.yaml

# Track B — zero-shot (no training; evaluate directly)
python -m bd_usbench.scripts.evaluate --track b --config bd_usbench/configs/track_b_biomedclip.yaml
python -m bd_usbench.scripts.evaluate --track b --config bd_usbench/configs/track_b_qwen.yaml

# Track C — LoRA fine-tune, then evaluate
python -m bd_usbench.scripts.train_lora --config bd_usbench/configs/track_c_qwen_lora.yaml
python -m bd_usbench.scripts.evaluate --track c --config bd_usbench/configs/track_c_qwen_lora.yaml

# Evaluate Track A on the test split (only after val-based selection!)
python -m bd_usbench.scripts.evaluate --track a \
    --config bd_usbench/configs/track_a_dinov2_linear.yaml --split test --domain all

# Collect everything into one markdown table
python -m bd_usbench.scripts.compare --results-dir results
```

`--domain fetal|adult|all` filters evaluation by domain. Evaluation writes to
`results/<run_name>/eval_<split>_<domain>/`: `metrics.json` (accuracy, macro-F1,
per-class precision/recall, ECE, mean per-image latency on the current device),
`summary.md`, `confusion_matrix.png`, `reliability_diagram.png` (when real
probabilities exist — the generative Qwen path produces hard predictions, so
ECE is reported as `n/a` there), and `unparsed.csv` logging any VLM outputs
that could not be mapped to a class (they count as errors).

## Running on Kaggle

Open `notebooks/train_kaggle.ipynb` in a Kaggle GPU notebook (T4/P100):

1. Attach your private image dataset (`metadata.csv` + images).
2. Attach this repo as a second Kaggle dataset, **or** set `REPO_GIT_URL` in the
   parameter cell.
3. Edit the top parameter cell (`DATASET_SLUG`, epochs, batch size, model names,
   which tracks to run) and *Run all*.

The notebook installs requirements, runs the data report, trains Track A (both
backbones, linear probe then fine-tune), runs Track B zero-shot (BiomedCLIP +
Qwen), trains and evaluates Track C LoRA, builds the comparison table, and
leaves `results/` (checkpoints, exported TorchScript model, metrics, plots)
in `/kaggle/working`.

## Still-image prediction (manual checking — NOT a live tool)

```bash
python -m bd_usbench.scripts.predict \
    --model results/track_a_dinov2_linear/export/model.torchscript.pt \
    --input path/to/image_or_folder \
    --threshold 0.6 \
    --save-annotated annotated_out/
```

Prints predicted organ + confidence per image, flags **uncertain** below the
threshold (default **0.6**), and optionally saves a copy of each image with the
prediction written on it. Runs fine on a laptop CPU via the exported
TorchScript (or ONNX) model — no training dependencies needed.

## Tests

```bash
pytest tests -v        # or: make test
```

The suite generates a synthetic dataset (random images with class-specific
patterns + a fake `metadata.csv`), then exercises the full pipeline: data
loading/filters, report + leakage checks, metrics/ECE/plots, VLM answer
parsing, Track A training on CPU (resnet18, `pretrained=false`, so no network),
evaluation, TorchScript export, and still-image prediction. No real data or
internet access required. `make smoke` runs the same pipeline via the CLI with
`configs/smoke.yaml`.

## Defaults chosen (and why)

- **Class list** is the sorted set of `primary_organ` values from the full CSV,
  frozen in `classes.json` on first run and shared by every track.
- **Horizontal flip off** by default — laterality matters in abdominal/fetal
  ultrasound. Enable with `data.hflip: true` if you decide it is safe.
- **Augmentation** is light (random resized crop 0.85–1.0, ±8° rotation,
  brightness/contrast 0.2) and train-only.
- **crop_box** (`[x, y, w, h]`) is `null` by default; set it per machine to cut
  burned-in UI/text before resizing.
- **Backbone input** 224 px (DINOv2 ViT-S/14 is created with `img_size=224`).
- **Class-weighted cross-entropy** (inverse frequency) is on by default.
- **Early stopping** on val macro-F1, patience 7–8 epochs.
- **Export** defaults to TorchScript (no extra runtime); ONNX via
  `--format onnx` (needs `onnx`/`onnxruntime`).
- **Qwen confidences:** the generative model emits a single word, so no
  calibrated probabilities exist; predictions are hard and ECE is `n/a`.
  Unparseable answers are logged and scored as errors.
- **4-bit (bitsandbytes) requires CUDA**; on CPU the Qwen wrapper falls back to
  full precision (debug only).

## Known limitations

- Data currently comes from a **single ultrasound machine**; expect a domain
  gap on other hardware.
- **Fetal and adult images are mixed** in one label space; use `--domain` to
  evaluate them separately.
- Labels cover the **primary organ only** — secondary organs, findings, and
  plane are recorded in the metadata but not yet used as targets.
- ~315 images today (growing to ~1000): several classes fall under the
  30-train-image flag, so treat per-class metrics with caution.
- This is a research benchmark, **not a medical device**; outputs must not be
  used for clinical decision-making.

## Future work

Live/real-time inference (screen-capture or video-stream overlay during
scanning) is a planned later phase and intentionally absent here. The exported
TorchScript/ONNX models and the CPU latency numbers in the evaluation reports
are the groundwork for it.
