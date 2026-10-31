"""Per-image prediction analysis across runs: bootstrap CIs, domain breakdowns,
per-class tables, and cross-run oracle / error-overlap comparisons.

Usage:
    python -m bd_usbench.scripts.analyze --results-dir results --out results/analysis

Reads every results/<run_name>/eval_<split>_<domain>/predictions.csv, then writes:
  - <out>/analysis.md and <out>/analysis.json (all tables below),
  - bootstrap.json next to each predictions.csv (so compare.py can show CIs).

Per run: bootstrap 95% CI (1000 resamples, seed 0) for accuracy and macro-F1
over present classes, accuracy/macro-F1 split by domain, and a per-class table.
Across runs sharing a (split, domain): an oracle table ("at least one correct")
for every pair of runs and for the best supervised vs best VLM run, the images
every run gets wrong, and the images only the LoRA model gets right.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from bd_usbench.utils import get_logger, load_json, save_json

logger = get_logger(__name__)

SUPERVISED_TRACKS = ("a",)
VLM_TRACKS = ("b", "c")


def find_prediction_files(results_dir: str | Path) -> list[dict]:
    """Every predictions.csv under ``results_dir`` with its run metadata.

    Metadata comes from the sibling metrics.json when present, else from the
    results/<run_name>/eval_<split>_<domain>/ path convention.
    """
    runs = []
    for csv_path in sorted(Path(results_dir).glob("**/predictions.csv")):
        eval_dir = csv_path.parent
        try:
            df = pd.read_csv(csv_path, dtype={"filename": str}, keep_default_na=False)
        except Exception as exc:
            logger.warning("Skipping unreadable %s: %s", csv_path, exc)
            continue
        meta = {"run_name": eval_dir.parent.name, "track": "?", "split": "?", "domain": "?"}
        parts = eval_dir.name.split("_")  # eval_<split>_<domain>
        if len(parts) == 3 and parts[0] == "eval":
            meta["split"], meta["domain"] = parts[1], parts[2]
        metrics_file = eval_dir / "metrics.json"
        if metrics_file.exists():
            try:
                m = load_json(metrics_file)
                meta.update({k: m[k] for k in ("run_name", "track", "split", "domain") if k in m})
            except Exception as exc:
                logger.warning("Could not read %s: %s", metrics_file, exc)
        runs.append({"eval_dir": eval_dir, "df": df, **meta})
    return runs


def _accuracy(df: pd.DataFrame) -> float:
    return float(df["correct"].astype(bool).mean()) if len(df) else 0.0


def _macro_f1_present(df: pd.DataFrame) -> float:
    """Macro-F1 averaged only over classes present in true_label."""
    if not len(df):
        return 0.0
    labels = sorted(df["true_label"].unique())
    return float(
        f1_score(df["true_label"], df["pred_label"], labels=labels,
                 average="macro", zero_division=0)
    )


def bootstrap_ci(
    df: pd.DataFrame, n_resamples: int = 1000, seed: int = 0
) -> dict:
    """Bootstrap 95% percentile CIs for accuracy and macro-F1 (present classes).

    Resamples rows with replacement; macro-F1 of each resample averages over
    the classes present in that resample's true labels.
    """
    n = len(df)
    result = {
        "accuracy": _accuracy(df),
        "macro_f1": _macro_f1_present(df),
        "n_samples": n,
        "n_resamples": n_resamples,
        "seed": seed,
    }
    if n == 0:
        result["accuracy_ci95"] = [0.0, 0.0]
        result["macro_f1_ci95"] = [0.0, 0.0]
        return result
    rng = np.random.default_rng(seed)
    correct = df["correct"].astype(bool).to_numpy()
    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()
    accs = np.empty(n_resamples)
    f1s = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        accs[b] = correct[idx].mean()
        labels = sorted(set(y_true[idx]))
        f1s[b] = f1_score(y_true[idx], y_pred[idx], labels=labels,
                          average="macro", zero_division=0)
    result["accuracy_ci95"] = [float(v) for v in np.percentile(accs, [2.5, 97.5])]
    result["macro_f1_ci95"] = [float(v) for v in np.percentile(f1s, [2.5, 97.5])]
    return result


def by_domain(df: pd.DataFrame) -> dict:
    """Accuracy and macro-F1 per domain value found in the predictions."""
    out = {}
    for dom in sorted(df["domain"].unique()):
        sub = df[df["domain"] == dom]
        out[dom] = {
            "n_samples": int(len(sub)),
            "accuracy": _accuracy(sub),
            "macro_f1": _macro_f1_present(sub),
        }
    return out


def per_class_table(df: pd.DataFrame) -> dict:
    """Precision/recall/F1/support per class present in true_label."""
    if not len(df):
        return {}
    labels = sorted(df["true_label"].unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        df["true_label"], df["pred_label"], labels=labels, zero_division=0
    )
    return {
        cls: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(labels)
    }


def _correct_by_filename(df: pd.DataFrame) -> pd.Series:
    return df.set_index("filename")["correct"].astype(bool)


def oracle_pair(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict | None:
    """Oracle ("at least one correct") accuracy over the common filenames."""
    a = _correct_by_filename(df_a)
    b = _correct_by_filename(df_b)
    common = a.index.intersection(b.index)
    if common.empty:
        return None
    a, b = a.loc[common], b.loc[common]
    return {
        "n_common": int(len(common)),
        "accuracy_a": float(a.mean()),
        "accuracy_b": float(b.mean()),
        "oracle_accuracy": float((a | b).mean()),
        "both_correct": float((a & b).mean()),
        "only_a_correct": int((a & ~b).sum()),
        "only_b_correct": int((~a & b).sum()),
    }


def cross_run_analysis(runs: list[dict]) -> dict:
    """Oracle pairs, all-wrong images, and LoRA-only-correct images, grouped by
    (split, domain) so only comparable evaluations are joined."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        groups.setdefault((run["split"], run["domain"]), []).append(run)

    analysis: dict = {}
    for (split, domain), group in sorted(groups.items()):
        key = f"{split}_{domain}"
        entry: dict = {"split": split, "domain": domain,
                       "runs": [r["run_name"] for r in group]}

        pairs = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pair = oracle_pair(group[i]["df"], group[j]["df"])
                if pair is not None:
                    pairs.append({"run_a": group[i]["run_name"],
                                  "run_b": group[j]["run_name"], **pair})
        entry["oracle_pairs"] = pairs

        supervised = [r for r in group if str(r["track"]) in SUPERVISED_TRACKS]
        vlms = [r for r in group if str(r["track"]) in VLM_TRACKS]
        if supervised and vlms:
            best_sup = max(supervised, key=lambda r: _accuracy(r["df"]))
            best_vlm = max(vlms, key=lambda r: _accuracy(r["df"]))
            pair = oracle_pair(best_sup["df"], best_vlm["df"])
            if pair is not None:
                entry["best_supervised_vs_best_vlm"] = {
                    "run_a": best_sup["run_name"], "run_b": best_vlm["run_name"], **pair
                }

        if len(group) >= 2:
            table = pd.DataFrame(
                {r["run_name"]: _correct_by_filename(r["df"]) for r in group}
            ).dropna()
            entry["n_common_images"] = int(len(table))
            entry["all_wrong"] = sorted(table.index[~table.any(axis=1)])
            lora_runs = [r["run_name"] for r in group if str(r["track"]) == "c"]
            for lora in lora_runs:
                others = [c for c in table.columns if c != lora]
                only = table.index[table[lora] & ~table[others].any(axis=1)]
                entry.setdefault("only_lora_correct", {})[lora] = sorted(only)

        analysis[key] = entry
    return analysis


def _fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def build_markdown(per_run: dict, cross: dict) -> str:
    """Human-readable analysis.md from the per-run and cross-run dicts."""
    lines = ["# BD-USBench prediction analysis", ""]

    lines += ["## Per-run metrics with bootstrap 95% CIs", "",
              "| run | track | split | domain | n | accuracy | acc 95% CI "
              "| macro-F1 | F1 95% CI |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, r in sorted(per_run.items(), key=lambda kv: -kv[1]["bootstrap"]["macro_f1"]):
        b = r["bootstrap"]
        lines.append(
            f"| {r['run_name']} | {str(r['track']).upper()} | {r['split']} | {r['domain']} "
            f"| {b['n_samples']} | {b['accuracy']:.4f} | {_fmt_ci(b['accuracy_ci95'])} "
            f"| {b['macro_f1']:.4f} | {_fmt_ci(b['macro_f1_ci95'])} |"
        )
    lines.append("")

    lines += ["## Accuracy / macro-F1 by domain", "",
              "| run | split | domain filter | domain | n | accuracy | macro-F1 |",
              "|---|---|---|---|---|---|---|"]
    for key, r in sorted(per_run.items()):
        for dom, m in r["by_domain"].items():
            lines.append(
                f"| {r['run_name']} | {r['split']} | {r['domain']} | {dom} "
                f"| {m['n_samples']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |"
            )
    lines.append("")

    lines += ["## Per-class tables", ""]
    for key, r in sorted(per_run.items()):
        lines += [f"### {r['run_name']} ({r['split']}, domain={r['domain']})", "",
                  "| class | precision | recall | f1 | support |",
                  "|---|---|---|---|---|"]
        for cls, m in r["per_class"].items():
            lines.append(
                f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} "
                f"| {m['f1']:.3f} | {m['support']} |"
            )
        lines.append("")

    for key, entry in sorted(cross.items()):
        lines += [f"## Cross-run analysis ({entry['split']}, domain={entry['domain']})", ""]
        if entry["oracle_pairs"]:
            lines += ["### Oracle (at least one correct) for every pair of runs", "",
                      "| run A | run B | n | acc A | acc B | oracle | both correct |",
                      "|---|---|---|---|---|---|---|"]
            for p in entry["oracle_pairs"]:
                lines.append(
                    f"| {p['run_a']} | {p['run_b']} | {p['n_common']} "
                    f"| {p['accuracy_a']:.4f} | {p['accuracy_b']:.4f} "
                    f"| {p['oracle_accuracy']:.4f} | {p['both_correct']:.4f} |"
                )
            lines.append("")
        if "best_supervised_vs_best_vlm" in entry:
            p = entry["best_supervised_vs_best_vlm"]
            lines += ["### Best supervised vs best VLM", "",
                      f"- Best supervised: **{p['run_a']}** (acc {p['accuracy_a']:.4f})",
                      f"- Best VLM: **{p['run_b']}** (acc {p['accuracy_b']:.4f})",
                      f"- Oracle accuracy: **{p['oracle_accuracy']:.4f}** "
                      f"over {p['n_common']} common images "
                      f"(only supervised right: {p['only_a_correct']}, "
                      f"only VLM right: {p['only_b_correct']})", ""]
        if "all_wrong" in entry:
            lines += [f"### Images every run gets wrong "
                      f"({len(entry['all_wrong'])} of {entry['n_common_images']})", ""]
            lines += [f"- {fn}" for fn in entry["all_wrong"]] or ["(none)"]
            lines.append("")
        for lora, files in entry.get("only_lora_correct", {}).items():
            lines += [f"### Images only the LoRA model ({lora}) gets right "
                      f"({len(files)})", ""]
            lines += [f"- {fn}" for fn in files] or ["(none)"]
            lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="Root results folder")
    parser.add_argument("--out", default=None,
                        help="Output folder (default <results-dir>/analysis)")
    parser.add_argument("--n-resamples", type=int, default=1000, help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed")
    args = parser.parse_args(argv)

    runs = find_prediction_files(args.results_dir)
    if not runs:
        logger.error("No predictions.csv files found under %s", args.results_dir)
        return 1

    per_run: dict = {}
    for run in runs:
        key = f"{run['run_name']}/{run['eval_dir'].name}"
        boot = bootstrap_ci(run["df"], n_resamples=args.n_resamples, seed=args.seed)
        save_json(boot, run["eval_dir"] / "bootstrap.json")  # picked up by compare.py
        per_run[key] = {
            "run_name": run["run_name"],
            "track": run["track"],
            "split": run["split"],
            "domain": run["domain"],
            "bootstrap": boot,
            "by_domain": by_domain(run["df"]),
            "per_class": per_class_table(run["df"]),
        }
        logger.info(
            "Analyzed %s: acc=%.4f %s macro-F1=%.4f %s",
            key, boot["accuracy"], _fmt_ci(boot["accuracy_ci95"]),
            boot["macro_f1"], _fmt_ci(boot["macro_f1_ci95"]),
        )

    cross = cross_run_analysis(runs)

    out_dir = Path(args.out) if args.out else Path(args.results_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({"per_run": per_run, "cross_run": cross}, out_dir / "analysis.json")
    (out_dir / "analysis.md").write_text(build_markdown(per_run, cross), encoding="utf-8")
    logger.info("Analysis written to %s (%d runs)", out_dir, len(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
