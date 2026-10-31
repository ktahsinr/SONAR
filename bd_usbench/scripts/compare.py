"""Collect every results/**/metrics.json into one markdown comparison table.

Usage:
    python -m bd_usbench.scripts.compare --results-dir results \
        [--out results/comparison.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bd_usbench.utils import get_logger, load_json

logger = get_logger(__name__)


def collect_rows(results_dir: str | Path) -> list[dict]:
    """Load all metrics.json files found under ``results_dir``.

    When a sibling bootstrap.json exists (written by scripts.analyze), its 95%
    CIs are attached so the comparison table can show a CI column.
    """
    rows = []
    for metrics_file in sorted(Path(results_dir).glob("**/metrics.json")):
        try:
            row = load_json(metrics_file)
        except Exception as exc:
            logger.warning("Skipping unreadable %s: %s", metrics_file, exc)
            continue
        bootstrap_file = metrics_file.parent / "bootstrap.json"
        if bootstrap_file.exists():
            try:
                boot = load_json(bootstrap_file)
                row["macro_f1_ci95"] = boot.get("macro_f1_ci95")
                row["accuracy_ci95"] = boot.get("accuracy_ci95")
            except Exception as exc:
                logger.warning("Could not read %s: %s", bootstrap_file, exc)
        rows.append(row)
    return rows


def _fmt_ci(ci) -> str:
    if not ci:
        return "n/a"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def build_table(rows: list[dict]) -> str:
    """Markdown table sorted by macro-F1 (desc); CI column when any run has
    bootstrap CIs from scripts.analyze."""
    with_ci = any(r.get("macro_f1_ci95") for r in rows)
    ci_header = "macro-F1 95% CI | " if with_ci else ""
    ci_sep = "---|" if with_ci else ""
    header = (
        f"| run | track | split | domain | accuracy | macro-F1 | {ci_header}ECE | "
        "latency (ms/img) | device | n | unparsed |\n"
        f"|---|---|---|---|---|---|{ci_sep}---|---|---|---|---|"
    )
    lines = [header]
    for m in sorted(rows, key=lambda r: r.get("macro_f1", 0.0), reverse=True):
        ece = f"{m['ece']:.4f}" if m.get("ece") is not None else "n/a"
        ci_cell = f"{_fmt_ci(m.get('macro_f1_ci95'))} | " if with_ci else ""
        lines.append(
            f"| {m.get('run_name', '?')} | {str(m.get('track', '?')).upper()} "
            f"| {m.get('split', '?')} | {m.get('domain', '?')} "
            f"| {m.get('accuracy', 0.0):.4f} | {m.get('macro_f1', 0.0):.4f} | {ci_cell}{ece} "
            f"| {m.get('latency', {}).get('mean_ms', '?')} | {m.get('device', '?')} "
            f"| {m.get('n_samples', '?')} | {m.get('n_unparsed', 0)} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="Root results folder")
    parser.add_argument("--out", default=None,
                        help="Output markdown path (default <results-dir>/comparison.md)")
    args = parser.parse_args(argv)

    rows = collect_rows(args.results_dir)
    if not rows:
        logger.error("No metrics.json files found under %s", args.results_dir)
        return 1
    table = "# BD-USBench results comparison\n\n" + build_table(rows) + "\n"
    out = Path(args.out) if args.out else Path(args.results_dir) / "comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table, encoding="utf-8")
    print(table)
    logger.info("Comparison written to %s (%d runs)", out, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
