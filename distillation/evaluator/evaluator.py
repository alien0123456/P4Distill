# -*- coding: utf-8 -*-
"""
evaluator.py

Compare baseline vs distilled experiments by:
- parsing train/test metrics from result txt files
- reading learning_curves.json (if exists)
- generating comparison plots

Inputs:
  --baseline-dir : directory containing baseline outputs (expects brnn-best-result.txt somewhere under it)
  --distill-dir  : directory containing distilled outputs (expects student-best-result.txt somewhere under it)
  --out-dir      : where to save plots & summary (default: distill-dir)

Expected filenames:
  baseline: brnn-best-result.txt
  distill : student-best-result.txt
Optional:
  learning_curves.json under each dir (distill strongly recommended)

This script does NOT run model inference; it is for evaluation+analysis reporting.
"""

import os
import re
import json
import argparse
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: file searching
# -----------------------------
def find_file_recursive(root_dir: str, filename: str) -> Optional[str]:
    """Find the first occurrence of filename under root_dir (recursive)."""
    if not root_dir or (not os.path.exists(root_dir)):
        return None
    for cur, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(cur, filename)
    return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Parsing: result txt
# -----------------------------
TRAINING_RE = re.compile(r"^\s*Training set:\s*(\d+)\s*segs,\s*average loss\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
TESTING_RE  = re.compile(r"^\s*Testing set:\s*(\d+)\s*segs,\s*average loss\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

LABEL_LINE_RE = re.compile(
    r"^\s*\|\s*label\s+(\d+)\s*\|\s*segs\s+(\d+)\s*\|\s*precision\s*([0-9]*\.?[0-9]+)\s*\|\s*recall\s*([0-9]*\.?[0-9]+)\s*\|\s*f1\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE
)
MACRO_LINE_RE = re.compile(
    r"^\s*\|\s*Macro\s*\|\s*precision\s*([0-9]*\.?[0-9]+)\s*\|\s*recall\s*([0-9]*\.?[0-9]+)\s*\|\s*f1\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE
)


def parse_result_txt(path: str) -> Dict[str, Any]:
    """
    Parse a result txt file like:
      Training set: ...
      | label ...
      | Macro| precision ... | recall ... | f1 ...
      Testing set: ...
      ...
    Returns:
      {
        "train": { "segs": int, "loss": float, "macro": {...}, "per_label": {label: {...}} },
        "test":  { ... }
      }
    """
    out: Dict[str, Any] = {
        "train": {"segs": None, "loss": None, "macro": None, "per_label": {}},
        "test":  {"segs": None, "loss": None, "macro": None, "per_label": {}},
    }
    if not path or (not os.path.exists(path)):
        return out

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip().replace("\xa0", " ") for ln in f.readlines()]

    mode = None  # "train" or "test"
    for ln in lines:
        m_tr = TRAINING_RE.match(ln)
        if m_tr:
            mode = "train"
            out["train"]["segs"] = int(m_tr.group(1))
            out["train"]["loss"] = float(m_tr.group(2))
            continue

        m_te = TESTING_RE.match(ln)
        if m_te:
            mode = "test"
            out["test"]["segs"] = int(m_te.group(1))
            out["test"]["loss"] = float(m_te.group(2))
            continue

        if mode is None:
            continue

        m_label = LABEL_LINE_RE.match(ln)
        if m_label:
            label = int(m_label.group(1))
            segs = int(m_label.group(2))
            p = float(m_label.group(3))
            r = float(m_label.group(4))
            f1 = float(m_label.group(5))
            out[mode]["per_label"][label] = {"segs": segs, "precision": p, "recall": r, "f1": f1}
            continue

        m_macro = MACRO_LINE_RE.match(ln)
        if m_macro:
            out[mode]["macro"] = {
                "precision": float(m_macro.group(1)),
                "recall": float(m_macro.group(2)),
                "f1": float(m_macro.group(3)),
            }
            continue

    return out


# -----------------------------
# Parsing: learning_curves.json
# -----------------------------
def load_learning_curves(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or (not os.path.exists(path)):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            curves = json.load(fp)
        if not isinstance(curves, dict):
            return None
        return curves
    except Exception:
        return None


# -----------------------------
# Plotting
# -----------------------------
def plot_macro_bar_compare(
    baseline: Dict[str, Any],
    distill: Dict[str, Any],
    out_path: str,
    title: str = "Macro Metrics Comparison (Deployment-Equivalent)"
):
    """
    A compact bar chart comparing macro precision/recall/f1 for train & test.
    """
    def get_macro(d: Dict[str, Any], split: str) -> Tuple[float, float, float]:
        m = (d.get(split, {}) or {}).get("macro", None)
        if not m:
            return (np.nan, np.nan, np.nan)
        return (m.get("precision", np.nan), m.get("recall", np.nan), m.get("f1", np.nan))

    b_tr = get_macro(baseline, "train")
    b_te = get_macro(baseline, "test")
    d_tr = get_macro(distill, "train")
    d_te = get_macro(distill, "test")

    # layout: 2 rows (train/test), 3 cols (P/R/F1), each has baseline vs distill bars
    metrics = ["precision", "recall", "f1"]
    splits = ["train", "test"]

    data = {
        ("train", "baseline"): b_tr,
        ("train", "distill"):  d_tr,
        ("test",  "baseline"): b_te,
        ("test",  "distill"):  d_te,
    }

    fig = plt.figure(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.35

    # Train subplot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(x - width/2, data[("train", "baseline")], width, label="baseline")
    ax1.bar(x + width/2, data[("train", "distill")],  width, label="distill")
    ax1.set_title("Train")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False)

    # Test subplot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(x - width/2, data[("test", "baseline")], width, label="baseline")
    ax2.bar(x + width/2, data[("test", "distill")],  width, label="distill")
    ax2.set_title("Test")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def plot_per_class_f1_compare(
    baseline: Dict[str, Any],
    distill: Dict[str, Any],
    out_path: str,
    split: str = "test",
    title: str = "Per-Class F1 Comparison (Test)"
):
    b = (baseline.get(split, {}) or {}).get("per_label", {}) or {}
    d = (distill.get(split, {}) or {}).get("per_label", {}) or {}
    labels = sorted(set(list(b.keys()) + list(d.keys())))
    if not labels:
        print("[WARN] No per-label info found for per-class plot.")
        return

    b_f1 = [b.get(k, {}).get("f1", np.nan) for k in labels]
    d_f1 = [d.get(k, {}).get("f1", np.nan) for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, b_f1, width, label="baseline")
    plt.bar(x + width/2, d_f1, width, label="distill")
    plt.xticks(x, [str(k) for k in labels])
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("Class label")
    plt.ylabel("F1")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def plot_learning_curves_compare(
    baseline_curves: Optional[Dict[str, Any]],
    distill_curves: Optional[Dict[str, Any]],
    out_prefix: str,
):
    """
    Plot key curves. If baseline exists, overlay baseline vs distill.
    If baseline is None, plot distill only.
    """
    if not distill_curves:
        print("[WARN] Distill learning_curves.json not found or invalid. Skip curve plots.")
        return

    metric_groups = [
        ("loss", ["train_loss", "test_loss", "train_kd_loss"]),
        ("accuracy_f1", ["train_accuracy", "test_accuracy", "train_f1", "test_f1"]),
        ("pr_re", ["train_precision", "test_precision", "train_recall", "test_recall"]),
        ("diagnostics", ["train_grad_norm", "train_pred_entropy", "test_pred_entropy",
                         "train_pred_margin", "test_pred_margin"]),
    ]

    def _series(curves, k):
        v = curves.get(k, None)
        if isinstance(v, list) and len(v) > 0:
            return v
        return None

    for group_name, keys in metric_groups:
        # gather distill available series
        dist_avail = {k: _series(distill_curves, k) for k in keys}
        dist_avail = {k: v for k, v in dist_avail.items() if v is not None}
        if not dist_avail:
            continue

        # determine x-axis length based on first available distill series
        any_key = next(iter(dist_avail.keys()))
        n = len(dist_avail[any_key])
        x = np.arange(1, n + 1)

        plt.figure(figsize=(10, 4))

        # plot distill
        for k, v in dist_avail.items():
            plt.plot(x, v[:n], label=f"distill:{k}")

        # overlay baseline if provided
        if baseline_curves:
            base_avail = {k: _series(baseline_curves, k) for k in keys}
            base_avail = {k: v for k, v in base_avail.items() if v is not None}
            for k, v in base_avail.items():
                m = min(len(v), n)
                plt.plot(x[:m], v[:m], linestyle="--", label=f"baseline:{k}")

        plt.title(f"Learning Curves - {group_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()

        out_path = f"{out_prefix}-curves-{group_name}.png"
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"[OK] Saved: {out_path}")


# -----------------------------
# Reporting: markdown summary
# -----------------------------
def fmt(x) -> str:
    if x is None:
        return "NA"
    try:
        if np.isnan(x):
            return "NA"
    except Exception:
        pass
    return f"{x:.3f}" if isinstance(x, (float, int, np.floating, np.integer)) else str(x)


def make_summary_md(
    baseline_dir: str,
    distill_dir: str,
    baseline_res_path: Optional[str],
    distill_res_path: Optional[str],
    baseline: Dict[str, Any],
    distill: Dict[str, Any],
    out_path: str,
):
    b_tr = (baseline.get("train", {}) or {}).get("macro", {}) or {}
    b_te = (baseline.get("test", {}) or {}).get("macro", {}) or {}
    d_tr = (distill.get("train", {}) or {}).get("macro", {}) or {}
    d_te = (distill.get("test", {}) or {}).get("macro", {}) or {}

    def get(m, k):
        return m.get(k, np.nan) if isinstance(m, dict) else np.nan

    delta_test_f1 = get(d_te, "f1") - get(b_te, "f1")

    md = []
    md.append("# Evaluation & Analysis Summary\n")
    md.append("## Inputs\n")
    md.append(f"- Baseline dir: `{baseline_dir}`\n")
    md.append(f"- Distill dir: `{distill_dir}`\n")
    md.append(f"- Baseline result txt: `{baseline_res_path or 'NOT FOUND'}`\n")
    md.append(f"- Distill result txt: `{distill_res_path or 'NOT FOUND'}`\n")

    md.append("\n## Macro Metrics (Train/Test)\n")
    md.append("| Split | Model | Precision | Recall | F1 |\n")
    md.append("|---|---|---:|---:|---:|\n")
    md.append(f"| Train | Baseline | {fmt(get(b_tr,'precision'))} | {fmt(get(b_tr,'recall'))} | {fmt(get(b_tr,'f1'))} |\n")
    md.append(f"| Train | Distill  | {fmt(get(d_tr,'precision'))} | {fmt(get(d_tr,'recall'))} | {fmt(get(d_tr,'f1'))} |\n")
    md.append(f"| Test  | Baseline | {fmt(get(b_te,'precision'))} | {fmt(get(b_te,'recall'))} | {fmt(get(b_te,'f1'))} |\n")
    md.append(f"| Test  | Distill  | {fmt(get(d_te,'precision'))} | {fmt(get(d_te,'recall'))} | {fmt(get(d_te,'f1'))} |\n")
    md.append(f"| Test  | Δ (Distill - Baseline) | {fmt(get(d_te,'precision')-get(b_te,'precision'))} | {fmt(get(d_te,'recall')-get(b_te,'recall'))} | **{fmt(delta_test_f1)}** |\n")

    md.append("\n## Plots\n")
    md.append("- `macro-compare.png`: Macro (P/R/F1) comparison for Train & Test\n")
    md.append("- `per-class-f1-test.png`: Per-class F1 comparison on Test set\n")
    md.append("- `*-curves-*.png`: Learning curves & diagnostics (if learning_curves.json exists)\n")

    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("".join(md))
    print(f"[OK] Saved: {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--baseline-dir", type=str, required=True,
                        help="Directory of baseline run (expects brnn-best-result.txt under it).")
    parser.add_argument("--distill-dir", type=str, required=True,
                        help="Directory of distill run (expects student-best-result.txt under it).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for plots & summary (default: distill-dir).")

    # (optional) override filenames
    parser.add_argument("--baseline-result-name", type=str, default="brnn-best-result.txt")
    parser.add_argument("--distill-result-name", type=str, default="student-best-result.txt")
    parser.add_argument("--curves-name", type=str, default="learning_curves.json")

    args = parser.parse_args()

    baseline_dir = os.path.abspath(args.baseline_dir)
    distill_dir = os.path.abspath(args.distill_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else distill_dir
    ensure_dir(out_dir)

    # locate files
    baseline_res_path = find_file_recursive(baseline_dir, args.baseline_result_name)
    distill_res_path = find_file_recursive(distill_dir, args.distill_result_name)

    baseline_curves_path = find_file_recursive(baseline_dir, args.curves_name)
    distill_curves_path = find_file_recursive(distill_dir, args.curves_name)

    if not distill_res_path:
        raise FileNotFoundError(f"Cannot find distill result file '{args.distill_result_name}' under: {distill_dir}")
    if not baseline_res_path:
        raise FileNotFoundError(f"Cannot find baseline result file '{args.baseline_result_name}' under: {baseline_dir}")

    # parse
    baseline = parse_result_txt(baseline_res_path)
    distill = parse_result_txt(distill_res_path)

    baseline_curves = load_learning_curves(baseline_curves_path)
    distill_curves = load_learning_curves(distill_curves_path)

    # plots
    macro_plot_path = os.path.join(out_dir, "macro-compare.png")
    plot_macro_bar_compare(baseline, distill, macro_plot_path)

    percls_plot_path = os.path.join(out_dir, "per-class-f1-test.png")
    plot_per_class_f1_compare(baseline, distill, percls_plot_path, split="test")

    curves_prefix = os.path.join(out_dir, "learning")
    plot_learning_curves_compare(baseline_curves, distill_curves, out_prefix=curves_prefix)

    # summary md
    summary_md_path = os.path.join(out_dir, "evaluation_summary.md")
    make_summary_md(
        baseline_dir=baseline_dir,
        distill_dir=distill_dir,
        baseline_res_path=baseline_res_path,
        distill_res_path=distill_res_path,
        baseline=baseline,
        distill=distill,
        out_path=summary_md_path,
    )

    print("\nDone.")
    print(f"- Output dir: {out_dir}")
    print(f"- Plots: macro-compare.png, per-class-f1-test.png, learning-curves-*.png")
    print(f"- Summary: evaluation_summary.md")


if __name__ == "__main__":
    main()
