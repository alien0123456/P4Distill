"""Helpers to bridge distillkit wrappers and distillation package."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List


def repo_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return current.parents[3]


def build_args(
    dataset: str,
    teacher_model: str,
    loss_scope: str,
    extra_args: List[str] | None = None,
) -> argparse.Namespace:
    if dataset != "ISCXVPN2016":
        raise ValueError(f"Unsupported dataset: {dataset}")

    from distillation.opts import distill_opts, model_opts, training_opts

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset",
        default=dataset,
        choices=["ISCXVPN2016"],
    )
    parser.add_argument(
        "--teacher_model",
        default=teacher_model,
        choices=[
            "BinaryLSTM",
            "BinaryLSTMWithAttention",
            "BinaryL3LSTM",
            "BiLSTMWithAttention",
            "BiLSTM2WithAttention",
        ],
    )
    parser.add_argument("--loss_scope", default=loss_scope, choices=["single", "all"])

    model_opts(parser)
    training_opts(parser)
    distill_opts(parser, require_teacher_ckpt=False)

    args = parser.parse_args(extra_args or [])
    return args


def dataset_stats_path(dataset: str) -> Path:
    dataset_root = os.getenv("DISTILLKIT_DATASET_ROOT")
    if dataset_root:
        return Path(dataset_root) / dataset / "json" / "statistics.json"
    return repo_root() / "dataset" / dataset / "json" / "statistics.json"
