"""Helpers to reuse existing distillation scripts safely."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def distillation_dir() -> Path:
    return repo_root() / "distillation"


def ensure_distillation_on_path() -> Path:
    ddir = distillation_dir()
    if str(ddir) not in sys.path:
        sys.path.insert(0, str(ddir))
    return ddir


def _opts_module_for_dataset(dataset: str):
    name_map = {
        "ISCXVPN2016": "opts_iscx2016",
        "BOTIOT": "opts_botiot",
        "CICIOT2022": "opts_ciciot",
    }
    if dataset not in name_map:
        raise ValueError(f"Unsupported dataset: {dataset}")
    module_name = name_map[dataset]
    return importlib.import_module(module_name)


def build_args(
    dataset: str,
    teacher_model: str,
    loss_type: str,
    extra_args: List[str] | None = None,
) -> argparse.Namespace:
    ensure_distillation_on_path()
    opts_mod = _opts_module_for_dataset(dataset)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset",
        default=dataset,
        choices=["ISCXVPN2016", "BOTIOT", "CICIOT2022", "PeerRush"],
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
    parser.add_argument("--loss_type", default=loss_type, choices=["KL"])

    opts_mod.model_opts(parser)
    opts_mod.training_opts(parser)

    args = parser.parse_args(extra_args or [])
    return args


def dataset_stats_path(dataset: str) -> Path:
    return repo_root() / "dataset" / dataset / "json" / "statistics.json"
