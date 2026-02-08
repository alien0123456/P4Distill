"""Command-line interface for distillkit."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from . import __version__
from .api import dataset_stats, evaluate_model, export_p4, run_distillation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distillkit")
    parser.add_argument("--version", action="version", version=__version__)

    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="Show dataset statistics.")
    p_stats.add_argument("--dataset", default="ISCXVPN2016", choices=["ISCXVPN2016"])
    p_stats.add_argument("--out", default="", help="Optional JSON output path.")

    p_distill = sub.add_parser("distill", help="Run knowledge distillation training.")
    p_distill.add_argument("--dataset", default="ISCXVPN2016", choices=["ISCXVPN2016"])
    p_distill.add_argument(
        "--teacher-model",
        required=True,
        choices=[
            "BinaryLSTM",
            "BinaryLSTMWithAttention",
            "BinaryL3LSTM",
            "BiLSTMWithAttention",
            "BiLSTM2WithAttention",
        ],
    )
    p_distill.add_argument("--loss-type", default="KL", choices=["KL"])
    p_distill.add_argument("--output-dir", default="", help="Override output directory.")
    p_distill.add_argument(
        "--student-model",
        default="BinaryRNN",
        choices=["BinaryRNN"],
        help="Distillation pipeline currently supports BinaryRNN only.",
    )
    p_distill.add_argument(
        "--teacher-bm-path",
        default="",
        help=(
            "Root directory containing teacher checkpoints (e.g. "
            "/root/kaiyuan2/teacher_bm_path)."
        ),
    )

    p_eval = sub.add_parser("eval", help="Evaluate a saved student model.")
    p_eval.add_argument("--dataset", default="ISCXVPN2016", choices=["ISCXVPN2016"])
    p_eval.add_argument("--model-path", required=True)
    p_eval.add_argument(
        "--student-model",
        default="BinaryRNN",
        choices=[
            "BinaryRNN",
            "BinaryLSTM",
            "BinaryLSTMWithAttention",
            "BinaryL3LSTM",
            "BiLSTMWithAttention",
            "BiLSTM2WithAttention",
        ],
    )
    p_eval.add_argument("--out", default="", help="Optional JSON output path.")

    p_export = sub.add_parser("export", help="Export model for P4 integration.")
    p_export.add_argument("--dataset", default="ISCXVPN2016", choices=["ISCXVPN2016"])
    p_export.add_argument("--model-path", required=True)
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--quantize", default="int8", choices=["none", "int8", "int16"])
    p_export.add_argument("--scale", type=float, default=None)
    p_export.add_argument(
        "--student-model",
        default="BinaryRNN",
        choices=[
            "BinaryRNN",
            "BinaryLSTM",
            "BinaryLSTMWithAttention",
            "BinaryL3LSTM",
            "BiLSTMWithAttention",
            "BiLSTM2WithAttention",
        ],
    )

    p_wizard = sub.add_parser("wizard", help="Interactive setup for distillation.")

    return parser


def _split_extra_args(argv: List[str]) -> List[str]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[idx + 1 :]
    return []


def _handle_wizard() -> None:
    print("DistillKit Wizard")
    dataset = input("Dataset (ISCXVPN2016): ").strip() or "ISCXVPN2016"
    teacher = input("Teacher model (BinaryLSTM/BinaryLSTMWithAttention/...): ").strip() or "BinaryLSTM"
    loss_type = input("Loss type (KL): ").strip() or "KL"
    output_dir = input("Output dir (blank for default): ").strip()
    teacher_bm_path = input("Teacher checkpoint root (blank for default): ").strip()
    run_now = input("Run now? (y/N): ").strip().lower() == "y"

    if run_now:
        out = run_distillation(
            dataset=dataset,
            teacher_model=teacher,
            loss_type=loss_type,
            output_dir=output_dir or None,
            teacher_bm_path=teacher_bm_path or None,
        )
        print(f"Training finished. Output: {out}")
    else:
        cmd = f"distillkit distill --dataset {dataset} --teacher-model {teacher} --loss-type {loss_type}"
        if output_dir:
            cmd += f" --output-dir {output_dir}"
        if teacher_bm_path:
            cmd += f" --teacher-bm-path {teacher_bm_path}"
        print("Run this command:")
        print(cmd)


def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    extra_args = _split_extra_args(argv) or unknown

    if args.command == "stats":
        stats = dataset_stats(args.dataset)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fp:
                json.dump(stats, fp, indent=2)
        print(json.dumps(stats, indent=2))
        return 0

    if args.command == "distill":
        out = run_distillation(
            dataset=args.dataset,
            teacher_model=args.teacher_model,
            loss_type=args.loss_type,
            extra_args=extra_args,
            output_dir=args.output_dir or None,
            student_model=args.student_model,
            teacher_bm_path=args.teacher_bm_path or None,
        )
        print(f"Output: {out}")
        return 0

    if args.command == "eval":
        report = evaluate_model(
            dataset=args.dataset,
            model_path=args.model_path,
            extra_args=extra_args,
            student_model=args.student_model,
        )
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fp:
                json.dump(report, fp, indent=2)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "export":
        out = export_p4(
            dataset=args.dataset,
            model_path=args.model_path,
            out_path=args.out,
            extra_args=extra_args,
            student_model=args.student_model,
            quantize=args.quantize,
            scale=args.scale,
        )
        print(f"Exported: {out}")
        return 0

    if args.command == "wizard":
        _handle_wizard()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
