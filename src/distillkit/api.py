"""Public API for distillation workflows."""

from __future__ import annotations

import json
import os
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional

from .core.legacy import build_args, dataset_stats_path, repo_root
from .export.p4_export import export_model_state_dict


def dataset_stats(dataset: str) -> Dict:
    stats_path = dataset_stats_path(dataset)
    if not stats_path.exists():
        bundled = pkgutil.get_data("distillkit", f"data/{dataset}/statistics.json")
        if bundled is not None:
            return json.loads(bundled.decode("utf-8"))
        raise FileNotFoundError(
            f"statistics.json not found: {stats_path}. "
            "Set DISTILLKIT_DATASET_ROOT or place dataset files under dataset/<DATASET>/json/."
        )
    with stats_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _dataset_json_path(dataset: str, filename: str) -> str:
    dataset_root = os.getenv("DISTILLKIT_DATASET_ROOT")
    if dataset_root:
        return str(Path(dataset_root) / dataset / "json" / filename)
    return str(repo_root() / "dataset" / dataset / "json" / filename)


def _get_model_class(name: str):
    if name == "BinaryRNN":
        from distillation.model.sturent_BRNN_from_BOS import BinaryRNN

        return BinaryRNN
    if name == "BinaryLSTM":
        from distillation.model.teacher_BLSTM import BinaryLSTM

        return BinaryLSTM
    if name == "BinaryL3LSTM":
        from distillation.model.teacher_BL3LSTM import BinaryL3LSTM

        return BinaryL3LSTM
    if name == "BinaryLSTMWithAttention":
        from distillation.model.teacher_BATLSTM import BinaryLSTMWithAttention

        return BinaryLSTMWithAttention
    if name == "BiLSTMWithAttention":
        from distillation.model.teacher_BBiATLSTM import BiLSTMWithAttention

        return BiLSTMWithAttention
    if name == "BiLSTM2WithAttention":
        from distillation.model.teacher_BBi2ATLSTM import BiLSTM2WithAttention

        return BiLSTM2WithAttention
    raise ValueError(f"Unknown model name: {name}")


def _teacher_ckpt_path(dataset: str, teacher_model: str, teacher_bm_path: Optional[str]) -> Path:
    if teacher_bm_path:
        root = Path(teacher_bm_path)
    elif os.getenv("DISTILLKIT_TEACHER_BM_ROOT"):
        root = Path(os.getenv("DISTILLKIT_TEACHER_BM_ROOT", ""))
    else:
        default_root = repo_root() / "teacher_bm_path"
        legacy_root = repo_root() / "distillation" / "teacher_bm_path"
        root = default_root if default_root.exists() else legacy_root
    return root / dataset / teacher_model / "teacher-brnn-best"


def run_distillation(
    dataset: str,
    teacher_model: str,
    loss_type: str = "KL",
    extra_args: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    student_model: str = "BinaryRNN",
    teacher_bm_path: Optional[str] = None,
) -> str:
    import torch

    if loss_type != "KL":
        raise ValueError("Only KL distillation is supported by distillation pipeline.")
    if student_model != "BinaryRNN":
        raise ValueError("Current distillation pipeline supports student_model='BinaryRNN' only.")

    args = build_args(dataset, teacher_model, "all", extra_args or [])

    args.train_path = _dataset_json_path(dataset, "train.json")
    args.test_path = _dataset_json_path(dataset, "test.json")

    stats = dataset_stats(dataset)
    args.num_classes = stats["label_num"]
    args.class_weights = [1] * args.num_classes

    if output_dir:
        args.output_dir = str(Path(output_dir))
    else:
        args.output_dir = (
            f"./save_kd/{args.dataset}/T_{args.teacher_model}_/S_BRNN/"
            f"len{args.pkt_len_embed_bits}_ipd{args.ipd_embed_bits}_"
            f"ev{args.embed_dim_bits}_hidden{args.rnn_hidden_state_bits}_/"
            f"kd_a{args.kd_alpha}_t{args.kd_temperature}_lr{args.lr}/"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    from distillation.trainers.student_trainer import DistillTrainer
    from distillation.trainers.teacher_trainer import build_optimizer
    from distillation.utils.checkpoint import initialize_parameters, load_model
    from distillation.utils.data_loader import build_data_loader
    from distillation.utils.seed import build_generator, seed_worker, set_seed

    set_seed(args.random_seed)
    dl_generator = build_generator(args.random_seed)

    args.teacher_ckpt_path = str(_teacher_ckpt_path(args.dataset, args.teacher_model, teacher_bm_path))
    if not Path(args.teacher_ckpt_path).exists():
        raise FileNotFoundError(
            "Teacher checkpoint not found: "
            f"{args.teacher_ckpt_path}. "
            "Use --teacher-bm-path or set DISTILLKIT_TEACHER_BM_ROOT."
        )

    TeacherClass = _get_model_class(args.teacher_model)
    StudentClass = _get_model_class("BinaryRNN")

    teacher = TeacherClass(args)
    student = StudentClass(args)

    load_model(teacher, args.teacher_ckpt_path)
    teacher.eval()

    initialize_parameters(args, student)

    if args.cuda_device_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device_id)
        teacher.cuda(args.cuda_device_id)
        student.cuda(args.cuda_device_id)
    else:
        args.cuda_device_id = None

    optimizer, _ = build_optimizer(args, student)

    train_loader = build_data_loader(
        args,
        args.train_path,
        args.train_batch_size,
        is_train=True,
        shuffle=True,
        generator=dl_generator,
        worker_init_fn=seed_worker,
    )
    test_loader = build_data_loader(
        args,
        args.test_path,
        args.train_batch_size,
        is_train=False,
        shuffle=False,
        generator=dl_generator,
        worker_init_fn=seed_worker,
    )

    trainer = DistillTrainer(args)
    trainer.train(args, train_loader, test_loader, student, teacher, optimizer)

    return args.output_dir


def evaluate_model(
    dataset: str,
    model_path: str,
    extra_args: Optional[List[str]] = None,
    student_model: str = "BinaryRNN",
) -> Dict:
    import numpy as np
    import torch

    args = build_args(dataset, teacher_model="BinaryLSTM", loss_scope="all", extra_args=extra_args or [])

    args.train_path = _dataset_json_path(dataset, "train.json")
    args.test_path = _dataset_json_path(dataset, "test.json")
    args.output_dir = str(Path(model_path).resolve().parent)

    stats = dataset_stats(dataset)
    args.num_classes = stats["label_num"]
    args.class_weights = [1] * args.num_classes

    from distillation.trainers.student_trainer import DistillTrainer
    from distillation.utils.checkpoint import load_model
    from distillation.utils.classification_metrics import metric_from_confuse_matrix
    from distillation.utils.data_loader import build_data_loader

    StudentClass = _get_model_class(student_model)
    model = StudentClass(args)
    load_model(model, model_path)

    if args.cuda_device_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device_id)
        model.cuda(args.cuda_device_id)
    else:
        args.cuda_device_id = None

    test_loader = build_data_loader(args, args.test_path, args.train_batch_size, is_train=False, shuffle=False)

    trainer = DistillTrainer(args)
    conf_mat, test_total_loss, test_samples, _ = trainer.validate(args, test_loader, model)

    pres, recs, f1s, logs = metric_from_confuse_matrix(conf_mat)

    return {
        "test_samples": int(test_samples),
        "test_loss": float(test_total_loss / max(test_samples, 1)),
        "precision_macro": float(np.mean(pres)),
        "recall_macro": float(np.mean(recs)),
        "f1_macro": float(np.mean(f1s)),
        "details": logs,
    }


def export_p4(
    dataset: str,
    model_path: str,
    out_path: str,
    extra_args: Optional[List[str]] = None,
    student_model: str = "BinaryRNN",
    quantize: str = "int8",
    scale: Optional[float] = None,
) -> str:
    args = build_args(dataset, teacher_model="BinaryLSTM", loss_scope="all", extra_args=extra_args or [])
    args.num_classes = dataset_stats(dataset)["label_num"]

    StudentClass = _get_model_class(student_model)
    model = StudentClass(args)

    from distillation.utils.checkpoint import load_model

    load_model(model, model_path)

    export_model_state_dict(
        model.state_dict(),
        out_path=out_path,
        quantize=quantize,
        scale=scale,
        metadata={
            "dataset": dataset,
            "student_model": student_model,
            "quantize": quantize,
        },
    )

    return out_path
