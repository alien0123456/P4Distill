"""Public API for distillation workflows."""

from __future__ import annotations

import json
import os
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional

from .core.legacy import build_args, dataset_stats_path, ensure_distillation_on_path, repo_root
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
    ensure_distillation_on_path()
    if name == "BinaryRNN":
        from model.model_BRNN import BinaryRNN

        return BinaryRNN
    if name == "BinaryLSTM":
        from model.Binary.model_BLSTM import BinaryLSTM

        return BinaryLSTM
    if name == "BinaryL3LSTM":
        from model.Binary.model_BL3LSTM import BinaryL3LSTM

        return BinaryL3LSTM
    if name == "BinaryLSTMWithAttention":
        from model.Binary.model_BATLSTM import BinaryLSTMWithAttention

        return BinaryLSTMWithAttention
    if name == "BiLSTMWithAttention":
        from model.Binary.model_BBiATLSTM import BiLSTMWithAttention

        return BiLSTMWithAttention
    if name == "BiLSTM2WithAttention":
        from model.Binary.model_BBi2ATLSTM import BiLSTM2WithAttention

        return BiLSTM2WithAttention
    raise ValueError(f"Unknown model name: {name}")


def _compute_class_weights(args, stats):
    import numpy as np
    import torch
    from sklearn.utils.class_weight import compute_class_weight

    labels_num = stats.get("label_num", args.labels_num)
    flow_split = stats.get("train num / test num (flow)")
    if args.dataset == "BOTIOT" and flow_split:
        class_names = ["DataExfiltration", "Keylogging", "OSScan", "ServiceScan"]
        if all(name in flow_split for name in class_names):
            train_counts = []
            for cls_name in class_names:
                train_num_str, _ = flow_split[cls_name].split(" / ")
                train_counts.append(int(train_num_str))
            y_train = []
            for class_id, count in enumerate(train_counts):
                y_train.extend([class_id] * count)
            y_train = np.array(y_train)
            class_weights = compute_class_weight(
                "balanced",
                classes=np.arange(labels_num),
                y=y_train,
            ).astype(np.float32)
            return torch.tensor(class_weights, dtype=torch.float)

    return torch.ones(labels_num, dtype=torch.float)


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

    ensure_distillation_on_path()
    args = build_args(dataset, teacher_model, loss_type, extra_args or [])

    args.train_path = _dataset_json_path(dataset, "train.json")
    args.test_path = _dataset_json_path(dataset, "test.json")

    stats = dataset_stats(dataset)
    args.labels_num = stats["label_num"]
    args.class_weights = _compute_class_weights(args, stats)

    if output_dir:
        args.output_dir = output_dir
    else:
        args.output_dir = (
            f"./save/{args.dataset}/{args.teacher_model}/{args.loss_type}/"
            f"studentbrnn_len{args.len_embedding_bits}_ipd{args.ipd_embedding_bits}_"
            f"ev{args.embedding_vector_bits}_hidden{args.rnn_hidden_bits}_"
            f"{args.loss_factor}_{args.focal_loss_gamma}_{args.loss_type}_{args.learning_rate}_"
            f"T{args.T}_a{args.a}/"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    from utils.seed import set_seed
    from utils.model_rwi import initialize_parameters, load_model
    from trainer2 import build_optimizer, build_data_loader, STrainer

    set_seed(args.seed)

    teacher_root = (
        teacher_bm_path
        or os.getenv("DISTILLKIT_TEACHER_BM_ROOT")
        or str(repo_root() / "distillation" / "teacher_bm_path")
    )
    args.teacher_bestmodel_path = str(
        Path(teacher_root) / args.dataset / args.teacher_model / "teacher-brnn-best"
    )
    if not Path(args.teacher_bestmodel_path).exists():
        raise FileNotFoundError(
            "Teacher checkpoint not found: "
            f"{args.teacher_bestmodel_path}. "
            "Use --teacher-bm-path or set DISTILLKIT_TEACHER_BM_ROOT."
        )

    TeacherClass = _get_model_class(args.teacher_model)
    StudentClass = _get_model_class(student_model)

    teacher = TeacherClass(args)
    student = StudentClass(args)

    load_model(teacher, args.teacher_bestmodel_path)
    teacher.eval()

    initialize_parameters(args, student)

    if args.gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        teacher.cuda(args.gpu_id)
        student.cuda(args.gpu_id)
    else:
        args.gpu_id = None

    optimizer, scheduler = build_optimizer(args, student)

    train_loader = build_data_loader(args, args.train_path, args.batch_size, is_train=True, shuffle=True)
    test_loader = build_data_loader(args, args.test_path, args.batch_size, is_train=False, shuffle=False)

    trainer = STrainer(args)
    trainer.Strain(args, train_loader, test_loader, teacher, student, optimizer, scheduler)

    return args.output_dir


def evaluate_model(
    dataset: str,
    model_path: str,
    extra_args: Optional[List[str]] = None,
    student_model: str = "BinaryRNN",
) -> Dict:
    import numpy as np
    import torch

    ensure_distillation_on_path()
    args = build_args(dataset, teacher_model="BinaryLSTM", loss_type="KL", extra_args=extra_args or [])

    args.train_path = _dataset_json_path(dataset, "train.json")
    args.test_path = _dataset_json_path(dataset, "test.json")
    # `STrainer` expects `args.output_dir`; for eval use model directory as default.
    args.output_dir = str(Path(model_path).resolve().parent)

    stats = dataset_stats(dataset)
    args.labels_num = stats["label_num"]

    from utils.model_rwi import load_model
    from trainer2 import build_data_loader, STrainer
    from utils.metric import metric_from_confuse_matrix

    StudentClass = _get_model_class(student_model)
    model = StudentClass(args)
    load_model(model, model_path)

    if args.gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        model.cuda(args.gpu_id)
    else:
        args.gpu_id = None

    test_loader = build_data_loader(args, args.test_path, args.batch_size, is_train=False, shuffle=False)

    trainer = STrainer(args)
    conf_mat, test_total_loss, test_samples = trainer.Svalidate(args, test_loader, model)

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
    ensure_distillation_on_path()
    args = build_args(dataset, teacher_model="BinaryLSTM", loss_type="KL", extra_args=extra_args or [])

    StudentClass = _get_model_class(student_model)
    model = StudentClass(args)

    from utils.model_rwi import load_model

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
