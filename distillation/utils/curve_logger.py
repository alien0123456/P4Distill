import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from distillation.utils.classification_metrics import metric_from_confuse_matrix


def update_conf_mat(conf_mat: np.ndarray, labels: torch.Tensor, logits: torch.Tensor):
    pred = logits.argmax(dim=1)
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    pred_np = pred.detach().cpu().numpy().astype(np.int64)
    for li, pi in zip(labels_np, pred_np):
        conf_mat[li, pi] += 1.0


def _mean_safe(total, count):
    return float(total / count) if count > 0 else float("nan")


def _to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _macro_metrics(conf_mat: np.ndarray):
    p, r, f1, _ = metric_from_confuse_matrix(conf_mat)
    return float(np.mean(p)), float(np.mean(r)), float(np.mean(f1))


def _accuracy(conf_mat: np.ndarray):
    total = float(conf_mat.sum())
    return float(np.trace(conf_mat) / total) if total > 0 else float("nan")


def _pred_stats(logits: torch.Tensor):
    probs = F.softmax(logits, dim=1).clamp(min=1e-12)
    entropy = -(probs * probs.log()).sum(dim=1).mean().item()
    top2 = torch.topk(probs, k=2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).mean().item()
    return float(entropy), float(margin)


class CurveLogger:
    """
    Simple curves logger for both teacher and student trainers.
    """

    KEYS = [
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "train_precision",
        "test_precision",
        "train_recall",
        "test_recall",
        "train_f1",
        "test_f1",
        "train_grad_norm",
        "train_pred_entropy",
        "test_pred_entropy",
        "train_pred_margin",
        "test_pred_margin",
        "train_kd_loss",
        "kd_alpha",
    ]

    def __init__(self, labels_num: int):
        self.labels_num = int(labels_num)
        self.data = {k: [] for k in self.KEYS}
        self._reset_epoch()

    def _reset_epoch(self):
        self.train_samples = 0
        self.train_total_loss = 0.0
        self.train_total_hard = 0.0
        self.train_total_kd = 0.0
        self.train_entropy_sum = 0.0
        self.train_margin_sum = 0.0
        self.grad_norm_sum = 0.0
        self.grad_steps = 0
        self.kd_alpha_epoch = float("nan")

    def start_epoch(self, kd_alpha: float):
        self._reset_epoch()
        self.kd_alpha_epoch = float(kd_alpha)

    def update_batch_losses(self, bs: int, loss, hard_loss=None, kd_loss=None):
        bs = int(bs)
        self.train_samples += bs
        self.train_total_loss += _to_float(loss) * bs
        if hard_loss is not None:
            self.train_total_hard += _to_float(hard_loss) * bs
        if kd_loss is not None:
            self.train_total_kd += _to_float(kd_loss) * bs

    def update_batch_preds(self, logits: torch.Tensor, labels: torch.Tensor, bs: int):
        bs = int(bs)
        entropy, margin = _pred_stats(logits)
        self.train_entropy_sum += entropy * bs
        self.train_margin_sum += margin * bs

    def update_batch_hidden(self, hidden_seq, prebin_seq, bs: int, sat_tau: float = 1.0):
        # Keep interface stable; hidden-related metrics are intentionally omitted.
        return

    def update_batch_grads(self, flat_grads: torch.Tensor):
        if flat_grads is None:
            return
        gnorm = torch.norm(flat_grads.float(), p=2).detach().cpu().item()
        self.grad_norm_sum += float(gnorm)
        self.grad_steps += 1

    def update_kd_grad_decomp(self, hard_grads: torch.Tensor, kd_grads: torch.Tensor, kd_alpha: float):
        # Keep interface stable; gradient decomposition metrics are intentionally omitted.
        return

    def finish_epoch_and_dump(
        self,
        output_dir: str,
        conf_mat_train: np.ndarray,
        conf_mat_test: np.ndarray,
        test_total_loss: float,
        test_samples: int,
        extra_test: dict,
    ):
        train_loss = _mean_safe(self.train_total_loss, self.train_samples)
        test_loss = _mean_safe(float(test_total_loss), int(test_samples))

        tr_p, tr_r, tr_f1 = _macro_metrics(conf_mat_train)
        te_p, te_r, te_f1 = _macro_metrics(conf_mat_test)
        tr_acc = _accuracy(conf_mat_train)
        te_acc = _accuracy(conf_mat_test)

        train_grad_norm = _mean_safe(self.grad_norm_sum, self.grad_steps)
        train_pred_entropy = _mean_safe(self.train_entropy_sum, self.train_samples)
        train_pred_margin = _mean_safe(self.train_margin_sum, self.train_samples)
        test_pred_entropy = float(extra_test.get("test_pred_entropy", float("nan")))
        test_pred_margin = float(extra_test.get("test_pred_margin", float("nan")))

        if self.train_samples > 0 and self.train_total_kd > 0:
            train_kd_loss = self.train_total_kd / self.train_samples
        else:
            train_kd_loss = float("nan")

        values = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": tr_acc,
            "test_accuracy": te_acc,
            "train_precision": tr_p,
            "test_precision": te_p,
            "train_recall": tr_r,
            "test_recall": te_r,
            "train_f1": tr_f1,
            "test_f1": te_f1,
            "train_grad_norm": train_grad_norm,
            "train_pred_entropy": train_pred_entropy,
            "test_pred_entropy": test_pred_entropy,
            "train_pred_margin": train_pred_margin,
            "test_pred_margin": test_pred_margin,
            "train_kd_loss": train_kd_loss,
            "kd_alpha": self.kd_alpha_epoch,
        }

        for key in self.KEYS:
            self.data[key].append(float(values.get(key, float("nan"))))

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "learning_curves.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.data, fp, indent=1)
        return path
