import time
import torch
import numpy as np
import torch.nn.functional as F
import os

from distillation.utils.classification_metrics import metric_from_confuse_matrix
from distillation.utils.checkpoint import  save_checkpoint
from distillation.utils.curve_logger import CurveLogger, update_conf_mat


class DistillTrainer(object):
    def __init__(self, args):
        self.current_epoch = 0
        self.total_epochs = args.max_epochs
        self.save_checkpoint_epochs = args.ckpt_save_interval_epochs

        self.labels_num = args.num_classes
        self.output_dir = args.output_dir

        self.loss_factor = args.ce_loss_weight
        self.focal_loss_gamma = args.focal_gamma
        self.loss_type = args.loss_scope

        self.base_kd_alpha = args.kd_alpha
        self.kd_alpha = args.kd_alpha
        self.kd_temperature = args.kd_temperature

        # ---- Optional schedules (may NOT exist in vanilla distil.py) ----
        # warmup
        self.kd_warmup_epochs = int(getattr(args, "kd_warmup_epochs", 0) or 0)     # kd_warmup_epochs=0锛氳烦杩?warmup 閫昏緫锛岀洿鎺ヤ娇鐢?base_kd_alpha 浣滀负 KD loss 鐨勬潈閲嶃€?

        # off-after (late no-KD)
        self.kd_off_after_epochs = int(getattr(args, "kd_off_after_epochs", 0) or 0)

    def _flatten_grads(self, model):
        flats = []
        for p in model.parameters():
            if p.grad is None:
                continue
            flats.append(p.grad.detach().reshape(-1))
        if len(flats) == 0:
            return None
        return torch.cat(flats, dim=0)

    def _hard_loss(self, logits, label_batch):
        eps = 1e-8
        softmax = F.softmax(logits, dim=1).clamp(min=eps, max=1 - eps)
        one_hot = F.one_hot(label_batch, num_classes=self.labels_num)

        p_y = softmax[one_hot == 1]
        loss_y = -(1 - p_y) ** self.focal_loss_gamma * torch.log(p_y)

        if self.loss_type == 'single':
            remove_p = one_hot.float()
            remove_p[remove_p == 1] = -torch.inf
            max_without_p, _ = (softmax + remove_p).max(dim=1, keepdim=True)
            max_without_p = torch.squeeze(max_without_p)
            one_minus = (1 - max_without_p).clamp(min=eps)
            loss_others = -max_without_p ** self.focal_loss_gamma * torch.log(one_minus)
        else:
            p_others = softmax[one_hot == 0].reshape(shape=(len(softmax), self.labels_num - 1))
            one_minus = (1 - p_others).clamp(min=eps)  # 鍏抽敭锛氶伩鍏?log(0)
            loss_others = -p_others ** self.focal_loss_gamma * torch.log(one_minus)
            loss_others = torch.sum(loss_others, dim=1)

        loss_1 = torch.sum(loss_y) / len(softmax)
        loss_2 = torch.sum(loss_others) / len(softmax)
        loss = loss_1 + self.loss_factor * loss_2
        return loss, loss_1, loss_2

    def _kd_loss(self, student_logits, teacher_logits):
        t = float(self.kd_temperature)
        s_log_probs = F.log_softmax(student_logits / t, dim=1)
        t_probs = F.softmax(teacher_logits / t, dim=1)
        return F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (t * t)

    def forward_propagation(self, len_x_batch, ipd_x_batch, label_batch, student, teacher):
        student_logits = student(len_x_batch, ipd_x_batch)
        with torch.no_grad():
            teacher_logits = teacher(len_x_batch, ipd_x_batch)

        hard_loss, loss_1, loss_2 = self._hard_loss(student_logits, label_batch)
        kd_loss = self._kd_loss(student_logits, teacher_logits)

        loss = self.kd_alpha * kd_loss + (1 - self.kd_alpha) * hard_loss
        return loss, student_logits, hard_loss, kd_loss, loss_1, loss_2

    def validate(self, args, test_loader, student):
        student.eval()

        test_samples = 0
        test_total_loss = 0.0
        conf_mat_test = np.zeros([args.num_classes, args.num_classes], dtype=np.float64)

        test_entropy_sum = 0.0
        test_margin_sum = 0.0
        test_maxprob_sum = 0.0

        with torch.no_grad():
            for len_x_batch, ipd_x_batch, label_batch in test_loader:
                logits = student(len_x_batch, ipd_x_batch)
                hard_loss, _, _ = self._hard_loss(logits, label_batch)

                bs = int(len_x_batch.shape[0])
                test_samples += bs
                test_total_loss += float(hard_loss.item()) * bs

                update_conf_mat(conf_mat_test, label_batch, logits)

                eps = 1e-12
                probs = F.softmax(logits, dim=1).clamp(min=eps)
                entropy = -(probs * probs.log()).sum(dim=1).mean().item()
                top2 = torch.topk(probs, k=2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).mean().item()
                max_prob = probs.max(dim=1).values.mean().item()

                test_entropy_sum += float(entropy) * bs
                test_margin_sum += float(margin) * bs
                test_maxprob_sum += float(max_prob) * bs

        extra = {
            "test_pred_entropy": (test_entropy_sum / test_samples) if test_samples > 0 else float("nan"),
            "test_pred_margin": (test_margin_sum / test_samples) if test_samples > 0 else float("nan"),
            "test_pred_max_prob": (test_maxprob_sum / test_samples) if test_samples > 0 else float("nan"),
        }
        return conf_mat_test, test_total_loss, test_samples, extra

    def _apply_kd_schedule(self):
        """Update self.kd_alpha in-place according to warmup/off schedule."""
        # warmup
        if self.kd_warmup_epochs and self.kd_warmup_epochs > 0:
            if self.current_epoch <= self.kd_warmup_epochs:
                self.kd_alpha = 0.0
        else:
            self.kd_alpha = self.kd_alpha

        # off after epochs
        if self.kd_off_after_epochs and self.kd_off_after_epochs > 0:
            if self.current_epoch > self.kd_off_after_epochs:
                self.kd_alpha = 0.0

    def train(self, args, train_loader, test_loader, student, teacher, optimizer):
        logger = CurveLogger(labels_num=args.num_classes)

        best_test_f1 = -1e18
        best_epoch = -1

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        for epoch in range(1, self.total_epochs + 1):
            self.current_epoch = epoch

            # 1) schedule FIRST
            self._apply_kd_schedule()

            # 2) logger epoch init (use the *current* kd_alpha)
            logger.start_epoch(kd_alpha=self.kd_alpha)

            start_time = time.time()
            student.train()

            conf_mat_train = np.zeros([args.num_classes, args.num_classes], dtype=np.float64)

            # ---------------- train loop ----------------
            for len_x_batch, ipd_x_batch, label_batch in train_loader:

                loss, logits, hard_loss, kd_loss, _, _ = self.forward_propagation(
                    len_x_batch, ipd_x_batch, label_batch, student, teacher
                )

                student.zero_grad(set_to_none=True)
                loss.backward()
                flat_grads = self._flatten_grads(student)
                logger.update_batch_grads(flat_grads)

                optimizer.step()
                student.zero_grad(set_to_none=True)

                bs = int(len_x_batch.shape[0])

                # record losses / preds / hidden
                logger.update_batch_losses(bs, loss=loss, hard_loss=hard_loss, kd_loss=kd_loss)
                logger.update_batch_preds(logits, label_batch, bs)

                # confusion matrix
                update_conf_mat(conf_mat_train, label_batch, logits)

            # ---------------- validate ----------------
            conf_mat_test, test_total_loss, test_samples, extra_test = self.validate(args, test_loader, student)

            # print summary (use logger internal accumulators)
            train_samples = logger.train_samples
            train_avg_loss = logger.train_total_loss / train_samples if train_samples > 0 else float("nan")
            train_avg_hard = logger.train_total_hard / train_samples if train_samples > 0 else float("nan")
            train_avg_kd = logger.train_total_kd / train_samples if train_samples > 0 else float("nan")
            test_avg_loss = (test_total_loss / test_samples) if test_samples > 0 else float("nan")

            print(f"delta(train_loss-hard)={train_avg_loss - train_avg_hard:+.6f}, kd_alpha={self.kd_alpha:.3f}")
            print("| {:5d}/{:5d} epochs ({:5.2f} s, lr {:8.5f})"
                  "| Train segs {:7d}, Test segs {:7d} "
                  "| Train loss {:7.2f} hard {:7.2f} kd {:7.2f}"
                  "| Test loss {:7.2f}".format(
                      self.current_epoch, self.total_epochs, time.time() - start_time, optimizer.param_groups[0]['lr'],
                      train_samples, test_samples,
                      train_avg_loss, train_avg_hard, train_avg_kd,
                      test_avg_loss
                  ))

            # compute test_f1 for best checkpoint (same鍙ｅ緞 as浠ュ墠)
            pres_test, recs_test, f1s_test, _ = metric_from_confuse_matrix(conf_mat_test)
            test_f1 = float(np.mean(f1s_test))

            is_best = test_f1 > (best_test_f1 + 1e-12)
            if is_best:
                best_test_f1 = test_f1
                best_epoch = self.current_epoch

            # checkpoint save (keep your original logs)
            if is_best or (self.current_epoch % self.save_checkpoint_epochs == 0):
                pres_train, recs_train, f1s_train, logs_train = metric_from_confuse_matrix(conf_mat_train)
                pres_test, recs_test, f1s_test, logs_test = metric_from_confuse_matrix(conf_mat_test)

                logs = [f'Training set: {train_samples} segs, average loss {train_avg_loss}']
                logs.extend(logs_train)
                logs.append(f'Testing set: {test_samples} segs, average loss {test_avg_loss}')
                logs.extend(logs_test)

                if is_best:
                    save_checkpoint(
                        output_dir=self.output_dir,
                        model_name='brnn-best',
                        model=student,
                        result_log=logs
                    )
                if self.current_epoch % self.save_checkpoint_epochs == 0:
                    save_checkpoint(
                        output_dir=self.output_dir,
                        model_name='brnn-' + str(self.current_epoch),
                        model=student,
                        result_log=logs
                    )

            # finally: dump learning_curves.json (single source of truth)
            out_path = logger.finish_epoch_and_dump(
                output_dir=self.output_dir,
                conf_mat_train=conf_mat_train,
                conf_mat_test=conf_mat_test,
                test_total_loss=test_total_loss,
                test_samples=test_samples,
                extra_test=extra_test,
            )
            # print(f"[curves] updated: {out_path}")

        print(f"[done] best_epoch={best_epoch}, best_test_f1={best_test_f1:.6f}")

