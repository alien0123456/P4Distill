# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import torch.nn.functional as F

import torch.optim as optim

from distillation.utils.checkpoint import *   # save_model, etc.
from distillation.utils.data_loader import FlowDataset
from distillation.utils.classification_metrics import metric_from_confuse_matrix



from distillation.utils.curve_logger import CurveLogger, update_conf_mat


def build_optimizer(args, model):
    if args.optim_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return optimizer, None


class Teacher_Trainer(object):
    def __init__(self, args):
        self.current_epoch = 0
        self.total_epochs = args.max_epochs
        self.save_checkpoint_epochs = args.ckpt_save_interval_epochs

        self.labels_num = args.num_classes
        self.output_dir = args.output_dir

        self.loss_factor = args.ce_loss_weight
        self.focal_loss_gamma = args.focal_gamma
        self.loss_type = args.loss_scope

    def _flatten_grads(self, model):
        flats = []
        for p in model.parameters():
            if p.grad is None:
                continue
            flats.append(p.grad.detach().reshape(-1))
        if len(flats) == 0:
            return None
        return torch.cat(flats, dim=0)

    def forward_propagation(self, len_x_batch, ipd_x_batch, label_batch, model):
        logits = model(len_x_batch, ipd_x_batch)

        eps = 1e-8
        softmax = F.softmax(logits, dim=1).clamp(min=eps, max=1 - eps)
        one_hot = F.one_hot(label_batch, num_classes=self.labels_num)

        p_y = softmax[one_hot == 1]
        loss_y = - (1 - p_y) ** self.focal_loss_gamma * torch.log(p_y)

        if self.loss_type == 'single':
            remove_p = one_hot.float()
            remove_p[remove_p == 1] = -torch.inf
            max_without_p, _ = (softmax + remove_p).max(dim=1, keepdim=True)
            max_without_p = torch.squeeze(max_without_p)

            one_minus = (1 - max_without_p).clamp(min=eps)
            loss_others = - max_without_p ** self.focal_loss_gamma * torch.log(one_minus)
        else:
            p_others = softmax[one_hot == 0].reshape(shape=(len(softmax), self.labels_num - 1))
            one_minus = (1 - p_others).clamp(min=eps)
            loss_others = - p_others ** self.focal_loss_gamma * torch.log(one_minus)
            loss_others = torch.sum(loss_others, dim=1)

        loss_1 = torch.sum(loss_y) / len(softmax)
        loss_2 = torch.sum(loss_others) / len(softmax)
        loss = loss_1 + self.loss_factor * loss_2

        return loss, logits, loss_1, loss_2

    def validate(self, args, test_loader, model):
        model.eval()

        test_samples = 0
        test_total_loss = 0.0
        test_total_loss_1 = 0.0
        test_total_loss_2 = 0.0
        conf_mat_test = np.zeros([args.num_classes, args.num_classes], dtype=np.float64)

        test_entropy_sum = 0.0
        test_margin_sum = 0.0
        test_maxprob_sum = 0.0

        with torch.no_grad():
            for len_x_batch, ipd_x_batch, label_batch in test_loader:
                loss, logits, loss_1, loss_2 = self.forward_propagation(
                    len_x_batch, ipd_x_batch, label_batch, model
                )

                bs = int(len_x_batch.shape[0])
                test_samples += bs
                test_total_loss += float(loss.item()) * bs
                test_total_loss_1 += float(loss_1.item()) * bs
                test_total_loss_2 += float(loss_2.item()) * bs

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

        return conf_mat_test, test_total_loss, test_total_loss_1, test_total_loss_2, test_samples, extra

    def train(self, args, train_loader, test_loader, model, optimizer):
        logger = CurveLogger(labels_num=args.num_classes)

        best_test_f1 = -1e18
        best_epoch = -1

        for epoch in range(1, self.total_epochs + 1):
            self.current_epoch = epoch

            logger.start_epoch(kd_alpha=0.0)

            logger.train_total_kd = float("nan")

            start_time = time.time()
            model.train()

            conf_mat_train = np.zeros([args.num_classes, args.num_classes], dtype=np.float64)

            # keep these only for print (not necessarily in curves)
            train_total_loss_1 = 0.0
            train_total_loss_2 = 0.0

            for len_x_batch, ipd_x_batch, label_batch in train_loader:

                optimizer.zero_grad(set_to_none=True)
                loss, logits, loss_1, loss_2 = self.forward_propagation(
                    len_x_batch, ipd_x_batch, label_batch, model
                )
                loss.backward()

                flat_grads = self._flatten_grads(model)
                logger.update_batch_grads(flat_grads)

                optimizer.step()

                bs = int(len_x_batch.shape[0])

                # curve logger
                logger.update_batch_losses(bs, loss=loss, hard_loss=loss, kd_loss=None)
                logger.update_batch_preds(logits, label_batch, bs)

                update_conf_mat(conf_mat_train, label_batch, logits)

                # for print only
                train_total_loss_1 += float(loss_1.item()) * bs
                train_total_loss_2 += float(loss_2.item()) * bs

            # validate
            conf_mat_test, test_total_loss, test_total_loss_1, test_total_loss_2, test_samples, extra_test = self.validate(args, test_loader, model)

            # epoch stats for print
            train_samples = logger.train_samples
            train_avg_loss = logger.train_total_loss / train_samples if train_samples > 0 else float("nan")
            train_avg_loss_1 = train_total_loss_1 / train_samples if train_samples > 0 else float("nan")
            train_avg_loss_2 = train_total_loss_2 / train_samples if train_samples > 0 else float("nan")

            test_avg_loss = (test_total_loss / test_samples) if test_samples > 0 else float("nan")
            test_avg_loss_1 = (test_total_loss_1 / test_samples) if test_samples > 0 else float("nan")
            test_avg_loss_2 = (test_total_loss_2 / test_samples) if test_samples > 0 else float("nan")

            print("| {:5d}/{:5d} epochs ({:5.2f} s, lr {:8.5f})"
                  "| Train segs {:7d}, Test segs {:7d} "
                  "| Train loss {:7.2f} loss_1 {:7.2f} loss_2 {:7.2f}"
                  "| Test loss {:7.2f} loss_1 {:7.2f} loss_2 {:7.2f}".format(
                      self.current_epoch, self.total_epochs, time.time() - start_time, optimizer.param_groups[0]['lr'],
                      train_samples, test_samples,
                      train_avg_loss, train_avg_loss_1, train_avg_loss_2,
                      test_avg_loss, test_avg_loss_1, test_avg_loss_2
                  ))

            _, _, f1s_test, _ = metric_from_confuse_matrix(conf_mat_test)
            test_f1 = float(np.mean(f1s_test))

            is_best = test_f1 > (best_test_f1 + 1e-12)
            if is_best:
                best_test_f1 = test_f1
                best_epoch = self.current_epoch

            # checkpoint save: best or periodic
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
                        model=model,
                        result_log=logs
                    )
                if self.current_epoch % self.save_checkpoint_epochs == 0:
                    save_checkpoint(
                        output_dir=self.output_dir,
                        model_name='brnn-' + str(self.current_epoch),
                        model=model,
                        result_log=logs
                    )

            # dump unified curves json (single source of truth)
            logger.finish_epoch_and_dump(
                output_dir=self.output_dir,
                conf_mat_train=conf_mat_train,
                conf_mat_test=conf_mat_test,
                test_total_loss=test_total_loss,
                test_samples=test_samples,
                extra_test=extra_test
            )

        print(f"[done] best_epoch={best_epoch}, best_test_f1={best_test_f1:.6f}")

