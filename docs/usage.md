# Usage

## CLI Overview

```bash
distillkit <command> [options] [-- <extra-args>]
```

Note: This repository is aligned with ISCXVPN2016 workflows. Provide your data under
`dataset/ISCXVPN2016/json/` (including `train.json`, `test.json`, `statistics.json`) or set
`DISTILLKIT_DATASET_ROOT` to your dataset root path.
Teacher checkpoints are not bundled. Provide them via `--teacher-bm-path`, or by setting
`DISTILLKIT_TEACHER_BM_ROOT`. If omitted, DistillKit looks for `teacher_bm_path`
under the repository root.

Commands
- `stats`: Show dataset statistics.
- `distill`: Run knowledge distillation training.
- `eval`: Evaluate a saved student model.
- `export`: Export a model for P4 integration.
- `wizard`: Interactive setup for distillation.

## Examples

Show dataset stats:

```bash
distillkit stats
```

Run distillation with extra training args:

```bash
distillkit distill --teacher-model BiLSTMWithAttention \
  --teacher-bm-path /root/kaiyuan2/teacher_bm_path \
  -- --train_batch_size 64 --kd_temperature 4 --max_epochs 10
```

Evaluate a model:

```bash
distillkit eval --model-path save_kd/.../student-best
```

Export to JSON for P4 integration:

```bash
distillkit export --model-path save_kd/.../student-best --out p4_export.json --quantize int8
```
