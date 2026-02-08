# Usage

## CLI Overview

```bash
distillkit <command> [options] [-- <extra-args>]
```

Note: This repository is BOTIOT-only and does not include full datasets or pretrained weights. Provide your data under
`dataset/BOTIOT/json/` (including `train.json`, `test.json`, `statistics.json`) or set
`DISTILLKIT_DATASET_ROOT` to your dataset root path.
Teacher checkpoints can be provided via `--teacher-bm-path`, or by setting
`DISTILLKIT_TEACHER_BM_ROOT`. If omitted, DistillKit uses `distillation/teacher_bm_path`
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
distillkit distill --teacher-model BinaryLSTMWithAttention \
  --teacher-bm-path /root/kaiyuan2/distillation/teacher_bm_path \
  -- --batch_size 64 --T 5
```

Evaluate a model:

```bash
distillkit eval --model-path distillation/save/.../student-best
```

Export to JSON for P4 integration:

```bash
distillkit export --model-path distillation/save/.../student-best --out p4_export.json --quantize int8
```
