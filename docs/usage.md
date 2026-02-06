# Usage

## CLI Overview

```bash
distillkit <command> [options] [-- <extra-args>]
```

Note: This repository does not include datasets or pretrained weights. Provide your own datasets under
`dataset/<DATASET>/json/` or update paths via CLI extra args as needed.

Commands
- `stats`: Show dataset statistics.
- `distill`: Run knowledge distillation training.
- `eval`: Evaluate a saved student model.
- `export`: Export a model for P4 integration.
- `wizard`: Interactive setup for distillation.

## Examples

Show dataset stats:

```bash
distillkit stats --dataset ISCXVPN2016
```

Run distillation with extra training args:

```bash
distillkit distill --dataset BOTIOT --teacher-model BinaryLSTMWithAttention -- --batch_size 64 --T 5
```

Evaluate a model:

```bash
distillkit eval --dataset BOTIOT --model-path distillation/save/.../student-best
```

Export to JSON for P4 integration:

```bash
distillkit export --dataset BOTIOT --model-path distillation/save/.../student-best --out p4_export.json --quantize int8
```
