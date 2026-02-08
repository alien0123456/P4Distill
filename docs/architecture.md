# Architecture

DistillKit is a thin product layer on top of the existing research code in `distillation/`.

Key layers
- CLI (`src/distillkit/cli.py`): user-facing commands and interactive wizard.
- API (`src/distillkit/api.py`): stable programmatic entry points for third-party use.
- Legacy adapters (`src/distillkit/core/legacy.py`): safely reuse `distillation/` modules.
- Export (`src/distillkit/export/p4_export.py`): model parameter export for switch integration.

Data flow
- Dataset JSON -> `FlowDataset` -> student model training via `trainer2`.
- Teacher model is loaded from `--teacher-bm-path` / `DISTILLKIT_TEACHER_BM_ROOT`
  or default `distillation/teacher_bm_path/...`.
- Metrics are computed via `utils.metric.metric_from_confuse_matrix`.
