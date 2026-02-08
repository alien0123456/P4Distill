# Architecture

DistillKit exposes a stable CLI/API wrapper and delegates training/evaluation to the `distillation/` implementation in this repository.

Key layers
- CLI (`src/distillkit/cli.py`): user-facing commands and interactive wizard.
- API (`src/distillkit/api.py`): stable programmatic entry points for third-party use.
- Runtime adapters (`src/distillkit/core/legacy.py`): normalize args and option parsing for `distillation.opts`.
- Export (`src/distillkit/export/p4_export.py`): model parameter export for switch integration.

Data flow
- Dataset JSON -> `distillation.utils.data_loader.FlowDataset` -> training/evaluation via `distillation.trainers.*`.
- Teacher model is loaded from `--teacher-bm-path` / `DISTILLKIT_TEACHER_BM_ROOT`
  or default `teacher_bm_path/...` under repository root.
- Metrics are computed via `distillation.utils.classification_metrics.metric_from_confuse_matrix`.
