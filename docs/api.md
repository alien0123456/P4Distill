# API

Importable functions from `distillkit.api`:

```python
from distillkit.api import dataset_stats, run_distillation, evaluate_model, export_p4
```

## `dataset_stats(dataset)`
Returns the parsed `statistics.json`. This release guarantees BOTIOT metadata.

## `run_distillation(dataset, teacher_model, loss_type="KL", extra_args=None, output_dir=None, student_model="BinaryRNN", teacher_bm_path=None)`
Runs the distillation pipeline and returns the output directory.
Use `dataset="BOTIOT"` for this release.
If `teacher_bm_path` is not provided, the API falls back to
`DISTILLKIT_TEACHER_BM_ROOT`, then `distillation/teacher_bm_path` under repo root.

## `evaluate_model(dataset, model_path, extra_args=None, student_model="BinaryRNN")`
Evaluates a saved model and returns a dictionary with summary metrics.
Use `dataset="BOTIOT"` for this release.

## `export_p4(dataset, model_path, out_path, extra_args=None, student_model="BinaryRNN", quantize="int8", scale=None)`
Exports model parameters to a JSON payload suitable for P4 integration.
Use `dataset="BOTIOT"` for this release.
