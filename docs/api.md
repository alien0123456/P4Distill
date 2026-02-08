# API

Importable functions from `distillkit.api`:

```python
from distillkit.api import dataset_stats, run_distillation, evaluate_model, export_p4
```

## `dataset_stats(dataset)`
Returns the parsed `statistics.json` for `ISCXVPN2016`.

## `run_distillation(dataset, teacher_model, loss_type="KL", extra_args=None, output_dir=None, student_model="BinaryRNN", teacher_bm_path=None)`
Runs the distillation pipeline and returns the output directory.
Use `dataset="ISCXVPN2016"` for this release.
`student_model` is currently fixed to `BinaryRNN` in the underlying `distillation` pipeline.
If `teacher_bm_path` is not provided, the API falls back to
`DISTILLKIT_TEACHER_BM_ROOT`, then `teacher_bm_path` under repo root.
Teacher checkpoints are not bundled in this repository.

## `evaluate_model(dataset, model_path, extra_args=None, student_model="BinaryRNN")`
Evaluates a saved model and returns a dictionary with summary metrics.
Use `dataset="ISCXVPN2016"` for this release.

## `export_p4(dataset, model_path, out_path, extra_args=None, student_model="BinaryRNN", quantize="int8", scale=None)`
Exports model parameters to a JSON payload suitable for P4 integration.
Use `dataset="ISCXVPN2016"` for this release.
