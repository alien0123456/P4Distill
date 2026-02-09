# API

Importable functions from `distillkit.api`:

```python
from distillkit.api import (
    dataset_stats,
    train_teacher,
    run_distillation,
    evaluate_model,
    evaluate_and_analyze,
    export_p4,
)
```

## `dataset_stats(dataset)`
Returns the parsed `statistics.json` for `ISCXVPN2016`.

## `train_teacher(dataset, model_name, extra_args=None, output_dir=None)`
Trains a teacher model and returns the output directory.
Use `dataset="ISCXVPN2016"` for this release.

## `run_distillation(dataset, teacher_model, teacher_ckpt_path=None, loss_type="KL", extra_args=None, output_dir=None, student_model="BinaryRNN", teacher_bm_path=None)`
Runs the distillation pipeline and returns the output directory.
Use `dataset="ISCXVPN2016"` for this release.
If `teacher_ckpt_path` is not provided, the API falls back to
`teacher_bm_path` or `DISTILLKIT_TEACHER_BM_ROOT`.
Teacher checkpoints are not bundled in this repository.

## `evaluate_model(dataset, model_path, extra_args=None, student_model="BinaryRNN")`
Evaluates a saved model and returns a dictionary with summary metrics.
Use `dataset="ISCXVPN2016"` for this release.

## `evaluate_and_analyze(baseline_dir, distill_dir, out_dir=None, baseline_result_name="brnn-best-result.txt", distill_result_name="student-best-result.txt", curves_name="learning_curves.json")`
Runs the evaluation + analysis pipeline and returns output paths (plots and summary).

## `export_p4(dataset, model_path, out_path, extra_args=None, student_model="BinaryRNN", quantize="int8", scale=None)`
Exports model parameters to a JSON payload suitable for P4 integration.
Use `dataset="ISCXVPN2016"` for this release.
