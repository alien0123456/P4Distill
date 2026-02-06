# API

Importable functions from `distillkit.api`:

```python
from distillkit.api import dataset_stats, run_distillation, evaluate_model, export_p4
```

## `dataset_stats(dataset)`
Returns the parsed `statistics.json` for a dataset.

## `run_distillation(dataset, teacher_model, loss_type="KL", extra_args=None, output_dir=None, student_model="BinaryRNN")`
Runs the distillation pipeline and returns the output directory.

## `evaluate_model(dataset, model_path, extra_args=None, student_model="BinaryRNN")`
Evaluates a saved model and returns a dictionary with summary metrics.

## `export_p4(dataset, model_path, out_path, extra_args=None, student_model="BinaryRNN", quantize="int8", scale=None)`
Exports model parameters to a JSON payload suitable for P4 integration.
