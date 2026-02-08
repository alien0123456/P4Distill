# P4 Integration

DistillKit exports model parameters to JSON for consumption by P4 control-plane tooling.

## Export Format

`export.json` fields
- `metadata`: dataset, model name, quantization mode
- `quantize`: quantization mode (`none`, `int8`, `int16`)
- `parameters`: dictionary of tensors with
  - `shape`: tensor shape
  - `scale`: quantization scale (if quantized)
  - `values`: flattened integer list or floats

## Example

```bash
distillkit export --model-path save_kd/.../student-best --out p4_export.json
```

## Recommended Next Step

Write a small control-plane translator to load `p4_export.json` and program registers/tables.
