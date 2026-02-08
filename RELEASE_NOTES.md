# Release Notes

## v0.1.0 (Competition Release)

Highlights
- CLI tool with commands: `stats`, `distill`, `eval`, `export`, `wizard`
- Importable API for third-party integration
- BOTIOT-only release profile (CLI defaults and docs aligned)
- Distillation training pipeline built on existing models and trainers
- P4 export JSON format with optional quantization
- Built-in BOTIOT `statistics.json` metadata in package data

Notes
- Full BOTIOT dataset files (`train.json`, `test.json`) and pretrained weights are excluded
- You can place dataset files under `dataset/BOTIOT/json/` or set `DISTILLKIT_DATASET_ROOT`
- Build single-file executable via `scripts/build_exe.ps1` or PyInstaller command
