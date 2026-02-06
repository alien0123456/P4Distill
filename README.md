# DistillKit (Competition Release)

Knowledge distillation toolkit for P4-deployable network traffic models.

This repository is a **clean competition release**. It includes the full source code for the toolkit and CLI, but excludes datasets and training artifacts.

## What's Included

- CLI tool: `distillkit`
- Importable API for third-party use
- Distillation training pipeline (teacher/student)
- Evaluation and metrics
- P4 export format (JSON)
- Documentation and build scripts

## What's Excluded

- Datasets (`dataset/`)
- Training artifacts (`save/`, `distillation/save/`)
- Teacher checkpoints (`distillation/teacher_bm_path/`)

## Quick Start

### 1) Install (editable)

```bash
pip install -e .
```

### 2) Show dataset statistics

```bash
distillkit stats --dataset BOTIOT
```

### 3) Run distillation

```bash
distillkit distill --dataset BOTIOT --teacher-model BinaryLSTMWithAttention
```

### 4) Evaluate a trained model

```bash
distillkit eval --dataset BOTIOT --model-path distillation/save/.../student-best
```

### 5) Export for P4

```bash
distillkit export --dataset BOTIOT --model-path distillation/save/.../student-best --out export.json
```

## Build Executable (Single File)

```bash
pip install pyinstaller
pyinstaller --onefile -n distillkit src/distillkit/cli.py
```

Output:
- `dist/distillkit.exe`

## Documentation

- `docs/usage.md`
- `docs/api.md`
- `docs/architecture.md`
- `docs/p4_integration.md`

## License

MIT License. See `LICENSE`.
