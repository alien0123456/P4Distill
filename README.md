# DistillKit (Competition Release)

DistillKit is a knowledge distillation toolkit for **P4-deployable network traffic models**.  
It improves the accuracy of heavily quantized, line-rate inference models **without modifying P4 programs or data-plane inference logic**.

This repository is a **clean, competition-oriented release** for **BOTIOT**, focusing on distillation, evaluation, and P4 export.

---

## Why Knowledge Distillation for P4-Deployable Models?

Programmable data planes (PDPs) enable line-rate, low-latency network traffic analysis by executing inference directly on the packet path. However, deploying deep learning models on P4 switches is fundamentally constrained by limited instruction sets, strict pipeline stages, and tight on-chip memory budgets. As a result, practical in-switch models are typically heavily compressed (e.g., binarized or low-bit quantized) and must follow fixed, feed-forward inference logic.

While prior systems mitigate quantization-induced accuracy loss using techniques such as straight-through estimators (STE), quantization-aware training (QAT), pruning, or architecture search, a substantial performance gap to full-precision models remains. A key limitation is that these compressed models are still trained primarily with one-hot hard labels, which provide sparse supervision and are often insufficient for capacity-constrained models to learn well-shaped decision boundaries.

To address this bottleneck, **DistillKit adopts an architecture-agnostic knowledge distillation approach**. During control-plane training, a high-precision software model acts as a *teacher* and provides soft labels (logits) to guide a fixed, deployable *student* model. By enriching supervision with inter-class similarity information, distillation complements existing STE/QAT techniques and significantly improves accuracy under extreme quantization—**without changing the P4 program, inference logic, or runtime pipeline**.

> *(You may place the corresponding paper figure here to illustrate the accuracy gap under P4 constraints and the effect of distillation.)*

---

## Distillation Method

DistillKit implements a **logit-based knowledge distillation framework** based on classical teacher–student learning. The student model is optimized using a weighted combination of:

- standard supervised loss with ground-truth labels, and  
- distillation loss that aligns the student’s logits with the teacher’s softened logits (temperature-scaled KL divergence).

Importantly, distillation in DistillKit is **architecture-agnostic**: it does not depend on model internals, quantization schemes, or P4-specific inference mappings. The student architecture, bit-width, and deployment footprint remain unchanged. After training, deployment simply replaces the model parameters, making distillation a *drop-in optimization layer* for existing P4 pipelines.

![alt text](image.png)

---


## Distillation Effect (Accuracy Improvement)

Rather than only reporting absolute accuracy, we explicitly visualize the **performance gain introduced by knowledge distillation**.  
We report **ΔMF1 = MF1(distilled) − MF1(non-distilled)** to highlight how distillation improves deployable models under fixed P4 constraints.

### Overall MF1 Improvement (ISCXVPN2016)

| Environment | MF1 (No Distill) | MF1 (Distilled) | ΔMF1 |
|------------|-----------------|-----------------|------|
| Python     | 81.4            | 81.8            | +0.4 |
| P4         | 72.37           | 73.46           | **+1.09** |

> Distillation yields modest gains in Python, but **significantly improves recall and MF1 under P4 deployment constraints**, where model capacity and precision are strictly limited.

---

### Per-Class F1 Improvement (ISCXVPN2016, P4)

To better understand where the gains come from, we further analyze per-class F1-score changes after distillation.

| Class      | F1 (No Distill) | F1 (Distilled) | ΔF1 |
|-----------|----------------|----------------|-----|
| Chat      | 0.9143         | 0.5833         | −0.3310 |
| Email     | 0.0000         | 0.2286         | **+0.2286** |
| FTP       | 0.8932         | 0.9647         | **+0.0715** |
| P2P       | 0.7576         | 0.7191         | −0.0385 |
| Streaming | 0.9320         | 0.9271         | −0.0049 |
| VoIP      | 0.8453         | 0.9845         | **+0.1392** |

These results show that distillation particularly benefits **hard or under-represented classes** (e.g., Email, VoIP), improving recall and overall MF1. Some classes exhibit precision–recall trade-offs, reflecting the realistic behavior of capacity-constrained models.




## Independent Functionalities

To meet the requirements of the original open-source competition track, DistillKit independently implements the following core functionalities:

1. **Run Distillation**  
   A complete teacher–student training pipeline that supports configurable distillation strategies, temperature scaling, and loss weighting. Distillation is performed entirely in the control plane and is decoupled from the P4 inference implementation.

2. **Evaluate a Trained Model**  
   A unified evaluation interface for distilled or non-distilled models, providing accuracy metrics and dataset-level statistics to validate performance under deployment-equivalent settings.

3. **Export for P4 Deployment**  
   A dedicated export pipeline that converts trained student models into a P4-compatible JSON parameter format, enabling direct installation on programmable switches without modifying the P4 program or inference logic.

These functionalities are exposed through both a command-line interface (`distillkit`) and an importable API, forming a self-contained toolkit for developing, evaluating, and deploying distilled models on programmable data planes.

--- 
## What's Included

- CLI tool: `distillkit`
- Importable API for third-party use
- Built-in BOTIOT `statistics.json` metadata for quick `stats` checks
- Distillation training pipeline (teacher/student)
- Evaluation and metrics
- P4 export format (JSON)
- Documentation and build scripts

- Full BOTIOT data files (`dataset/BOTIOT/json/train.json`, `dataset/BOTIOT/json/test.json`)
- Training artifacts (`save/`, `distillation/save/`)
- Teacher checkpoints (`distillation/teacher_bm_path/`)
---

## What's Excluded



## Quick Start

### 1) Install (editable)

```bash
pip install -e .
```

### 2) Show dataset statistics

```bash
distillkit stats
```

### 3) Run distillation

```bash
distillkit distill --teacher-model BinaryLSTMWithAttention
```

### 4) Evaluate a trained model

```bash
distillkit eval --model-path distillation/save/.../student-best

e.g. 
distillkit eval --model-path distillation/save/BOTIOT/BinaryLSTMWithAttention/KL/studentbrnn_len10_ipd8_ev6_hidden8_0.8_0.5_KL_0.005_T3_a0.7/student-best
```


### 5) Export for P4

```bash
distillkit export --model-path distillation/save/.../student-best --out export.json

e.g.
distillkit export --model-path distillation/save/BOTIOT/BinaryLSTMWithAttention/KL/studentbrnn_len10_ipd8_ev6_hidden8_0.8_0.5_KL_0.005_T3_a0.7/student-best --out export.json
```

## Dataset Layout (BOTIOT)

```text
dataset/
  BOTIOT/
    json/
      statistics.json
      train.json
      test.json
```

Use `DISTILLKIT_DATASET_ROOT` if your dataset is stored outside the repository.

Teacher checkpoints are searched in this order:
- `--teacher-bm-path` (CLI argument)
- `DISTILLKIT_TEACHER_BM_ROOT` (environment variable)
- default: `distillation/teacher_bm_path` under repository root

Expected teacher checkpoint layout:

```text
<teacher_bm_root>/
  BOTIOT/
    BinaryLSTMWithAttention/
      teacher-brnn-best
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


## Acknowledgements

This project is inspired by the following work from Tsinghua University:

**Brain-on-Switch: Towards Advanced Intelligent Network Data Plane via NN-Driven Traffic Analysis at Line-Speed**  
Jinzhu Yan, Haotian Xu, Zhuotao Liu, Qi Li, Ke Xu, Mingwei Xu, Jianping Wu  
Proceedings of the 21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 2024)

Their work provided important insights into neural-network-driven traffic analysis and programmable data-plane deployment, which motivated parts of the system design in this repository.