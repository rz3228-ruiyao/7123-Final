# Pixels to Predictions: DL Vision Challenge

LoRA fine-tuning of SmolVLM-500M-Instruct for the NYU Deep Learning Spring 2026 ScienceQA Kaggle competition.

**Public LB: 0.86519**
**Author:** Ruiyao Zhang (`rz3228@nyu.edu`)
**Kaggle:** `rz3228-ruiyao` (Team: OOM)

## Approach

The task is multiple-choice classification from image + text context, with 2–5 candidate choices per question. Inference is reduced to a single forward pass per question by scoring the model's logits at the answer-letter position over valid choice letters, then taking the argmax.

The pipeline applies LoRA (with DoRA decomposition) only to the language-model attention projections (`q_proj, k_proj, v_proj, o_proj`), keeping the vision tower frozen. This stays under the 5M trainable-parameter cap with `r=24, alpha=48` while reaching the dataset's effective ceiling.

Three measures address position bias diagnosed during data analysis:
- **Choice-order permutation** during training (every epoch sees a different ordering)
- **Test-time augmentation** with K=4 random permutations at inference
- **Answer-letter-only loss masking** so the small adapter focuses capacity on classification

The most consequential learning-rate choice is the *extended cosine* schedule: schedule for 11 epochs but train for only 9, so the LR never enters its near-zero cooldown phase during actual training.

## Repository Structure

```
.
├── README.md                  # This file
├── requirements.txt           # Pinned dependencies
├── data_analysis.ipynb        # Dataset characterization and zero-shot baselines
├── training_v2.ipynb          # End-to-end training and inference pipeline
├── run_logger.py              # Append-only JSONL run logger imported by the notebook
├── evidence.json              # Per-feature justification with measured numbers
└── report/
    ├── report.tex             # LaTeX report
    └── pics/                  # Figures (extracted from data_analysis.ipynb)
```

## Reproducing the Best Submission

### 1. Environment

Tested on Ubuntu 24.04 with an NVIDIA RTX 4080 Super (16 GB VRAM), CUDA 12.4, Python 3.11.

```bash
pip install -r requirements.txt
```

### 2. Data Layout

Place the competition data so the notebooks can load it from the working directory:

```
.
├── train.csv
├── val.csv
├── test.csv
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── (notebooks here)
```

If your data lives elsewhere, edit `cfg.data_dir` in `training_v2.ipynb` (Section 1).

### 3. Run the Notebooks

Run `data_analysis.ipynb` first if you want to reproduce the baseline numbers and figures (~10 minutes). It is read-only on the data and writes nothing.

Run `training_v2.ipynb` to train and produce a submission (~2.5 hours):

1. Edit `RUN_NOTES` in Section 1 with a brief description of the run.
2. Run all cells top to bottom.
3. The notebook writes `runs/lora_v2/<run_id>/best/` (LoRA weights) and `runs/lora_v2/<run_id>/submission.csv`.
4. After Kaggle scoring, fill in `logger.add_lb_score(run_id, public_lb=...)` in the final cell to append the LB number to `runs/runs.jsonl`.

### 4. Best-Configuration Settings

The submission scoring 0.86519 used:

| Setting | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| LoRA rank / alpha | `r=24, alpha=48` (with `use_dora=True`) |
| Trainable params | 4,915,200 |
| Epochs | 9 |
| Batch size | 4 (with `grad_accum=4`, effective batch 16) |
| Learning rate | `2e-4` |
| LR schedule | Cosine, scheduled for 11 epochs, train for 9 (extended cosine) |
| Warmup ratio | 0.03 |
| Image size | 224×224 |
| TTA | K=4 random choice permutations |
| Random seed | 42 |

The `RUN_NOTES` for this configuration was: `"dora, only on train, e=9, cos schedule, totalsteps=e11"`.

## Hardware and Timing

| Stage | Time |
|---|---|
| `data_analysis.ipynb` (zero-shot baselines) | ~10 min |
| `training_v2.ipynb` (training, 9 epochs) | ~2.5 hours |
| `training_v2.ipynb` (final TTA + test inference) | ~25 min |

Total: about 3 hours per run on a 16 GB GPU.

## Run Log

Every training run appends a record to `runs/runs.jsonl` with the full configuration, per-epoch loss and validation accuracy (with per-`num_choices` breakdown), final TTA validation accuracy, and the public LB score once submitted. Use `logger.show_table()` (last cell of `training_v2.ipynb`) to view all runs side by side.

The 16 experiments described in the report can all be reproduced by editing the relevant CFG fields in `training_v2.ipynb` and re-running.

## Model Weights

Best LoRA adapter weights: [Google Drive link](https://drive.google.com/file/d/<placeholder>/view?usp=sharing)

Place the downloaded `best/` directory under `runs/lora_v2/<your_run_id>/` and load with `model.load_adapter()` to skip retraining.

## AI Tooling Disclosure

Claude (Anthropic) was used for coding assistance and debugging.

## License

MIT.
