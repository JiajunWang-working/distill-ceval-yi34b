# Yi-34B Distillation Reproducible Release

This repository is a release of the `Yi-1.5-34B-Chat -> Qwen2.5-7B-Instruct` distillation pipeline on full `CEval official`.

It is organized for reproducibility:

1. Only the `Yi-34B` teacher line is kept.
2. The processed `CEval` splits and the generated teacher cache used by the final experiments are included.
3. Large model weights, Hugging Face cache, checkpoints, and training outputs are intentionally excluded.

## 1. What This Repository Reproduces

This release focuses on the final valid experiment line:

1. Dataset: full `CEval official`
2. Teacher model: `01-ai/Yi-1.5-34B-Chat`
3. Student model: `Qwen/Qwen2.5-7B-Instruct`
4. Distillation target: real teacher option-probability distribution
5. Training method: `4-bit QLoRA + LoRA`
6. Main evaluator: `0-shot benchmark-style option likelihood`

The main student variants included here are:

1. `KL`
2. `BayesKD`
3. `KL + noise`
4. `BayesKD + noise`
5. `SABO v3-fast2 Top2`, which is the final best student configuration on this line

## 2. Repository Layout

```text
.
├── configs/                     # Repro configs for teacher eval, student baseline, KD, SABO-best
├── data/
│   ├── ceval_subject_mapping.json
│   ├── processed/
│   │   └── ceval_official_full/ # train/val/test splits used in the final line
│   └── teacher_cache/
│       └── ceval_official_full_yi34b/
│           ├── train.teacher.jsonl
│           └── val.teacher.jsonl
├── docs/                        # Experiment records and result summaries
├── scripts/                     # Entry-point scripts
├── src/                         # Core library
├── environment.yml
├── pyproject.toml
└── requirements*.txt
```

## 3. What Is Not Included

To make GitHub upload practical, this repository does not include:

1. `Yi-1.5-34B-Chat` model weights
2. `Qwen2.5-7B-Instruct` model weights
3. Any Hugging Face cache directory
4. Any trained LoRA adapter checkpoints
5. Any `outputs/` directory contents

You need to download the models yourself before running training or evaluation.

## 4. Models Used

### 4.1 Teacher Model

Teacher model:

1. `01-ai/Yi-1.5-34B-Chat`

How it is used:

1. As a local teacher model
2. Loaded with `trust_remote_code=True`
3. Typically loaded in `4-bit` for local feasibility
4. Used to produce:
   - teacher option probability distributions for distillation
   - teacher benchmark-style evaluation results

### 4.2 Student Model

Student model:

1. `Qwen/Qwen2.5-7B-Instruct`

How it is used:

1. As the base student for all CEval distillation runs
2. Fine-tuned with `QLoRA + LoRA`
3. Evaluated with the same benchmark-style multiple-choice evaluator

## 5. Training and Distillation Setup

### 5.1 Student Fine-Tuning Style

The student training setup in this project is:

1. Base model loaded in `4-bit`
2. `LoRA` adapters on:
   - `q_proj`
   - `k_proj`
   - `v_proj`
   - `o_proj`
   - `up_proj`
   - `down_proj`
   - `gate_proj`
3. `bf16` training
4. Small per-device batch with gradient accumulation

### 5.2 Distillation Target

This project does not use only hard labels.

Instead, the valid final line uses real teacher distributions:

1. For each training example, the local `Yi-34B` teacher is probed on the answer position
2. The probability distribution over options such as `A/B/C/D` is stored in the teacher cache
3. The student is trained against that teacher distribution

### 5.3 Distillation Variants

Included variants:

1. `KL`: plain distillation on teacher option distribution
2. `BayesKD`: Bayesian-style distillation variant implemented in this codebase
3. `noise` variants: noise injected at `model.norm`
4. `SABO`: hyperparameter search over `alpha`, `temperature`, `noise_scale`, `learning_rate`, and epoch budget

## 6. Environment

This project was run in Python `3.10`.

Recommended setup:

```bash
conda env create -f environment.yml
conda activate tight-distill-py310
pip install -r requirements.txt
pip install -r requirements-core.txt
pip install -r requirements-qlora.txt
pip install -e .
```

If you prefer `venv`, make sure you install a CUDA-compatible `torch`, `transformers`, `peft`, `accelerate`, and `bitsandbytes`.

## 7. Required Hardware

Practical notes:

1. The teacher model `Yi-1.5-34B-Chat` is large
2. Teacher-side cache generation and benchmark evaluation were run with quantized loading
3. The original experiments were run on an `NVIDIA L40S`

For a similar reproduction, use a GPU with enough VRAM for:

1. `Yi-34B` in quantized inference
2. `Qwen2.5-7B-Instruct` QLoRA fine-tuning

## 8. Download the Models

This repository is now configured in a portable way:

1. config files default to Hugging Face model IDs
2. you can optionally override them with environment variables
3. dataset and subject-mapping paths are repository-relative

Before running the pipeline, make sure you have access to:

1. `01-ai/Yi-1.5-34B-Chat`
2. `Qwen/Qwen2.5-7B-Instruct`

Recommended environment variables:

```bash
export TEACHER_MODEL_PATH=/path/to/Yi-1.5-34B-Chat
export STUDENT_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
```

If you do not set them, the configs fall back to:

1. `01-ai/Yi-1.5-34B-Chat`
2. `Qwen/Qwen2.5-7B-Instruct`

That means the code can also download from Hugging Face directly if your environment allows it.

## 9. Reproduction Pipeline

### 9.1 Step 1: Evaluate the Teacher on CEval

Teacher benchmark config:

1. `configs/yi15_34b_ceval_benchmark_plain.yaml`

Example command:

```bash
python scripts/evaluate_benchmark_mcq.py \
  --config configs/yi15_34b_ceval_benchmark_plain.yaml \
  --checkpoint "${TEACHER_MODEL_PATH:-01-ai/Yi-1.5-34B-Chat}" \
  --dataset data/processed/ceval_official_full/test.jsonl \
  --few-shot-dataset data/processed/ceval_official_full/train.jsonl \
  --num-shots 0
```

### 9.2 Step 2: Build Teacher Cache

This repository already includes the final `train` and `val` teacher cache used in the final experiments:

1. `data/teacher_cache/ceval_official_full_yi34b/train.teacher.jsonl`
2. `data/teacher_cache/ceval_official_full_yi34b/val.teacher.jsonl`

If you want to rebuild them yourself:

```bash
python scripts/build_teacher_cache_from_model.py \
  --dataset data/processed/ceval_official_full/train.jsonl \
  --output data/teacher_cache/ceval_official_full_yi34b/train.teacher.jsonl \
  --teacher-model "${TEACHER_MODEL_PATH:-01-ai/Yi-1.5-34B-Chat}" \
  --load-in-4bit \
  --trust-remote-code
```

And similarly for `val`:

```bash
python scripts/build_teacher_cache_from_model.py \
  --dataset data/processed/ceval_official_full/val.jsonl \
  --output data/teacher_cache/ceval_official_full_yi34b/val.teacher.jsonl \
  --teacher-model "${TEACHER_MODEL_PATH:-01-ai/Yi-1.5-34B-Chat}" \
  --load-in-4bit \
  --trust-remote-code
```

### 9.3 Step 3: Train the Student

Main training entry:

```bash
python scripts/train_distill.py --config configs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top2_full.yaml
```

Other included configs:

1. `configs/qwen25_7b_ceval_distill_full_yi34b_kl.yaml`
2. `configs/qwen25_7b_ceval_distill_full_yi34b_bayes.yaml`
3. `configs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_kl.yaml`
4. `configs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes.yaml`
5. `configs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top2_full.yaml`

### 9.4 Step 4: Evaluate the Student

Baseline student benchmark:

```bash
python scripts/evaluate_benchmark_mcq.py \
  --config configs/qwen25_7b_ceval_benchmark_plain_full.yaml \
  --checkpoint "${STUDENT_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}" \
  --dataset data/processed/ceval_official_full/test.jsonl \
  --few-shot-dataset data/processed/ceval_official_full/train.jsonl \
  --num-shots 0
```

Distilled student benchmark:

```bash
python scripts/evaluate_benchmark_mcq.py \
  --config configs/qwen25_7b_ceval_distill_full_yi34b_noise_bayes_sabo_v3_fast2_top2_full.yaml \
  --checkpoint /path/to/your/output/final \
  --dataset data/processed/ceval_official_full/test.jsonl \
  --few-shot-dataset data/processed/ceval_official_full/train.jsonl \
  --num-shots 0
```

### 9.5 Step 5: Re-run SABO Search

The search script kept in this release is:

1. `scripts/search_sabo.py`

You can reproduce the final style of search by using:

1. base config: `configs/qwen25_7b_ceval_distill_full_yi34b_noise_u002_bayes.yaml`
2. benchmark evaluator
3. full `CEval official`
4. search over:
   - `alpha`
   - `temperature`
   - `noise_scale`
   - `learning_rate`
   - `num_train_epochs`

## 10. Final Mainline Results

The final valid mainline numbers for this release are:

| Model | Test Accuracy |
| --- | ---: |
| `Qwen2.5-7B-Instruct` baseline | 0.7657591962404796 |
| `Qwen2.5-7B + KL` | 0.7668125101280181 |
| `Qwen2.5-7B + KL + noise` | 0.7702155242262194 |
| `Qwen2.5-7B + BayesKD` | 0.7707826932425863 |
| `Qwen2.5-7B + BayesKD + noise` | 0.7743477556311781 |
| `Qwen2.5-7B + SABO v3-fast2 Top2` | 0.7775887214389888 |

The final best student on this line is:

1. `Qwen2.5-7B + SABO v3-fast2 Top2`
2. Distilled from `Yi-1.5-34B-Chat`
3. With real teacher option probabilities
4. Final test accuracy: `0.7775887214389888`

## 11. PTQ Observation

Additional PTQ-style evaluation was also run for deployment-side reference:

1. `4-bit` evaluation preserved the original scores exactly on the tested models
2. `8-bit` evaluation showed stable positive score shifts on this benchmark path

For the full PTQ notes, see:

1. `docs/experiment_result_ptq_ceval_20260601.md`

## 12. Experiment Records Included

This release includes the core experiment writeups:

1. `docs/experiment_result_yi34b_noise_ceval_20260527.md`
2. `docs/experiment_result_sabo_yi34b_noise_ceval_20260527.md`
3. `docs/experiment_result_ptq_ceval_20260601.md`

## 13. Notes for GitHub Upload

Before pushing:

1. double-check that no `outputs/` directory exists in this release folder
2. double-check that no Hugging Face cache or model snapshots were copied here
3. optionally set `TEACHER_MODEL_PATH` and `STUDENT_MODEL_PATH` for local model directories

This release folder is intentionally lightweight so it can be uploaded directly without large-model files.
