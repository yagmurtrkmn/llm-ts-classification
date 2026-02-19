# LLM Few-Shot Time Series Classification

A framework for evaluating Large Language Models on structured time series classification tasks using zero-shot and few-shot prompting.

---

## Overview

This project investigates whether LLMs can classify time series based on their structural properties — trends, breaks, volatility, and anomalies — using only a textual representation of the series and a structured prompt. It supports multiple classification granularities and a pluggable model registry.

---

## Project Structure

```
.
├── data/
│   ├── raw/                        # Source time series CSVs, organized by category
│   │   ├── anomaly/
│   │   │   ├── collective_anomaly/
│   │   │   ├── contextual_anomaly/
│   │   │   └── point_anomaly/
│   │   ├── deterministic_trend/
│   │   │   ├── cubic_trend/
│   │   │   ├── damped_trend/
│   │   │   ├── exponential_trend/
│   │   │   ├── linear_trend/
│   │   │   └── quadratic_trend/
│   │   ├── stochastic_trend/
│   │   │   ├── ARI/
│   │   │   ├── ARIMA/
│   │   │   ├── IMA/
│   │   │   ├── RW/
│   │   │   └── RWD/
│   │   ├── structural_break/
│   │   │   ├── mean_shift/
│   │   │   ├── trend_shift/
│   │   │   └── variance_shift/
│   │   └── volatility/
│   │       ├── APARCH/
│   │       ├── ARCH/
│   │       ├── EGARCH/
│   │       └── GARCH/
│   ├── few_shot_example_pool/      # Examples used as LLM context (produced by prepare_example_subset.py)
│   └── test_subsets/               # Held-out test sets per task (produced by prepare_test_subsets.py)
│
├── prompts/
│   ├── base.py                     # System-level instruction for the LLM
│   ├── tasks.py                    # Task definitions: labels, descriptions, hierarchies
│   └── builder.py                  # Assembles full prompts from task config + examples
│
├── prepare_example_subset.py       # Samples the few-shot example pool from raw data
├── prepare_test_subsets.py         # Samples held-out test sets (leak-safe) for all tasks
├── evaluation.py                   # Runs evaluation: calls LLM, parses responses, saves metrics
└── README.md
```

---

## Tasks

Six classification tasks are supported, defined in `prompts/tasks.py`:

| Task Key             | Description                                        | # Classes |
|----------------------|----------------------------------------------------|-----------|
| `5_class`            | High-level structural category                     | 5         |
| `9_class`            | Fine-grained structural category                   | 9         |
| `trend_5_class`      | Deterministic trend subtype                        | 5         |
| `break_3_class`      | Structural break subtype                           | 3         |
| `anomaly_3_class`    | Anomaly subtype                                    | 3         |
| `2_class_stationarity` | Stationary vs. Non-Stationary                    | 2         |

### Label Hierarchy

```
├── Deterministic Trend
│   ├── Linear Trend
│   ├── Quadratic Trend
│   ├── Cubic Trend
│   ├── Exponential Trend
│   └── Damped Trend
├── Stochastic Trend
├── Structural Break
│   ├── Mean Shift
│   ├── Variance Shift
│   └── Trend Shift
├── Volatility
└── Anomaly
    ├── Point Anomaly
    ├── Collective Anomaly
    └── Contextual Anomaly
```

---

## Setup

### Requirements

```bash
pip install openai pandas numpy scikit-learn tqdm
```

### Running a Local Model

This project uses [Ollama](https://ollama.com) to serve models locally. Install it and pull your model:

```bash
ollama pull qwen2.5
ollama serve        # starts the local API at http://localhost:11434
```

---

## Usage

### Step 1 — Prepare the Few-Shot Example Pool

Samples a pool of labelled examples from `data/raw/` to use as in-context few-shot examples. Run this **once** before evaluation.

```bash
python prepare_example_subset.py
```

Output: `data/few_shot_example_pool/<label>/samples.csv`

### Step 2 — Prepare Test Subsets

Samples held-out test sets for every task. Automatically excludes any file already present in the few-shot pool to **prevent data leakage**.

```bash
python prepare_test_subsets.py
```

Output: `data/test_subsets/<task>/<label>/samples.csv`

### Step 3 — Run Evaluation

```bash
python evaluation.py --task <task_key> --model <model_key> --shots <n>
```

| Argument  | Required | Description                                              |
|-----------|----------|----------------------------------------------------------|
| `--task`  | Yes      | Task key from the table above (e.g. `5_class`)           |
| `--model` | Yes      | Model key from `MODEL_REGISTRY` (e.g. `qwen2.5`)        |
| `--shots` | No       | Few-shot examples per label. `0` = zero-shot (default)  |

#### Examples

```bash
# Zero-shot, 5-class task
python evaluation.py --task 5_class --model qwen2.5 --shots 0

# 3-shot, fine-grained 9-class task
python evaluation.py --task 9_class --model qwen2.5 --shots 3

# 1-shot, anomaly subtype classification
python evaluation.py --task anomaly_3_class --model qwen2.5 --shots 1
```

---

## Outputs

Each evaluation run produces three files in `evaluation_results/`, named by task, model, and shot count:

```
evaluation_results/
  5_class__qwen2.5__3shot__predictions.csv       # Per-sample: true label, predicted label, raw LLM response
  5_class__qwen2.5__3shot__metrics.csv           # Per-class precision, recall, F1 + macro/weighted averages
  5_class__qwen2.5__3shot__confusion_matrix.csv  # N×N confusion matrix
```

### Sample Console Output

```
============================================================
CLASSIFICATION REPORT  |  task=5_class  |  model=qwen2.5  |  3shot
============================================================
Accuracy: 74.00%  (on 150 valid predictions)

                    precision  recall  f1-score  support
Deterministic Trend   0.81      0.87     0.84       30
Stochastic Trend      0.70      0.63     0.67       30
Structural Break      0.68      0.70     0.69       30
Volatility            0.79      0.83     0.81       30
Anomaly               0.72      0.67     0.69       30

CONFUSION MATRIX:
true \ predicted   Deterministic Trend  Stochastic Trend  ...
Deterministic Trend         26                 2          ...
...
```

---

## Adding a New Model

Open `evaluation.py` and add an entry to `MODEL_REGISTRY`:

```python
MODEL_REGISTRY = {
    "qwen2.5": {
        "base_url": "http://localhost:11434/v1",
        "api_key":  "ollama",
        "model_id": "qwen2.5",
    },
    "llama3": {                              # ← new entry
        "base_url": "http://localhost:11434/v1",
        "api_key":  "ollama",
        "model_id": "llama3",
    },
}
```

Then run:

```bash
ollama pull llama3
python evaluation.py --task 5_class --model llama3 --shots 3
```

---

## Prompt Design

Prompts are assembled in `prompts/builder.py` from three blocks:

1. **Base instruction** (`base.py`) — sets the LLM's role and reasoning style.
2. **Task block** — describes the classification task and provides the label hierarchy.
3. **Few-shot block** — includes `N` labelled examples per class (if `--shots > 0`).

The model is asked to respond in a strict format:

```
Thought: <brief structural reasoning>
Action: <one label from the allowed set>
```

---

## Data Leakage Prevention

`prepare_test_subsets.py` loads all `full_path` values from `data/few_shot_example_pool/` into a set before sampling test files. Any file already in the pool is excluded from all test subsets, ensuring clean train/test separation across all tasks.
