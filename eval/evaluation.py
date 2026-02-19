# evaluation.py

import os
import sys
import argparse
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from openai import OpenAI

# ================= IMPORT PROMPT BUILDER =================
sys.path.insert(0, "../prompts")
try:
    from builder import build_prompt
    from tasks import TASK_CONFIGS
except ImportError as e:
    print(f"Error importing prompt modules: {e}")
    print("Ensure 'prompts/builder.py' and 'prompts/tasks.py' exist.")
    sys.exit(1)
# =========================================================

# ================= CONFIGURATION =================
# Resolve all paths relative to the project root (one level up from
# this script if it lives in eval/, or the same dir if at the root).
# This ensures full_path values from samples.csv are always found
# regardless of which directory you run the script from.
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..")) if os.path.basename(_HERE) == "eval" else _HERE

TEST_SUBSET_DIR   = os.path.join(_PROJECT_ROOT, "data", "test_subsets")
FEW_SHOT_POOL_DIR = os.path.join(_PROJECT_ROOT, "data", "few_shot_example_pool")
OUTPUT_DIR        = os.path.join(_PROJECT_ROOT, "evaluation_results")
TARGET_COL        = "data"   # Column name inside each time series CSV
MAX_POINTS        = 500      # Max series length sent to LLM

# --- Supported models ---
MODEL_REGISTRY = {
    "qwen2.5": {
        "base_url": "http://localhost:11434/v1",
        "api_key":  "ollama",
        "model_id": "qwen2.5",
    },
    # Add more models here later, e.g.:
    # "llama3": {
    #     "base_url": "http://localhost:11434/v1",
    #     "api_key":  "ollama",
    #     "model_id": "llama3",
    # },
}
# =================================================


# ------------------------------------------------------------------
# Path resolver
# ------------------------------------------------------------------

def resolve_path(raw_path: str) -> str:
    """
    Return an absolute path for raw_path using the following strategy:
    1. If raw_path is already absolute and exists → return as-is.
    2. If raw_path is relative and exists from cwd → return abspath.
    3. Try resolving relative to _PROJECT_ROOT → return if exists.
    4. Return the original string (will fail later with a clear error).
    """
    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path
    if os.path.exists(raw_path):
        return os.path.abspath(raw_path)
    # Strip leading ./ or ../ and try joining with project root
    stripped = raw_path.lstrip("./").lstrip("../")
    candidate = os.path.join(_PROJECT_ROOT, stripped)
    if os.path.exists(candidate):
        return candidate
    # Last resort: try joining the raw string directly with project root
    candidate2 = os.path.normpath(os.path.join(_PROJECT_ROOT, raw_path))
    if os.path.exists(candidate2):
        return candidate2
    return raw_path   # will produce a clear READ ERROR downstream



# ------------------------------------------------------------------
# Few-shot example loader
# ------------------------------------------------------------------

# Maps each task's labels to the fine-grained pool folder names they
# should draw examples from.  For grouped labels (e.g. "Deterministic
# Trend" in 5_class) we sample evenly across all contributing sub-folders.
_TASK_LABEL_TO_POOL_FOLDERS: dict[str, dict[str, list[str]]] = {
    "5_class": {
        "Deterministic Trend": [
            "Linear Trend", "Quadratic Trend", "Cubic Trend",
            "Exponential Trend", "Damped Trend",
        ],
        "Stochastic Trend": ["Stochastic Trend"],
        "Structural Break": ["Mean Shift", "Variance Shift", "Trend Shift"],
        "Volatility":       ["Volatility"],
        "Anomaly":          ["Point Anomaly", "Collective Anomaly", "Contextual Anomaly"],
    },
    "9_class": {
        "Deterministic Trend": [
            "Linear Trend", "Quadratic Trend", "Cubic Trend",
            "Exponential Trend", "Damped Trend",
        ],
        "Stochastic Trend":   ["Stochastic Trend"],
        "Mean Shift":         ["Mean Shift"],
        "Variance Shift":     ["Variance Shift"],
        "Trend Shift":        ["Trend Shift"],
        "Volatility":         ["Volatility"],
        "Point Anomaly":      ["Point Anomaly"],
        "Collective Anomaly": ["Collective Anomaly"],
        "Contextual Anomaly": ["Contextual Anomaly"],
    },
    "trend_5_class": {
        "Linear Trend":      ["Linear Trend"],
        "Quadratic Trend":   ["Quadratic Trend"],
        "Cubic Trend":       ["Cubic Trend"],
        "Exponential Trend": ["Exponential Trend"],
        "Damped Trend":      ["Damped Trend"],
    },
    "break_3_class": {
        "Mean Shift":     ["Mean Shift"],
        "Variance Shift": ["Variance Shift"],
        "Trend Shift":    ["Trend Shift"],
    },
    "anomaly_3_class": {
        "Point Anomaly":      ["Point Anomaly"],
        "Collective Anomaly": ["Collective Anomaly"],
        "Contextual Anomaly": ["Contextual Anomaly"],
    },
    "2_class_stationarity": {
        "Stationary": [
            "Volatility",
            "Point Anomaly", "Collective Anomaly", "Contextual Anomaly",
        ],
        "Non-Stationary": [
            "Linear Trend", "Quadratic Trend", "Cubic Trend",
            "Exponential Trend", "Damped Trend",
            "Stochastic Trend",
            "Mean Shift", "Variance Shift", "Trend Shift",
        ],
    },
}


def load_few_shot_examples(
    task_name: str,
    shots_per_label: int,
    rng: "random.Random",
) -> list[dict]:
    """
    Load `shots_per_label` examples per label from the few-shot pool.

    Each returned dict has the keys expected by build_prompt():
        observation  : comma-separated z-scored series string
        description  : human-readable label description from pool CSV
        label        : the task-level label (e.g. "Deterministic Trend")

    Examples whose series file cannot be read are silently skipped and
    replaced by the next available candidate so the shot count stays
    consistent where possible.
    """
    import random as _random

    if shots_per_label == 0:
        return []

    label_to_folders = _TASK_LABEL_TO_POOL_FOLDERS.get(task_name)
    if label_to_folders is None:
        print(f"[WARN] No pool mapping defined for task '{task_name}'. Running zero-shot.")
        return []

    examples: list[dict] = []

    for task_label, pool_folders in label_to_folders.items():
        # Gather all candidate rows from every contributing pool folder
        candidates: list[dict] = []
        for folder_name in pool_folders:
            samples_csv = os.path.join(FEW_SHOT_POOL_DIR, folder_name, "samples.csv")
            if not os.path.isfile(samples_csv):
                print(f"  [WARN] Pool CSV not found: {samples_csv}")
                continue
            try:
                df = pd.read_csv(samples_csv)
                if "full_path" in df.columns:
                    df["full_path"] = df["full_path"].apply(
                        lambda p: resolve_path(p) if isinstance(p, str) else p
                    )
                for row in df.to_dict(orient="records"):
                    row["_task_label"] = task_label   # attach the task-level label
                    candidates.append(row)
            except Exception as e:
                print(f"  [WARN] Could not read pool CSV {samples_csv}: {e}")

        if not candidates:
            print(f"  [WARN] No pool candidates for label '{task_label}'.")
            continue

        # Shuffle and pick until we have shots_per_label readable series
        rng.shuffle(candidates)
        collected = 0
        for cand in candidates:
            if collected >= shots_per_label:
                break
            fp = cand.get("full_path", "")
            series_str = format_series(fp)
            if series_str is None:
                continue   # skip unreadable files silently

            description = cand.get("description", f"There is a {task_label} in this series.")
            examples.append({
                "observation": f"[{series_str}]",
                "description": description,
                "label":       task_label,
            })
            collected += 1

        if collected < shots_per_label:
            print(
                f"  [WARN] Only {collected}/{shots_per_label} examples available "
                f"for label '{task_label}'."
            )

    print(
        f"Few-shot examples loaded: {len(examples)} total "
        f"({shots_per_label} per label × {len(label_to_folders)} labels)."
    )
    return examples


# ------------------------------------------------------------------
# LLM client
# ------------------------------------------------------------------

def get_client(model_key: str) -> tuple[OpenAI, str]:
    """Return (OpenAI client, model_id) for the requested model key."""
    if model_key not in MODEL_REGISTRY:
        print(f"Unknown model '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)
    cfg = MODEL_REGISTRY[model_key]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    return client, cfg["model_id"]


def call_llm(client: OpenAI, model_id: str, prompt_text: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in time-series analysis and classification. "
                        "Follow the output format exactly as instructed."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return "API_ERROR"


# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------

def format_series(filepath: str) -> str | None:
    """
    Read TARGET_COL, optionally downsample, z-score normalise,
    and return a comma-separated string.
    """
    try:
        df = pd.read_csv(filepath)
        if TARGET_COL not in df.columns:
            # Fallback: use the first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return None
            col = numeric_cols[0]
        else:
            col = TARGET_COL

        series = df[col].dropna().values

        # Downsample
        if len(series) > MAX_POINTS:
            idx = np.linspace(0, len(series) - 1, MAX_POINTS).astype(int)
            series = series[idx]

        # Z-score normalise
        std = np.std(series)
        series = (series - np.mean(series)) / std if std != 0 else series - np.mean(series)

        return ", ".join(f"{x:.2f}" for x in series)

    except Exception as e:
        print(f"  [READ ERROR] {filepath}: {e}")
        return None


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------

def parse_action(text: str, allowed_labels: list[str]) -> str:
    """
    Extract the predicted label from the LLM response.

    Priority:
    1. 'Action: <label>' line (as defined in builder.py output format)
    2. \\boxed{...} fallback (in case model wraps answer)
    3. Fuzzy match against allowed labels (case-insensitive substring)
    4. PARSE_ERROR
    """
    if text in ("API_ERROR",):
        return text

    # 1. Action: line
    action_match = re.search(r"Action\s*:\s*(.+)", text, re.IGNORECASE)
    if action_match:
        candidate = action_match.group(1).strip().strip('"').strip("'")
        # Exact match first
        if candidate in allowed_labels:
            return candidate
        # Case-insensitive
        for label in allowed_labels:
            if label.lower() == candidate.lower():
                return label

    # 2. \boxed{...} fallback
    box_match = re.search(r"\\boxed\d*\{(.*?)\}", text, re.DOTALL)
    if box_match:
        candidate = box_match.group(1).strip().strip('"').strip("'")
        if candidate in allowed_labels:
            return candidate
        for label in allowed_labels:
            if label.lower() == candidate.lower():
                return label

    # 3. Fuzzy: check if any allowed label appears verbatim in the response
    for label in allowed_labels:
        if label.lower() in text.lower():
            return label

    return "PARSE_ERROR"


# ------------------------------------------------------------------
# Test-subset loader
# ------------------------------------------------------------------

def load_test_samples(task_name: str) -> list[dict]:
    """
    Read all samples.csv files under TEST_SUBSET_DIR/<task_name>/
    and return a flat list of dicts with keys:
        full_path, label, fine_label, file_name
    """
    task_dir = os.path.join(TEST_SUBSET_DIR, task_name)
    if not os.path.isdir(task_dir):
        print(f"Test subset directory not found: {task_dir}")
        sys.exit(1)

    samples = []
    for label_dir in os.listdir(task_dir):
        samples_csv = os.path.join(task_dir, label_dir, "samples.csv")
        if not os.path.isfile(samples_csv):
            continue
        df = pd.read_csv(samples_csv)
        # Resolve full_path values so relative paths from prepare scripts
        # are always found regardless of current working directory
        if "full_path" in df.columns:
            df["full_path"] = df["full_path"].apply(
                lambda p: resolve_path(p) if isinstance(p, str) else p
            )
        samples.extend(df.to_dict(orient="records"))

    if not samples:
        print(f"No samples found for task '{task_name}' in {task_dir}")
        sys.exit(1)

    print(f"Loaded {len(samples)} test samples for task '{task_name}'.")
    return samples


# ------------------------------------------------------------------
# Core evaluation loop
# ------------------------------------------------------------------

def run_evaluation(task_name: str, model_key: str, shots_per_label: int = 0):
    """
    Run LLM-based classification for the given task and model.

    Parameters:
        task_name       : Key from TASK_CONFIGS (e.g. '5_class')
        model_key       : Key from MODEL_REGISTRY (e.g. 'qwen2.5')
        shots_per_label : Number of few-shot examples per label (0 = zero-shot)
    """
    import random as _random

    if task_name not in TASK_CONFIGS:
        print(f"Unknown task '{task_name}'. Available: {list(TASK_CONFIGS.keys())}")
        sys.exit(1)

    allowed_labels = TASK_CONFIGS[task_name]["labels"]
    client, model_id = get_client(model_key)
    samples = load_test_samples(task_name)

    # Load few-shot examples once — reused for every query
    rng = _random.Random(42)
    few_shot_examples = load_few_shot_examples(task_name, shots_per_label, rng)
    shot_tag = f"{shots_per_label}shot" if shots_per_label > 0 else "0shot"

    results_log = []
    y_true, y_pred = [], []

    print(f"\n--- Evaluating task='{task_name}' | model='{model_key}' | {shot_tag} ---\n")

    for sample in tqdm(samples):
        full_path  = sample.get("full_path", "")
        true_label = sample.get("label", "")
        fine_label = sample.get("fine_label", true_label)
        file_name  = sample.get("file_name", os.path.basename(full_path))

        # Prepare series string
        series_str = format_series(full_path)
        if series_str is None:
            print(f"  [SKIP] Could not read series: {full_path}")
            continue

        observation = f"[{series_str}]"

        # Build prompt via builder.py
        prompt = build_prompt(
            task_name=task_name,
            query_observation=observation,
            support_examples=few_shot_examples,
            include_hierarchy=True,
        )

        # Call LLM
        raw_response = call_llm(client, model_id, prompt)

        # Parse prediction
        prediction = parse_action(raw_response, allowed_labels)

        y_true.append(true_label)
        y_pred.append(prediction)

        results_log.append({
            "file_name":       file_name,
            "full_path":       full_path,
            "true_label":      true_label,
            "fine_label":      fine_label,
            "predicted_label": prediction,
            "is_correct":      (true_label == prediction),
            "raw_response":    raw_response,
        })

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = f"{task_name}__{model_key}__{shot_tag}"
    out_csv = os.path.join(OUTPUT_DIR, f"{base_name}__predictions.csv")
    df_results = pd.DataFrame(results_log)
    df_results.to_csv(out_csv, index=False)
    print(f"\nDetailed predictions saved to: {out_csv}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"CLASSIFICATION REPORT  |  task={task_name}  |  model={model_key}  |  {shot_tag}")
    print("=" * 60)

    if not y_true:
        print("No valid samples were processed.")
        return

    # Separate clean predictions from errors
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred)
                   if p not in ("PARSE_ERROR", "API_ERROR")]
    error_count = len(y_true) - len(valid_pairs)

    if error_count:
        print(f"  Parse/API errors (excluded from metrics): {error_count}/{len(y_true)}\n")

    if not valid_pairs:
        print("No valid predictions to evaluate.")
        return

    vt, vp = zip(*valid_pairs)
    vt, vp = list(vt), list(vp)

    # ---- Accuracy ----
    acc = accuracy_score(vt, vp)
    print(f"Accuracy: {acc:.2%}  (on {len(valid_pairs)} valid predictions)\n")

    # ---- Per-class precision / recall / F1 ----
    print(classification_report(vt, vp, labels=allowed_labels, zero_division=0))

    # ---- Confusion matrix ----
    cm = confusion_matrix(vt, vp, labels=allowed_labels)
    cm_df = pd.DataFrame(cm, index=allowed_labels, columns=allowed_labels)
    cm_df.index.name   = "true \\ predicted"

    print("CONFUSION MATRIX:")
    print(cm_df.to_string())
    print()

    # ---- Save metrics CSV ----
    precision, recall, f1, support = precision_recall_fscore_support(
        vt, vp, labels=allowed_labels, zero_division=0
    )
    metrics_rows = []
    for i, label in enumerate(allowed_labels):
        metrics_rows.append({
            "label":     label,
            "precision": round(precision[i], 4),
            "recall":    round(recall[i],    4),
            "f1":        round(f1[i],        4),
            "support":   int(support[i]),
        })
    # Macro / weighted averages
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            vt, vp, labels=allowed_labels, average=avg, zero_division=0
        )
        metrics_rows.append({
            "label":     f"avg_{avg}",
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f, 4),
            "support":   len(valid_pairs),
        })
    metrics_rows.append({
        "label":     "accuracy",
        "precision": "",
        "recall":    "",
        "f1":        round(acc, 4),
        "support":   len(valid_pairs),
    })

    df_metrics = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base_name}__metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Per-class metrics saved to:  {metrics_csv}")

    # Save confusion matrix CSV
    cm_csv = os.path.join(OUTPUT_DIR, f"{base_name}__confusion_matrix.csv")
    cm_df.to_csv(cm_csv)
    print(f"Confusion matrix saved to:   {cm_csv}")

    # ---- Summary line ----
    print(f"\n[SUMMARY] task={task_name} | model={model_key} | {shot_tag} | "
          f"n={len(y_true)} | errors={error_count} | "
          f"acc={acc:.2%} | macro-f1={f1.mean():.2%}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM time-series classification on a test subset."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_CONFIGS.keys()),
        help="Task name defined in tasks.py",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model key defined in MODEL_REGISTRY",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Number of few-shot examples per label (0 = zero-shot, default: 0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(task_name=args.task, model_key=args.model, shots_per_label=args.shots)