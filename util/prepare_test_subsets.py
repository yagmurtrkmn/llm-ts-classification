import os
import random
import pandas as pd
from tqdm import tqdm

# ================= CONFIG =================
# All paths are resolved to absolute at runtime so that the saved
# full_path values in samples.csv are portable across scripts that
# may be run from different working directories (e.g. eval/).
_HERE = os.path.dirname(os.path.abspath(__file__))   # project root (where this script lives)

SOURCE_DIR        = os.path.join(_HERE, "../data", "raw")
DEST_DIR          = os.path.join(_HERE, "../data", "test_subsets")
FEW_SHOT_POOL_DIR = os.path.join(_HERE, "../data", "few_shot_example_pool")
MAX_SERIES_LENGTH = 500
RANDOM_SEED       = 42

# --------------- Per-task sample quotas ---------------

# 5-class: n samples per high-level class
N_5_CLASS = 3          # per high-level class (Deterministic Trend, Stochastic Trend, Structural Break, Volatility, Anomaly)

# 9-class: samples per fine-grained class
N_9_CLASS = 3          # per label

# trend_5_class: samples per trend subtype
N_TREND_5 = 3

# break_3_class: samples per structural break subtype
N_BREAK_3 = 3

# anomaly_3_class: samples per anomaly subtype
N_ANOMALY_3 = 3

# 2_class_stationarity: m samples per class
M_STATIONARITY = 3     # per class (Stationary, Non-Stationary)

# ==========================================

# ------------------------------------------------------------------
# Raw folder → fine-grained label mapping (same logic as prepare_example_subset.py)
# ------------------------------------------------------------------
FINE_GRAINED_FOLDER_MAP = {
    "deterministic_trend/linear_trend":     "Linear Trend",
    "deterministic_trend/quadratic_trend":  "Quadratic Trend",
    "deterministic_trend/cubic_trend":      "Cubic Trend",
    "deterministic_trend/exponential_trend":"Exponential Trend",
    "deterministic_trend/damped_trend":     "Damped Trend",
    "stochastic_trend":                     "Stochastic Trend",
    "structural_break/mean_shift":          "Mean Shift",
    "structural_break/variance_shift":      "Variance Shift",
    "structural_break/trend_shift":         "Trend Shift",
    "volatility":                           "Volatility",
    "anomaly/point_anomaly":                "Point Anomaly",
    "anomaly/collective_anomaly":           "Collective Anomaly",
    "anomaly/contextual_anomaly":           "Contextual Anomaly",
}

# ------------------------------------------------------------------
# Groupings for high-level tasks
# ------------------------------------------------------------------

# 5-class grouping: high-level label → list of fine-grained labels
HIGH_LEVEL_GROUPS = {
    "Deterministic Trend": [
        "Linear Trend", "Quadratic Trend", "Cubic Trend",
        "Exponential Trend", "Damped Trend"
    ],
    "Stochastic Trend": ["Stochastic Trend"],
    "Structural Break": ["Mean Shift", "Variance Shift", "Trend Shift"],
    "Volatility":       ["Volatility"],
    "Anomaly":          ["Point Anomaly", "Collective Anomaly", "Contextual Anomaly"],
}

# 2-class stationarity grouping
#   Stationary     → Volatility  (stationary but heteroskedastic, still I(0))
#   Non-Stationary → everything with a unit root / trend / break / anomaly
STATIONARITY_GROUPS = {
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
}

# ------------------------------------------------------------------
# Few-shot leak guard
# ------------------------------------------------------------------

def load_few_shot_paths(pool_dir: str) -> set[str]:
    """
    Walk every samples.csv inside the few-shot example pool and collect
    all full_path values into a set.  These files must be excluded from
    test subsets to prevent data leakage.

    Returns an empty set (with a warning) if the pool directory does not
    exist yet, so the script remains runnable before the pool is created.
    """
    seen: set[str] = set()

    if not os.path.isdir(pool_dir):
        print(f"[WARN] Few-shot pool directory not found: {pool_dir}")
        print("       Leakage check skipped — run prepare_example_subset.py first.")
        return seen

    for root, _, files in os.walk(pool_dir):
        for f in files:
            if f == "samples.csv":
                fp = os.path.join(root, f)
                try:
                    df = pd.read_csv(fp)
                    if "full_path" in df.columns:
                        seen.update(df["full_path"].dropna().tolist())
                except Exception as e:
                    print(f"[WARN] Could not read {fp}: {e}")

    print(f"Few-shot pool loaded: {len(seen)} files will be excluded from test sets.")
    return seen


def exclude_few_shot(file_list: list[str], few_shot_paths: set[str]) -> list[str]:
    """Return file_list with any path that appears in few_shot_paths removed."""
    clean = [fp for fp in file_list if fp not in few_shot_paths]
    removed = len(file_list) - len(clean)
    return clean, removed


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

def is_valid_length(filepath, max_len=MAX_SERIES_LENGTH):
    try:
        df = pd.read_csv(filepath, usecols=[0])
        return len(df) <= max_len
    except Exception:
        return False


def collect_fine_grained_files(few_shot_paths: set[str]):
    """
    Returns dict: fine_label -> list of valid file paths, with any path
    that appears in the few-shot example pool already removed.
    """
    fine_files: dict[str, list[str]] = {label: [] for label in FINE_GRAINED_FOLDER_MAP.values()}

    print("Scanning & filtering source files...")
    for raw_rel, fine_label in tqdm(FINE_GRAINED_FOLDER_MAP.items()):
        folder = os.path.join(SOURCE_DIR, raw_rel)
        if not os.path.isdir(folder):
            print(f"  [WARN] Folder not found: {folder}")
            continue

        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".csv") and f != "metadata.csv":
                    fp = os.path.join(root, f)
                    if is_valid_length(fp):
                        fine_files[fine_label].append(fp)

    print("\nApplying few-shot leakage filter...")
    total_removed = 0
    for label in list(fine_files.keys()):
        clean, removed = exclude_few_shot(fine_files[label], few_shot_paths)
        fine_files[label] = clean
        total_removed += removed
        status = f"  {label}: {len(clean)} usable"
        if removed:
            status += f"  ({removed} excluded — in few-shot pool)"
        print(status)

    print(f"\nTotal files excluded across all classes: {total_removed}")
    return fine_files


def sample_files(file_list: list[str], n: int, rng: random.Random) -> list[str]:
    return rng.sample(file_list, min(n, len(file_list)))


def save_subset(rows: list[dict], task_name: str, label: str):
    """Save a list of row dicts to DEST_DIR/<task_name>/<label>/samples.csv"""
    out_dir = os.path.join(DEST_DIR, task_name, label)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "samples.csv"), index=False)
    print(f"    [{task_name}] {label}: {len(rows)} samples saved.")


def make_rows(chosen_files: list[str], label: str, fine_label: str | None = None) -> list[dict]:
    rows = []
    for fp in chosen_files:
        row = {
            "file_name":  os.path.basename(fp),
            "full_path":  fp,
            "label":      label,
            "fine_label": fine_label if fine_label else label,
        }
        rows.append(row)
    return rows


# ------------------------------------------------------------------
# Task builders
# ------------------------------------------------------------------

def build_5_class(fine_files: dict, rng: random.Random):
    """
    5-class classification test subset.
    Draws N_5_CLASS samples per high-level class, sampling uniformly
    across all contributing fine-grained sub-folders.
    """
    print("\n[Task] 5_class")
    for high_label, fine_labels in HIGH_LEVEL_GROUPS.items():
        # Pool all fine-grained files that belong to this high-level class
        pool = []
        for fl in fine_labels:
            for fp in fine_files.get(fl, []):
                pool.append((fp, fl))

        chosen = rng.sample(pool, min(N_5_CLASS, len(pool)))
        rows = []
        for fp, fl in chosen:
            rows.append({
                "file_name":  os.path.basename(fp),
                "full_path":  fp,
                "label":      high_label,
                "fine_label": fl,
            })
        save_subset(rows, "5_class", high_label)


def build_9_class(fine_files: dict, rng: random.Random):
    """
    9-class classification test subset.
    Draws N_9_CLASS per fine-grained label, collapsing the five deterministic
    trend sub-types into a single 'Deterministic Trend' label.
    """
    print("\n[Task] 9_class")
    # Labels in 9_class:
    #   Deterministic Trend (pooled from 5 sub-types), Stochastic Trend,
    #   Mean Shift, Variance Shift, Trend Shift, Volatility,
    #   Point Anomaly, Collective Anomaly, Contextual Anomaly
    deterministic_fine = [
        "Linear Trend", "Quadratic Trend", "Cubic Trend",
        "Exponential Trend", "Damped Trend"
    ]
    single_labels = [
        "Stochastic Trend", "Mean Shift", "Variance Shift", "Trend Shift",
        "Volatility", "Point Anomaly", "Collective Anomaly", "Contextual Anomaly"
    ]

    # Deterministic Trend (pooled)
    pool = [(fp, fl)
            for fl in deterministic_fine
            for fp in fine_files.get(fl, [])]
    chosen = rng.sample(pool, min(N_9_CLASS, len(pool)))
    rows = [{"file_name": os.path.basename(fp), "full_path": fp,
             "label": "Deterministic Trend", "fine_label": fl}
            for fp, fl in chosen]
    save_subset(rows, "9_class", "Deterministic Trend")

    # All other single-mapped labels
    for label in single_labels:
        pool = fine_files.get(label, [])
        chosen = sample_files(pool, N_9_CLASS, rng)
        rows = make_rows(chosen, label)
        save_subset(rows, "9_class", label)


def build_trend_5_class(fine_files: dict, rng: random.Random):
    """trend_5_class: one subset per deterministic trend subtype."""
    print("\n[Task] trend_5_class")
    for label in ["Linear Trend", "Quadratic Trend", "Cubic Trend",
                  "Exponential Trend", "Damped Trend"]:
        pool = fine_files.get(label, [])
        chosen = sample_files(pool, N_TREND_5, rng)
        rows = make_rows(chosen, label)
        save_subset(rows, "trend_5_class", label)


def build_break_3_class(fine_files: dict, rng: random.Random):
    """break_3_class: one subset per structural break subtype."""
    print("\n[Task] break_3_class")
    for label in ["Mean Shift", "Variance Shift", "Trend Shift"]:
        pool = fine_files.get(label, [])
        chosen = sample_files(pool, N_BREAK_3, rng)
        rows = make_rows(chosen, label)
        save_subset(rows, "break_3_class", label)


def build_anomaly_3_class(fine_files: dict, rng: random.Random):
    """anomaly_3_class: one subset per anomaly subtype."""
    print("\n[Task] anomaly_3_class")
    for label in ["Point Anomaly", "Collective Anomaly", "Contextual Anomaly"]:
        pool = fine_files.get(label, [])
        chosen = sample_files(pool, N_ANOMALY_3, rng)
        rows = make_rows(chosen, label)
        save_subset(rows, "anomaly_3_class", label)


def build_2_class_stationarity(fine_files: dict, rng: random.Random):
    """
    2_class_stationarity: M_STATIONARITY samples per class.
    Stationary  → Volatility + Anomaly sub-types (I(0) processes).
    Non-Stationary → all trend and structural-break sub-types.
    """
    print("\n[Task] 2_class_stationarity")
    for stat_label, fine_labels in STATIONARITY_GROUPS.items():
        pool = [(fp, fl)
                for fl in fine_labels
                for fp in fine_files.get(fl, [])]
        chosen = rng.sample(pool, min(M_STATIONARITY, len(pool)))
        rows = [{"file_name": os.path.basename(fp), "full_path": fp,
                 "label": stat_label, "fine_label": fl}
                for fp, fl in chosen]
        save_subset(rows, "2_class_stationarity", stat_label)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def create_test_subsets():
    rng = random.Random(RANDOM_SEED)
    os.makedirs(DEST_DIR, exist_ok=True)

    # Load few-shot pool paths FIRST so we can exclude them everywhere
    few_shot_paths = load_few_shot_paths(FEW_SHOT_POOL_DIR)

    fine_files = collect_fine_grained_files(few_shot_paths)

    build_5_class(fine_files, rng)
    build_9_class(fine_files, rng)
    build_trend_5_class(fine_files, rng)
    build_break_3_class(fine_files, rng)
    build_anomaly_3_class(fine_files, rng)
    build_2_class_stationarity(fine_files, rng)

    print("\nAll test subsets ready.")


if __name__ == "__main__":
    create_test_subsets()