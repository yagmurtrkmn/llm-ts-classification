import os
import random
import pandas as pd
from tqdm import tqdm

# ================= CONFIG =================
SOURCE_DIR = "./data/raw"
DEST_DIR = "./data/few_shot_example_pool"
MAX_SERIES_LENGTH = 500
RANDOM_SEED = 42

# örnek quota (istersen değiştir)
SAMPLES_PER_CLASS = {
    "Linear Trend": 15,
    "Quadratic Trend": 15,
    "Cubic Trend": 15,
    "Exponential Trend": 15,
    "Damped Trend": 15,
    "Stochastic Trend": 75,
    "Mean Shift": 25,
    "Variance Shift": 25,
    "Trend Shift": 25,
    "Volatility": 75,
    "Point Anomaly": 25,
    "Collective Anomaly": 25,
    "Contextual Anomaly": 25,
}

FOLDER_MAPPING = {
    "deterministic_trend/linear_trend": "Linear Trend",
    "deterministic_trend/quadratic_trend": "Quadratic Trend",
    "deterministic_trend/cubic_trend": "Cubic Trend",
    "deterministic_trend/exponential_trend": "Exponential Trend",
    "deterministic_trend/damped_trend": "Damped Trend",
    "stochastic_trend": "Stochastic Trend",
    "structural_break/mean_shift": "Mean Shift",
    "structural_break/variance_shift": "Variance Shift",
    "structural_break/trend_shift": "Trend Shift",
    "volatility": "Volatility",
    "anomaly/point_anomaly": "Point Anomaly",
    "anomaly/collective_anomaly": "Collective Anomaly",
    "anomaly/contextual_anomaly": "Contextual Anomaly",
}
# ==========================================


def is_valid_length(filepath, max_len):
    try:
        df = pd.read_csv(filepath, usecols=[0])
        return len(df) <= max_len
    except:
        return False


def collect_valid_files():
    class_files = {}

    print("Scanning & filtering...")
    for raw_name, nice_name in tqdm(FOLDER_MAPPING.items()):
        folder = os.path.join(SOURCE_DIR, raw_name)
        if not os.path.isdir(folder):
            continue

        valid = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".csv") and f != "metadata.csv":
                    fp = os.path.join(root, f)
                    if is_valid_length(fp, MAX_SERIES_LENGTH):
                        valid.append(fp)

        class_files[nice_name] = valid

    return class_files


def create_pool():
    random.seed(RANDOM_SEED)
    os.makedirs(DEST_DIR, exist_ok=True)

    class_files = collect_valid_files()

    print("Sampling...")
    for cls, files in class_files.items():
        if len(files) == 0:
            print(f"No files for {cls}")
            continue

        quota = SAMPLES_PER_CLASS[cls]
        chosen = random.sample(files, min(quota, len(files)))

        class_dir = os.path.join(DEST_DIR, "", cls)
        os.makedirs(class_dir, exist_ok=True)

        rows = []
        for fp in chosen:
            rows.append({
                "file_name": os.path.basename(fp),
                "full_path": fp,
                "label": cls,
                "description": f"There is a {cls} in this series."   # burayı sonra dolduracaksın
            })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(class_dir, "samples.csv"), index=False)

        print(f"{cls}: {len(rows)}")

    print("Pool hazır.")


if __name__ == "__main__":
    create_pool()
