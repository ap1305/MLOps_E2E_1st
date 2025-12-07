import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from prometheus_client import start_http_server, Gauge
import time
import os

TRAIN_DATA_PATH = "/app/winequality-red.csv"
PROD_DATA_PATH = "/app/prod_data.csv"

if not os.path.exists(TRAIN_DATA_PATH):
    print("Training data not found:", TRAIN_DATA_PATH)
    exit(1)
train_df = pd.read_csv(TRAIN_DATA_PATH)

drift_metric = Gauge("data_drift_detected", "1 if any drift detected else 0")

def check_drift():
    if not os.path.exists(PROD_DATA_PATH):
        print("Production data missing:", PROD_DATA_PATH)
        drift_metric.set(0)
        return

    prod_df = pd.read_csv(PROD_DATA_PATH)

    # --- FIX STARTS HERE ---
    # Ensure both dataframes have the exact same columns
    # by selecting only the columns present in the original training data
    common_columns = train_df.columns.tolist()

    # Check if production data has all expected columns
    if not all(col in prod_df.columns for col in common_columns):
        missing_cols = set(common_columns) - set(prod_df.columns)
        print(f"Warning: Production data is missing expected columns: {missing_cols}. Drift check skipped.")
        drift_metric.set(0)
        return

    # Align production dataframe to have the same column order and subset as training data
    prod_df = prod_df[common_columns]

    # Optional: ensure data types match if you run into further TypeErrors
    # prod_df = prod_df.astype(train_df.dtypes)
    # --- FIX ENDS HERE ---

    # Initialize a Report with the DataDriftPreset
    report = Report(metrics=[
        DataDriftPreset(),
    ])

    # Run drift calculation
    # Both inputs now have identical schemas
    report.run(reference_data=train_df, current_data=prod_df)

    result = report.as_dict()

    try:
        any_drift = result['metrics']['result']['dataset_drift']
    except (KeyError, IndexError):
        any_drift = False
        print("Could not determine drift from report structure, assuming no drift.")

    drift_metric.set(1 if any_drift else 0)
    print(f"Drift detected: {any_drift}")

if __name__ == "__main__":
    start_http_server(8001)
    print("Drift exporter running on port 8001...")

    while True:
        check_drift()
        time.sleep(60)