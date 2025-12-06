import pandas as pd
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

    # Initialize preset
    preset = DataDriftPreset()
    # Run drift calculation
    preset.calculate(train_df, prod_df)

    # preset.result() contains a dictionary with results
    result = preset.result()
    # result['metrics'] contains all drift info
    metrics = result['metrics']

    # Detect if any column drifted
    any_drift = any(metric['result']['drift_detected'] for metric in metrics)
    drift_metric.set(1 if any_drift else 0)
    print("Drift detected:", any_drift)

if __name__ == "__main__":
    start_http_server(8001)
    print("Drift exporter running on port 8001...")

    while True:
        check_drift()
        time.sleep(60)
