import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset
from prometheus_client import start_http_server, Gauge
import time

# Load training data
train_df = pd.read_csv("winequality-red.csv")  # Replace with your training CSV path

# Prometheus metric
drift_metric = Gauge("data_drift_detected", "1 if data drift detected else 0")

def check_drift():
    try:
        prod_df = pd.read_csv("prod_data.csv")  # Production logs
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=train_df, current_data=prod_df)
        result = report.as_dict()
        # Check if any feature drifted
        any_drift = any([f["drift_detected"] for f in result["metrics"][0]["result"]["metrics"]])
        drift_metric.set(1 if any_drift else 0)
        print("Drift detected:", any_drift)
    except Exception as e:
        print("Error checking drift:", e)

if __name__ == "__main__":
    start_http_server(8001)  # Expose Prometheus metrics
    print("Drift exporter running on port 8001...")
    while True:
        check_drift()
        time.sleep(3600)  # Run every hour
