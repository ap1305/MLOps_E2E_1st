import pandas as pd
from evidently.model_monitoring import DataDriftMonitor
from evidently.model_monitoring.monitoring import Dataset
from prometheus_client import start_http_server, Gauge
import time

# Load training data
train_df = pd.read_csv("winequality-red.csv")  # Replace with your training CSV path

# Prometheus metric
drift_metric = Gauge("data_drift_detected", "1 if data drift detected else 0")

# Setup Evidently drift monitor
monitor = DataDriftMonitor()

def check_drift():
    try:
        prod_df = pd.read_csv("prod_data.csv")  # Production logs

        # Wrap reference and current data
        reference_data = Dataset(train_df)
        current_data = Dataset(prod_df)

        # Calculate drift
        result = monitor.calculate(reference_data, current_data)

        # Check if any feature drifted
        drift_metrics = result.metrics
        any_drift = any([feature["drift_detected"] for feature in drift_metrics["data_drift"]["metrics"]])
        
        # Set Prometheus metric
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
