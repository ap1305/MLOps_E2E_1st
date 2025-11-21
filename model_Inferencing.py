import mlflow

mlflow.set_tracking_uri("https://dagshub.com/ap1305/MLOps_E2E_1st.mlflow")

local_path = mlflow.artifacts.download_artifacts(
    artifact_uri="mlflow-artifacts:/edf06d9bb69748a2920e5a0f07d5e011/09baf419dfde490eb1ac0593e8adb0e1/artifacts/best_model/best_model.pkl"
)

print("Downloaded to:", local_path)
import pickle

with open(local_path, "rb") as f:
    model = pickle.load(f)

print(model.predic())