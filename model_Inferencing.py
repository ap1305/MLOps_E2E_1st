import mlflow
mlflow.set_tracking_uri("https://dagshub.com/ap1305/MLOps_E2E_1st.mlflow")
import mlflow
model=mlflow.pyfunc.load_model("mlflow-artifacts:/edf06d9bb69748a2920e5a0f07d5e011/09baf419dfde490eb1ac0593e8adb0e1/artifacts/best_model/best_model.pkl")
