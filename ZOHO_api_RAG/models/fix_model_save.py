# fix_model_save.py
import joblib
data = joblib.load("Datasets/hallucination_risk_model_best_v3.joblib")
model = data["model"] if isinstance(data, dict) else data
joblib.dump(model, "Datasets/hallucination_risk_model_best_v3_fixed.joblib")
print("Fixed model saved as hallucination_risk_model_best_v3_fixed.joblib")