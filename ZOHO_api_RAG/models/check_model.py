import joblib
model = joblib.load(r"Datasets\hallucination_risk_model.joblib")
print("n_features_in_:", model.n_features_in_)
print("coef_:", model.coef_)
print("intercept_:", model.intercept_)