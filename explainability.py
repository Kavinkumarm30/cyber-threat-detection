"""
explainability.py
-------------------
Generates SHAP (SHapley Additive exPlanations) plots for the Random Forest
and XGBoost base models, so every threat prediction can be explained in
terms of *which network features drove the decision* -- important for
real-world SOC (Security Operations Center) trust and adoption.
"""

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

DATA_DIR = "data"
MODEL_DIR = "models"

test = pd.read_csv(f"{DATA_DIR}/processed_test.csv")
feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
X_sample = test[feature_cols].sample(500, random_state=42)  # SHAP on a sample for speed

xgb = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_sample)

# Global feature importance (summary plot across all classes)
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/shap_summary.png", dpi=150)
plt.close()

print("SHAP summary plot saved to data/shap_summary.png")
print("Top features driving threat classification (use this in your report's",
      "'Explainable AI' section).")
