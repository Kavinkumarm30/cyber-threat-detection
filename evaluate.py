"""
evaluate.py
------------
Runs the FULL hybrid pipeline (RF -> XGB -> LSTM -> meta-learner) on the
untouched NSL-KDD test set (KDDTest+.txt) and reports final performance.

Also compares against each base model alone, to prove the hybrid ensemble
adds value over any single model -- this comparison table is the core
"innovation justification" evidence for your report.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
MODEL_DIR = "models"

test = pd.read_csv(f"{DATA_DIR}/processed_test.csv")
feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
target_encoder = joblib.load(f"{MODEL_DIR}/target_encoder.pkl")

X_test = test[feature_cols]
y_test = test["target"]
n_classes = len(target_encoder.classes_)

rf = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
xgb = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
lstm = load_model(f"{MODEL_DIR}/lstm_model.keras")
meta_learner = joblib.load(f"{MODEL_DIR}/meta_learner.pkl")

rf_probs = rf.predict_proba(X_test)
xgb_probs = xgb.predict_proba(X_test)
X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
lstm_probs = lstm.predict(X_test_lstm)

meta_features_test = np.hstack([rf_probs, xgb_probs, lstm_probs])
hybrid_preds = meta_learner.predict(meta_features_test)


def report_model(name, preds, probs=None):
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted")
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


results = []
results.append(report_model("Random Forest (alone)", rf.predict(X_test)))
results.append(report_model("XGBoost (alone)", xgb.predict(X_test)))
results.append(report_model("LSTM (alone)", np.argmax(lstm_probs, axis=1)))
results.append(report_model("HYBRID Stacked Ensemble (RF+XGB+LSTM+Meta)", hybrid_preds))

results_df = pd.DataFrame(results)
results_df.to_csv(f"{DATA_DIR}/model_comparison.csv", index=False)
print("\nFull comparison table saved to data/model_comparison.csv")
print(results_df)

# ---- Detailed classification report for the hybrid model ----
print("\nDetailed classification report (Hybrid model):")
print(classification_report(y_test, hybrid_preds, target_names=target_encoder.classes_))

# ---- Confusion matrix plot ----
cm = confusion_matrix(y_test, hybrid_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hybrid Ensemble - Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/confusion_matrix.png", dpi=150)
print("Confusion matrix saved to data/confusion_matrix.png")
