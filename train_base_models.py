"""
train_base_models.py
---------------------
Trains 3 base learners on the processed NSL-KDD training data:
  1. Random Forest      (strong on tabular splits/thresholds)
  2. XGBoost            (strong on complex non-linear feature interactions)
  3. LSTM               (captures sequential/temporal structure in the flow features)

To avoid leakage in the later stacking step, the balanced training set is split into:
  - base_train (70%)  -> used to fit the 3 base models
  - meta_train (30%)  -> used ONLY to generate out-of-sample predictions,
                         which become the input features for the meta-learner

Outputs: models/rf_model.pkl, models/xgb_model.pkl, models/lstm_model.keras,
         data/meta_train_features.csv, data/meta_train_labels.csv
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data"
MODEL_DIR = "models"

train = pd.read_csv(f"{DATA_DIR}/processed_train.csv")
feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
n_classes = train["target"].nunique()

X = train[feature_cols]
y = train["target"]

X_base, X_meta, y_base, y_meta = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------- 1. Random Forest ----------------
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
rf.fit(X_base, y_base)
joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl")
rf_meta_probs = rf.predict_proba(X_meta)

# ---------------- 2. XGBoost ----------------
print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.1,
    objective="multi:softprob", num_class=n_classes,
    eval_metric="mlogloss", n_jobs=-1, random_state=42
)
xgb.fit(X_base, y_base)
joblib.dump(xgb, f"{MODEL_DIR}/xgb_model.pkl")
xgb_meta_probs = xgb.predict_proba(X_meta)

# ---------------- 3. LSTM ----------------
print("Training LSTM...")
# reshape tabular features into a pseudo-sequence: (samples, timesteps=1, features)
# each feature is treated as one "timestep" so the LSTM learns dependencies across features
X_base_lstm = X_base.values.reshape((X_base.shape[0], X_base.shape[1], 1))
X_meta_lstm = X_meta.values.reshape((X_meta.shape[0], X_meta.shape[1], 1))
y_base_cat = to_categorical(y_base, num_classes=n_classes)

lstm_model = Sequential([
    LSTM(64, input_shape=(X_base.shape[1], 1), return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(n_classes, activation="softmax")
])
lstm_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
lstm_model.fit(
    X_base_lstm, y_base_cat,
    validation_split=0.1, epochs=15, batch_size=256, verbose=1
)
lstm_model.save(f"{MODEL_DIR}/lstm_model.keras")
lstm_meta_probs = lstm_model.predict(X_meta_lstm)

# ---------------- Save meta-features for stacking_ensemble.py ----------------
meta_features = np.hstack([rf_meta_probs, xgb_meta_probs, lstm_meta_probs])
pd.DataFrame(meta_features).to_csv(f"{DATA_DIR}/meta_train_features.csv", index=False)
y_meta.to_csv(f"{DATA_DIR}/meta_train_labels.csv", index=False)

print("Base model training complete. Meta-features saved for stacking.")
