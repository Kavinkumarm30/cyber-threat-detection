"""
stacking_ensemble.py
----------------------
Trains the meta-learner on the stacked probability outputs of the 3 base
models (RF, XGBoost, LSTM).

Uses GradientBoostingClassifier with strong regularization to prevent
overfitting on the 15-dimensional meta-feature space. The previous version
achieved 100% training accuracy (overfitting), resulting in worse test
performance than individual models. This version uses:
  - Shallower trees (max_depth=2)
  - Fewer estimators (100)
  - Minimum leaf samples (50)
  - Feature subsampling (sqrt)
These constraints force the meta-learner to learn generalizable combination
rules rather than memorizing the training set.

Output: models/meta_learner.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import cross_val_score

DATA_DIR = "data"
MODEL_DIR = "models"


def main():
    X_meta = pd.read_csv(f"{DATA_DIR}/meta_train_features.csv")
    y_meta = pd.read_csv(f"{DATA_DIR}/meta_train_labels.csv").values.ravel()

    # Compute sample weights to handle class imbalance in meta-learner training
    sample_weights = compute_sample_weight("balanced", y_meta)

    # GradientBoosting meta-learner with STRONG regularization to prevent
    # overfitting on the 15-feature meta-space. Previous version (max_depth=4,
    # n_estimators=200) got 100% train accuracy = overfitting.
    meta_learner = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=50,
        max_features="sqrt",
        random_state=42
    )
    meta_learner.fit(X_meta, y_meta, sample_weight=sample_weights)

    # Cross-validation sanity check to detect overfitting
    cv_scores = cross_val_score(meta_learner, X_meta, y_meta, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    joblib.dump(meta_learner, f"{MODEL_DIR}/meta_learner.pkl")
    print("Meta-learner (GradientBoosting) trained and saved to models/meta_learner.pkl")
    print(f"Meta-learner training accuracy: {meta_learner.score(X_meta, y_meta):.4f}")

    # Per-class accuracy diagnostics
    preds = meta_learner.predict(X_meta)
    target_encoder = joblib.load(f"{MODEL_DIR}/target_encoder.pkl")
    for cls_idx in np.unique(y_meta):
        mask = y_meta == cls_idx
        cls_acc = (preds[mask] == y_meta[mask]).mean()
        cls_name = target_encoder.inverse_transform([int(cls_idx)])[0]
        print(f"  {cls_name}: {cls_acc:.4f} ({mask.sum()} samples)")


if __name__ == "__main__":
    main()
