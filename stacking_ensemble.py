"""
stacking_ensemble.py
----------------------
Trains the meta-learner (Logistic Regression) on the stacked probability
outputs of the 3 base models (RF, XGBoost, LSTM). This meta-learner learns
which base model to trust more for which type of traffic pattern -- this is
what makes it a *hybrid* ensemble rather than a simple majority vote.

Output: models/meta_learner.pkl
"""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

DATA_DIR = "data"
MODEL_DIR = "models"

X_meta = pd.read_csv(f"{DATA_DIR}/meta_train_features.csv")
y_meta = pd.read_csv(f"{DATA_DIR}/meta_train_labels.csv").values.ravel()

meta_learner = LogisticRegression(max_iter=1000, multi_class="multinomial")
meta_learner.fit(X_meta, y_meta)

joblib.dump(meta_learner, f"{MODEL_DIR}/meta_learner.pkl")
print("Meta-learner trained and saved to models/meta_learner.pkl")
print(f"Meta-learner training accuracy: {meta_learner.score(X_meta, y_meta):.4f}")
