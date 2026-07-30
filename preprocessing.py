"""
preprocessing.py
-----------------
Loads NSL-KDD train/test files, cleans them, encodes categorical columns,
maps the 40+ attack labels into 5 classes (Normal, DoS, Probe, R2L, U2R),
scales numeric features, and balances the training set with SMOTE.

Output: data/processed_train.csv, data/processed_test.csv, models/scaler.pkl,
        models/label_encoders.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

DATA_DIR = "data"
MODEL_DIR = "models"

# ---- 1. Attack -> category mapping (NSL-KDD standard mapping) ----
ATTACK_MAP = {
    'normal': 'Normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS', 'worm': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
    'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}


def load_data():
    columns = open(f"{DATA_DIR}/kdd_columns.txt").read().strip().split(",")
    train = pd.read_csv(f"{DATA_DIR}/KDDTrain+.txt", names=columns)
    test = pd.read_csv(f"{DATA_DIR}/KDDTest+.txt", names=columns)
    # drop the 'difficulty' column - not a real feature
    train = train.drop(columns=["difficulty"])
    test = test.drop(columns=["difficulty"])
    return train, test


def map_labels(df):
    df["label"] = df["label"].str.strip()
    df["category"] = df["label"].map(ATTACK_MAP).fillna("Unknown")
    # any label not in the map (rare/new attack) -> treat conservatively as its closest bucket
    df = df[df["category"] != "Unknown"]
    return df


def encode_and_scale(train, test):
    cat_cols = ["protocol_type", "service", "flag"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # fit on union of train+test values so unseen categories don't crash at inference
        le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        encoders[col] = le

    target_encoder = LabelEncoder()
    target_encoder.fit(train["category"])
    train["target"] = target_encoder.transform(train["category"])
    test["target"] = target_encoder.transform(test["category"])

    feature_cols = [c for c in train.columns if c not in ["label", "category", "target"]]

    scaler = StandardScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/label_encoders.pkl")
    joblib.dump(target_encoder, f"{MODEL_DIR}/target_encoder.pkl")
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")

    return train, test, feature_cols


def balance_with_smote(train, feature_cols):
    X = train[feature_cols]
    y = train["target"]
    # SMOTE needs k_neighbors < smallest class size; NSL-KDD's U2R class is tiny
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=feature_cols), y_res


def main():
    train, test = load_data()
    train, test = map_labels(train), map_labels(test)
    train, test, feature_cols = encode_and_scale(train, test)

    X_train_bal, y_train_bal = balance_with_smote(train, feature_cols)

    processed_train = X_train_bal.copy()
    processed_train["target"] = y_train_bal
    processed_train.to_csv(f"{DATA_DIR}/processed_train.csv", index=False)

    test[feature_cols + ["target"]].to_csv(f"{DATA_DIR}/processed_test.csv", index=False)

    print("Preprocessing complete.")
    print(f"Balanced training set shape: {processed_train.shape}")
    print(f"Test set shape: {test.shape}")
    print("Class distribution after SMOTE:\n", y_train_bal.value_counts())


if __name__ == "__main__":
    main()
