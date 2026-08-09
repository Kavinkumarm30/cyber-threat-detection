"""
app.py
-------
Streamlit dashboard: a security analyst pastes/uploads traffic feature values
and gets an instant threat classification from the hybrid ensemble, along
with a confidence score and a plain-language explanation.

The Manual Single-Record Check tab now supports ALL 41 NSL-KDD features,
organized into logical groups, with full prediction pipeline execution.

Run with: streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Fix unpickling compatibility across sklearn versions on Cloud environments
try:
    import sklearn.ensemble._gb
    import sklearn._loss
    if "_loss" not in sys.modules and hasattr(sklearn, "_loss"):
        sys.modules["_loss"] = sklearn._loss
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

MODEL_DIR = "models"

st.set_page_config(page_title="AI Cybersecurity Threat Detector", layout="wide")

def load_lstm_safely(feature_cols_count, num_classes=5):
    # Try direct load with compile=False
    try:
        return load_model(f"{MODEL_DIR}/lstm_model.keras", compile=False)
    except Exception:
        pass

    try:
        return load_model(f"{MODEL_DIR}/lstm_model.h5", compile=False)
    except Exception:
        pass

    # Fallback: Reconstruct architecture and load weights
    model = Sequential([
        LSTM(64, input_shape=(feature_cols_count, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    if os.path.exists(f"{MODEL_DIR}/lstm.weights.h5"):
        model.load_weights(f"{MODEL_DIR}/lstm.weights.h5")
    elif os.path.exists(f"{MODEL_DIR}/lstm_weights.h5"):
        model.load_weights(f"{MODEL_DIR}/lstm_weights.h5")
    return model


@st.cache_resource
def load_all_models():
    rf = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
    xgb = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    meta_learner = joblib.load(f"{MODEL_DIR}/meta_learner.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
    target_encoder = joblib.load(f"{MODEL_DIR}/target_encoder.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    lstm = load_lstm_safely(len(feature_cols), len(target_encoder.classes_))
    return rf, xgb, lstm, meta_learner, scaler, encoders, target_encoder, feature_cols


rf, xgb, lstm, meta_learner, scaler, encoders, target_encoder, feature_cols = load_all_models()

# ---- Threat descriptions for display ----
THREAT_INFO = {
    "Normal": {
        "color": "green",
        "icon": "✅",
        "severity": "None",
        "description": "Normal network traffic. No threat detected."
    },
    "DoS": {
        "color": "orange",
        "icon": "⚠️",
        "severity": "High",
        "description": "Denial of Service attack — flooding the target with excessive traffic to exhaust resources and deny legitimate access."
    },
    "Probe": {
        "color": "orange",
        "icon": "🔍",
        "severity": "Medium",
        "description": "Probing/Surveillance attack — scanning the network to gather information about targets (open ports, running services, OS fingerprinting)."
    },
    "R2L": {
        "color": "red",
        "icon": "🔴",
        "severity": "Critical",
        "description": "Remote-to-Local attack — unauthorized access from a remote machine, exploiting vulnerabilities to gain local user privileges."
    },
    "U2R": {
        "color": "red",
        "icon": "🚨",
        "severity": "Critical",
        "description": "User-to-Root privilege escalation — a local user exploiting system vulnerabilities to gain root/admin privileges."
    }
}


def run_hybrid_prediction(input_df):
    """Run the full hybrid ensemble pipeline on a prepared DataFrame."""
    rf_probs = rf.predict_proba(input_df)
    xgb_probs = xgb.predict_proba(input_df)
    lstm_input = input_df.values.reshape((input_df.shape[0], len(feature_cols), 1))
    lstm_probs = lstm.predict(lstm_input)

    meta_features = np.hstack([rf_probs, xgb_probs, lstm_probs])
    preds = meta_learner.predict(meta_features)
    confidences = meta_learner.predict_proba(meta_features).max(axis=1)

    return preds, confidences, rf_probs, xgb_probs, lstm_probs


st.title("🛡️ AI-Powered Cybersecurity Threat Detection")
st.caption("Hybrid Stacked Ensemble: Random Forest + XGBoost + LSTM + Meta-Learner")

tab1, tab2 = st.tabs(["📁 Upload Traffic CSV", "🔧 Manual Single-Record Check"])

with tab1:
    uploaded = st.file_uploader("Upload a CSV of raw network traffic records (NSL-KDD format)", type="csv")
    if uploaded:
        raw_df = pd.read_csv(uploaded)
        df = raw_df.copy()
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))
        df[feature_cols] = scaler.transform(df[feature_cols])

        preds, confidences, _, _, _ = run_hybrid_prediction(df[feature_cols])

        raw_df["Predicted_Threat"] = target_encoder.inverse_transform(preds)
        raw_df["Confidence"] = (confidences * 100).round(2).astype(str) + "%"

        st.success(f"Processed {len(raw_df)} records.")
        st.dataframe(raw_df[["Predicted_Threat", "Confidence"] +
                             [c for c in raw_df.columns if c not in ["Predicted_Threat", "Confidence"]]])

        threat_counts = raw_df["Predicted_Threat"].value_counts()
        st.bar_chart(threat_counts)

with tab2:
    st.write("Enter network traffic features for a single-record threat analysis:")
    st.caption("All 41 NSL-KDD features are available below, organized into logical groups.")

    # ---- Group 1: Basic Connection Features ----
    st.subheader("🔗 Basic Connection Features")
    g1_col1, g1_col2, g1_col3 = st.columns(3)
    with g1_col1:
        duration = st.number_input("Duration (seconds)", min_value=0, max_value=100000, value=0, key="duration")
        protocol_type = st.selectbox("Protocol Type", options=list(encoders["protocol_type"].classes_), key="protocol")
        service = st.selectbox("Service", options=list(encoders["service"].classes_), key="service")
    with g1_col2:
        flag = st.selectbox("Flag", options=list(encoders["flag"].classes_), key="flag")
        src_bytes = st.number_input("Source Bytes", min_value=0, max_value=10000000, value=0, key="src_bytes")
        dst_bytes = st.number_input("Destination Bytes", min_value=0, max_value=10000000, value=0, key="dst_bytes")
    with g1_col3:
        land = st.selectbox("Land (same src/dst host+port)", [0, 1], key="land")
        wrong_fragment = st.number_input("Wrong Fragments", min_value=0, max_value=3, value=0, key="wrong_frag")
        urgent = st.number_input("Urgent Packets", min_value=0, max_value=14, value=0, key="urgent")

    # ---- Group 2: Content Features ----
    st.subheader("📄 Content Features")
    g2_col1, g2_col2, g2_col3 = st.columns(3)
    with g2_col1:
        hot = st.number_input("Hot Indicators", min_value=0, max_value=100, value=0, key="hot")
        num_failed_logins = st.number_input("Failed Logins", min_value=0, max_value=5, value=0, key="failed_logins")
        logged_in = st.selectbox("Logged In", [0, 1], key="logged_in")
        num_compromised = st.number_input("Compromised Conditions", min_value=0, max_value=9000, value=0, key="compromised")
    with g2_col2:
        root_shell = st.selectbox("Root Shell Obtained", [0, 1], key="root_shell")
        su_attempted = st.selectbox("SU Attempted", [0, 1, 2], key="su_attempted")
        num_root = st.number_input("Root Accesses", min_value=0, max_value=8000, value=0, key="num_root")
        num_file_creations = st.number_input("File Creations", min_value=0, max_value=100, value=0, key="file_creations")
    with g2_col3:
        num_shells = st.number_input("Shell Prompts", min_value=0, max_value=5, value=0, key="shells")
        num_access_files = st.number_input("Access Control Files", min_value=0, max_value=10, value=0, key="access_files")
        num_outbound_cmds = st.number_input("Outbound Commands", min_value=0, max_value=1, value=0, key="outbound_cmds")
        is_host_login = st.selectbox("Is Host Login", [0, 1], key="host_login")
        is_guest_login = st.selectbox("Is Guest Login", [0, 1], key="guest_login")

    # ---- Group 3: Time-based Traffic Features ----
    st.subheader("⏱️ Time-based Traffic Features")
    g3_col1, g3_col2, g3_col3 = st.columns(3)
    with g3_col1:
        count = st.number_input("Connection Count (same host, 2s window)", min_value=0, max_value=511, value=1, key="count")
        srv_count = st.number_input("Service Count (same service, 2s window)", min_value=0, max_value=511, value=1, key="srv_count")
    with g3_col2:
        serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, 0.0, key="serror_rate")
        srv_serror_rate = st.slider("Service SYN Error Rate", 0.0, 1.0, 0.0, key="srv_serror_rate")
        rerror_rate = st.slider("REJ Error Rate", 0.0, 1.0, 0.0, key="rerror_rate")
    with g3_col3:
        srv_rerror_rate = st.slider("Service REJ Error Rate", 0.0, 1.0, 0.0, key="srv_rerror_rate")
        same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 1.0, key="same_srv_rate")
        diff_srv_rate = st.slider("Diff Service Rate", 0.0, 1.0, 0.0, key="diff_srv_rate")
        srv_diff_host_rate = st.slider("Service Diff Host Rate", 0.0, 1.0, 0.0, key="srv_diff_host_rate")

    # ---- Group 4: Host-based Traffic Features ----
    st.subheader("🖥️ Host-based Traffic Features")
    g4_col1, g4_col2, g4_col3 = st.columns(3)
    with g4_col1:
        dst_host_count = st.number_input("Dst Host Count", min_value=0, max_value=255, value=0, key="dst_host_count")
        dst_host_srv_count = st.number_input("Dst Host Srv Count", min_value=0, max_value=255, value=0, key="dst_host_srv_count")
        dst_host_same_srv_rate = st.slider("Dst Host Same Srv Rate", 0.0, 1.0, 0.0, key="dst_host_same_srv_rate")
        dst_host_diff_srv_rate = st.slider("Dst Host Diff Srv Rate", 0.0, 1.0, 0.0, key="dst_host_diff_srv_rate")
    with g4_col2:
        dst_host_same_src_port_rate = st.slider("Dst Host Same Src Port Rate", 0.0, 1.0, 0.0, key="dst_host_same_src_port_rate")
        dst_host_srv_diff_host_rate = st.slider("Dst Host Srv Diff Host Rate", 0.0, 1.0, 0.0, key="dst_host_srv_diff_host_rate")
        dst_host_serror_rate = st.slider("Dst Host SYN Error Rate", 0.0, 1.0, 0.0, key="dst_host_serror_rate")
    with g4_col3:
        dst_host_srv_serror_rate = st.slider("Dst Host Srv SYN Error Rate", 0.0, 1.0, 0.0, key="dst_host_srv_serror_rate")
        dst_host_rerror_rate = st.slider("Dst Host REJ Error Rate", 0.0, 1.0, 0.0, key="dst_host_rerror_rate")
        dst_host_srv_rerror_rate = st.slider("Dst Host Srv REJ Error Rate", 0.0, 1.0, 0.0, key="dst_host_srv_rerror_rate")

    # ---- Analyze Button ----
    if st.button("🔍 Analyze Traffic", type="primary", use_container_width=True):
        # Build the raw input record matching kdd_columns.txt order
        raw_record = {
            "duration": duration,
            "protocol_type": protocol_type,
            "service": service,
            "flag": flag,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "land": land,
            "wrong_fragment": wrong_fragment,
            "urgent": urgent,
            "hot": hot,
            "num_failed_logins": num_failed_logins,
            "logged_in": logged_in,
            "num_compromised": num_compromised,
            "root_shell": root_shell,
            "su_attempted": su_attempted,
            "num_root": num_root,
            "num_file_creations": num_file_creations,
            "num_shells": num_shells,
            "num_access_files": num_access_files,
            "num_outbound_cmds": num_outbound_cmds,
            "is_host_login": is_host_login,
            "is_guest_login": is_guest_login,
            "count": count,
            "srv_count": srv_count,
            "serror_rate": serror_rate,
            "srv_serror_rate": srv_serror_rate,
            "rerror_rate": rerror_rate,
            "srv_rerror_rate": srv_rerror_rate,
            "same_srv_rate": same_srv_rate,
            "diff_srv_rate": diff_srv_rate,
            "srv_diff_host_rate": srv_diff_host_rate,
            "dst_host_count": dst_host_count,
            "dst_host_srv_count": dst_host_srv_count,
            "dst_host_same_srv_rate": dst_host_same_srv_rate,
            "dst_host_diff_srv_rate": dst_host_diff_srv_rate,
            "dst_host_same_src_port_rate": dst_host_same_src_port_rate,
            "dst_host_srv_diff_host_rate": dst_host_srv_diff_host_rate,
            "dst_host_serror_rate": dst_host_serror_rate,
            "dst_host_srv_serror_rate": dst_host_srv_serror_rate,
            "dst_host_rerror_rate": dst_host_rerror_rate,
            "dst_host_srv_rerror_rate": dst_host_srv_rerror_rate,
        }

        input_df = pd.DataFrame([raw_record])

        # Encode categorical columns
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        # Scale all features
        input_df[feature_cols] = scaler.transform(input_df[feature_cols])

        # Run full hybrid ensemble prediction
        preds, confidences, rf_probs, xgb_probs, lstm_probs = run_hybrid_prediction(
            input_df[feature_cols]
        )

        predicted_class = target_encoder.inverse_transform(preds)[0]
        confidence = confidences[0] * 100
        threat = THREAT_INFO.get(predicted_class, THREAT_INFO["Normal"])

        # ---- Display Results ----
        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        # Main prediction card
        if predicted_class == "Normal":
            st.success(f"{threat['icon']} **{predicted_class}** — Confidence: **{confidence:.1f}%**")
        elif threat["severity"] == "Critical":
            st.error(f"{threat['icon']} **{predicted_class}** — Severity: **{threat['severity']}** — Confidence: **{confidence:.1f}%**")
        else:
            st.warning(f"{threat['icon']} **{predicted_class}** — Severity: **{threat['severity']}** — Confidence: **{confidence:.1f}%**")

        st.info(threat["description"])

        # Per-model breakdown
        st.subheader("📊 Per-Model Breakdown")
        model_breakdown = {}
        for i, cls_name in enumerate(target_encoder.classes_):
            model_breakdown[cls_name] = {
                "Random Forest": f"{rf_probs[0][i]*100:.1f}%",
                "XGBoost": f"{xgb_probs[0][i]*100:.1f}%",
                "LSTM": f"{lstm_probs[0][i]*100:.1f}%",
            }

        breakdown_df = pd.DataFrame(model_breakdown).T
        breakdown_df.index.name = "Class"
        st.dataframe(breakdown_df, use_container_width=True)

        # Show which model is most confident for the predicted class
        pred_idx = preds[0]
        model_confs = {
            "Random Forest": rf_probs[0][pred_idx],
            "XGBoost": xgb_probs[0][pred_idx],
            "LSTM": lstm_probs[0][pred_idx]
        }
        most_confident_model = max(model_confs, key=model_confs.get)
        st.caption(
            f"💡 **{most_confident_model}** was most confident about this prediction "
            f"({model_confs[most_confident_model]*100:.1f}% for {predicted_class})"
        )
