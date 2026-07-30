"""
app.py
-------
Streamlit dashboard: a security analyst pastes/uploads traffic feature values
and gets an instant threat classification from the hybrid ensemble, along
with a confidence score and a plain-language explanation.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

MODEL_DIR = "models"

st.set_page_config(page_title="AI Cybersecurity Threat Detector", layout="wide")

@st.cache_resource
def load_all_models():
    rf = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
    xgb = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    lstm = load_model(f"{MODEL_DIR}/lstm_model.keras")
    meta_learner = joblib.load(f"{MODEL_DIR}/meta_learner.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
    target_encoder = joblib.load(f"{MODEL_DIR}/target_encoder.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    return rf, xgb, lstm, meta_learner, scaler, encoders, target_encoder, feature_cols


rf, xgb, lstm, meta_learner, scaler, encoders, target_encoder, feature_cols = load_all_models()

st.title("🛡️ AI-Powered Cybersecurity Threat Detection")
st.caption("Hybrid Stacked Ensemble: Random Forest + XGBoost + LSTM + Meta-Learner")

tab1, tab2 = st.tabs(["Upload Traffic CSV", "Manual Single-Record Check"])

with tab1:
    uploaded = st.file_uploader("Upload a CSV of raw network traffic records (NSL-KDD format)", type="csv")
    if uploaded:
        raw_df = pd.read_csv(uploaded)
        df = raw_df.copy()
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))
        df[feature_cols] = scaler.transform(df[feature_cols])

        rf_probs = rf.predict_proba(df[feature_cols])
        xgb_probs = xgb.predict_proba(df[feature_cols])
        lstm_input = df[feature_cols].values.reshape((df.shape[0], len(feature_cols), 1))
        lstm_probs = lstm.predict(lstm_input)

        meta_features = np.hstack([rf_probs, xgb_probs, lstm_probs])
        preds = meta_learner.predict(meta_features)
        confidences = meta_learner.predict_proba(meta_features).max(axis=1)

        raw_df["Predicted_Threat"] = target_encoder.inverse_transform(preds)
        raw_df["Confidence"] = (confidences * 100).round(2).astype(str) + "%"

        st.success(f"Processed {len(raw_df)} records.")
        st.dataframe(raw_df[["Predicted_Threat", "Confidence"] +
                             [c for c in raw_df.columns if c not in ["Predicted_Threat", "Confidence"]]])

        threat_counts = raw_df["Predicted_Threat"].value_counts()
        st.bar_chart(threat_counts)

with tab2:
    st.write("Enter key traffic features manually for a quick single-record check:")
    col1, col2, col3 = st.columns(3)
    with col1:
        duration = st.number_input("Duration", 0, 100000, 0)
        src_bytes = st.number_input("Source Bytes", 0, 1000000, 0)
        dst_bytes = st.number_input("Destination Bytes", 0, 1000000, 0)
    with col2:
        protocol_type = st.selectbox("Protocol Type", options=list(encoders["protocol_type"].classes_))
        service = st.selectbox("Service", options=list(encoders["service"].classes_))
        flag = st.selectbox("Flag", options=list(encoders["flag"].classes_))
    with col3:
        count = st.number_input("Connection Count", 0, 1000, 1)
        serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, 0.0)
        same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 1.0)

    if st.button("Analyze Traffic"):
        st.info("This quick-check tool only demonstrates the interface using a handful of "
                "the 41 required features — use the CSV upload tab for full accuracy.")
