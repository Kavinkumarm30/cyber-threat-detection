# AI-Powered Cybersecurity Threat Detection using a Hybrid ML+DL Stacked Ensemble

## 1. Overview
This project detects network intrusions (DoS, Probe, R2L, U2R, Normal) using a
**stacked hybrid ensemble** that combines two classical ML models (Random Forest,
XGBoost) with a Deep Learning model (LSTM), fused by a meta-learner. This hybrid
architecture typically beats any single model because it combines:
- RF/XGBoost → strong on tabular, structured, non-sequential features
- LSTM → captures temporal/sequential patterns in traffic flows
- Meta-learner (Logistic Regression) → learns *when to trust which base model*

## 2. Dataset
**NSL-KDD** (Kaggle: search "NSL-KDD dataset"). Download these two files into `data/`:
- `KDDTrain+.txt`
- `KDDTest+.txt`

Column names are provided in `data/kdd_columns.txt` (already included).

## 3. Project Structure
```
cyber_threat_detection/
├── data/
│   ├── kdd_columns.txt
│   ├── KDDTrain+.txt        <- download from Kaggle
│   └── KDDTest+.txt         <- download from Kaggle
├── models/                  <- trained models saved here
├── preprocessing.py
├── train_base_models.py
├── stacking_ensemble.py
├── evaluate.py
├── explainability.py
├── app.py                   <- Streamlit dashboard
├── requirements.txt
└── README.md
```

## 4. How to Run (in order)
```bash
pip install -r requirements.txt

python preprocessing.py          # cleans, encodes, scales, balances data
python train_base_models.py      # trains RF, XGBoost, LSTM
python stacking_ensemble.py      # trains meta-learner on base model outputs
python evaluate.py               # final metrics + confusion matrix + ROC-AUC
python explainability.py         # SHAP plots for explainability
streamlit run app.py             # live interactive threat detector
```

## 5. Innovation Points (for your report / viva)
1. **Hybrid stacked ensemble** (not just voting/averaging) — a meta-learner
   learns optimal combination weights instead of fixed rules.
2. **Explainable AI (SHAP)** — every alert shows *why* it was flagged, which
   is critical for real SOC (Security Operations Center) adoption.
3. **Class-imbalance handling with SMOTE** — rare attack types (U2R, R2L) are
   heavily under-represented in NSL-KDD; SMOTE prevents the model from ignoring them.
4. **Deployable dashboard** — a security analyst can paste/upload traffic
   features and get an instant classification + confidence + explanation.
