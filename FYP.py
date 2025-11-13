# ============================================================
# PETRONAS-STYLE RESPONSIVE DASHBOARD (Streamlit)
# Author: Adrian Anthony A/L R. Vikneswaran (UTP)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from datetime import datetime  # <-- added for timestamp

# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------
st.set_page_config(
    page_title="Digital Wellbeing & Mental Health Risk Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------------
# STYLE SETTINGS
# ----------------------------------------
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #DFF7F9 0%, #F8FBFC 100%);
        color: #1E1E1E;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #007C91;
        font-weight: 700;
    }
    .main-card {
        background-color: white;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 20px 25px;
        margin-bottom: 20px;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        height: 50px;
    }
    div.stButton > button:first-child {
        background-color: #007C91;
        color: white;
    }
    div.stButton > button:first-child:hover {
        background-color: #005C68;
    }
    div.stButton > button:last-child {
        background-color: #E0E0E0;
        color: #333;
    }
    div.stButton > button:last-child:hover {
        background-color: #BDBDBD;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# HEADER
# ----------------------------------------
st.markdown("""
<div style='text-align:center;'>
    <h1>🧠 Digital Wellbeing & Mental Health Risk Dashboard</h1>
    <p style='font-size:18px; color:#475569;'>Universiti Teknologi PETRONAS</p>
    <hr style='border: 1px solid #CCE5E9;'>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------
# BACKEND EXCEL LOGGING
# ----------------------------------------
PRED_EXCEL_PATH = "dashboard_predictions.xlsx"

def save_prediction_to_excel(record: dict, excel_path: str = PRED_EXCEL_PATH):
    """Append a prediction record to an Excel file (backend only, invisible to user)."""
    df_new = pd.DataFrame([record])

    if os.path.exists(excel_path):
        try:
            df_existing = pd.read_excel(excel_path)
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_out = df_new
    else:
        df_out = df_new

    df_out.to_excel(excel_path, index=False, engine="openpyxl")

# ----------------------------------------
# LOAD DATA & MODEL
# ----------------------------------------
data_path = "digital_wellbeing_dataset_binned.csv"
model_path = "RandomForest_best_pipeline.joblib"

df = pd.read_csv(data_path)
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("⚠️ Model file not found! Please make sure RandomForest_best_pipeline.joblib is in this folder.")
    st.stop()

target_col = "mh_risk"

# ----------------------------------------
# ALIGNMENT FUNCTION
# ----------------------------------------
def align_features(input_df, model):
    try:
        expected = model.named_steps['preproc'].feature_names_in_
        for f in expected:
            if f not in input_df.columns:
                input_df[f] = 0
        return input_df[expected]
    except Exception:
        return input_df

# ----------------------------------------
# TAB SETUP
# ----------------------------------------
tabs = st.tabs([
    "📘 Dataset Overview",
    "📊 Correlation Analysis",
    "📈 Feature Importance",
    "🧮 Model Evaluation",
    "🤖 Prediction Panel"
])

# TAB 1: Dataset Overview
with tabs[0]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("📘 Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Class Distribution")
    fig = px.pie(df, names=target_col, hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 2: Correlation
with tabs[1]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("📊 Correlation Analysis")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="BuGn")
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 3: Feature Importance
with tabs[2]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("📈 Feature Importance")
    feat_names = model.named_steps['preproc'].get_feature_names_out()
    feat_vals = model.named_steps['clf'].feature_importances_
    feat_df = pd.DataFrame({"Feature": feat_names, "Importance": feat_vals})
    feat_df = feat_df.sort_values("Importance", ascending=False).head(15)
    fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 4: Model Evaluation
with tabs[3]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("🧮 Model Evaluation")

    X = align_features(df.drop(columns=[target_col]), model)
    y_true = df[target_col]
    y_pred = model.predict(X)

    st.metric("Model Accuracy", f"{accuracy_score(y_true, y_pred)*100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="crest", fmt="g")
    st.pyplot(fig)

    st.dataframe(pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)))
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 5: Prediction Panel
with tabs[4]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("🤖 Prediction Panel")

    original_features = [
        'Daily_Screen_Time_Hours',
        'Phone_Unlocks_Per_Day',
        'Social_Media_Usage_Hours',
        'Gaming_Usage_Hours',
        'Streaming_Usage_Hours',
        'Messaging_Usage_Hours',
        'Sleep_Hours',
        'Physical_Activity_Hours',
        'Stress_Level',
        'Self_Reported_Addiction_Level'
    ]

    user_input = {}
    col1, col2 = st.columns(2)

    for i, col in enumerate(original_features):
        if df[col].dtype in ['int64', 'float64']:
            min_v, max_v, def_v = df[col].min(), df[col].max(), df[col].mean()
            widget = col1.slider if i % 2 == 0 else col2.slider
            user_input[col] = widget(col, float(min_v), float(max_v), float(def_v))
        else:
            widget = col1.selectbox if i % 2 == 0 else col2.selectbox
            user_input[col] = widget(col, df[col].unique().tolist())

    if st.button("🔍 Predict Mental Health Risk"):
        user_df = pd.DataFrame([user_input])
        user_df_aligned = align_features(user_df.copy(), model)
        prediction = model.predict(user_df_aligned)[0]
        probs = model.predict_proba(user_df_aligned)[0]

        st.subheader(f"Predicted Risk: {prediction}")

        # Risk Message
        if prediction == "High":
            st.error("⚠️ High Risk — Seek professional guidance and review your habits.")
        elif prediction == "Medium":
            st.warning("🟡 Moderate Risk — Balance your digital and daily routines.")
        else:
            st.success("🟢 Low Risk — Your digital habits are healthy!")

        # Backend Excel Logging
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **user_input,
            "prediction": prediction,
            "proba_low": probs[0],
            "proba_medium": probs[1],
            "proba_high": probs[2]
        }
        save_prediction_to_excel(record)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------
# FOOTER (UPDATED)
# ----------------------------------------
st.markdown("""
---
<div style='text-align:center; font-size:14px; color:#475569;'>
    📊 Dashboard created by <strong>Adrian Anthony (UTP)</strong><br>
    Built with <strong>Streamlit</strong> | Random Forest Model | Responsive UI Design
</div>
---
""", unsafe_allow_html=True)
