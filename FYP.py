# ============================================================
# PETRONAS-STYLE RESPONSIVE DASHBOARD (Streamlit)
# Author: Adrian Anthony A/L R. Vikneswaran (UTP)
# ============================================================
try:
    import joblib
except ModuleNotFoundError:
    import subprocess
    subprocess.run(["pip", "install", "joblib"])
    import joblib

import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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
    <p style='font-size:18px; color:#475569;'  Universiti Teknologi PETRONAS</p>
    <hr style='border: 1px solid #CCE5E9;'>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------
# LOAD DATA & MODEL
# ----------------------------------------
data_path = "digital_wellbeing_dataset_binned.csv"
model_path = "RandomForest_best_pipeline.joblib"

df = pd.read_csv(data_path)
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("⚠️ Model file not found! Please make sure RandomForest_best_pipeline.joblib is in the same folder.")
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

# ----------------------------------------
# TAB 1: Dataset Overview
# ----------------------------------------
with tabs[0]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("📘 Dataset Overview")
    st.write(f"Dataset shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(), use_container_width=True)
    st.subheader("Class Distribution")
    fig = px.pie(df, names=target_col, title="Mental Health Risk Distribution", hole=0.4,
                 color_discrete_sequence=px.colors.sequential.Teal)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------
# TAB 2: Correlation Analysis
# ----------------------------------------
with tabs[1]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("📊 Correlation Analysis")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='BuGn', fmt=".2f", linewidths=0.5)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------
# TAB 3: Feature Importance
# ----------------------------------------
with tabs[2]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("📈 Feature Importance (Random Forest)")
    try:
        feature_names = model.named_steps['preproc'].get_feature_names_out()
        importances = model.named_steps['clf'].feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=False).head(15)
        fig2 = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color='Importance',
                      color_continuous_scale='Tealgrn')
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------
# TAB 4: Model Evaluation
# ----------------------------------------
with tabs[3]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("🧮 Model Evaluation")
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y_true = df[target_col]
        X = align_features(X, model)
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        st.metric(label="Model Accuracy", value=f"{acc*100:.2f}%")

        cm = confusion_matrix(y_true, y_pred)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, cmap='crest', fmt='g')
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
        st.pyplot(fig3)

        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------
    # TAB 5: PREDICTION PANEL
    # -------------------------------
    with tabs[4]:
        st.header("🤖 Prediction Panel")
        st.markdown("Provide your details below to predict mental health risk.")

        # Original features used in training
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
            if df[col].dtype in ['float64', 'int64']:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                default_val = float(df[col].mean())
                if i % 2 == 0:
                    user_input[col] = col1.slider(col, min_val, max_val, default_val)
                else:
                    user_input[col] = col2.slider(col, min_val, max_val, default_val)
            else:
                options = df[col].dropna().unique().tolist()
                if i % 2 == 0:
                    user_input[col] = col1.selectbox(col, options)
                else:
                    user_input[col] = col2.selectbox(col, options)

        if st.button("🔍 Predict Mental Health Risk"):
            user_df = pd.DataFrame([user_input])
            user_df = align_features(user_df, model)  # Align with model pipeline
            prediction = model.predict(user_df)[0]
            probs = model.predict_proba(user_df)[0]

            st.markdown(f"#### Predicted Mental Health Risk Level: **{prediction}**")
            st.write(f"Prediction probabilities: Low={probs[0]:.2f}, Medium={probs[1]:.2f}, High={probs[2]:.2f}")

            # Risk message
            if prediction == 'High':
                st.error(
                    "⚠️ High risk detected. Consider seeking professional support or adjusting screen time habits.")
            elif prediction == 'Medium':
                st.warning("🟡 Moderate risk detected. Maintain healthy sleep and activity balance.")
            else:
                st.success("🟢 Low risk detected. Keep up balanced digital habits!")



    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------
# FOOTER
# ----------------------------------------
st.markdown("""
---
<div style='text-align:center; font-size:14px; color:#475569;'>
    📊 Dashboard created by <strong>Adrian Anthony (UTP)</strong> for PETRONAS Digital Skunkworks Project<br>
    Built with <strong>Streamlit</strong> | Random Forest Model | Responsive UI Design
</div>
---
""", unsafe_allow_html=True)
