import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from datetime import datetime





# ====================================================
# 1. LOAD TRAINED MODEL
# ====================================================
MODEL_PATH = "RandomForest_best_pipeline1.pkl"

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error("❌ Failed to load model.")
        st.error(e)
        st.stop()

model = load_model()

# Pull feature names from preprocessing
preproc = model.named_steps["preproc"]
numeric_features = preproc.transformers_[0][2]
categorical_features = preproc.transformers_[1][2]
ALL_FEATURES = list(numeric_features) + list(categorical_features)

# ====================================================
# 2. LOAD DATASET
# ====================================================
DATA_PATH = "digital_wellbeing_dataset.csv"

@st.cache_resource
def load_dataset():
    return pd.read_csv(DATA_PATH)

df = load_dataset()

# ====================================================
# 3. MAIN TABS
# ====================================================
tabs = st.tabs([
    "📘 Dataset Overview",
    "📊 Correlation Analysis",
    "📈 Feature Importance",
    "🧮 Model Evaluation",
    "🤖 Prediction Panel"
])

# ====================================================
# TAB 1 — DATASET OVERVIEW
# ====================================================
with tabs[0]:
    st.header("📘 Dataset Overview")
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Class Distribution Detection")

    possible_targets = [
        "mh_risk", "risk_level", "mh_risk_binned", "risk",
        "mental_health", "mental_health_score",
        "wellbeing_score", "stress", "stress_level",
        "anxiety", "anxiety_level",
        "depression", "depression_level"
    ]

    df_cols_lower = {c.lower(): c for c in df.columns}
    label_col = None

    # Auto-detect target
    for t in possible_targets:
        if t.lower() in df_cols_lower:
            label_col = df_cols_lower[t.lower()]
            break

    # Fallback: any “risk” column
    if label_col is None:
        for c in df.columns:
            if "risk" in c.lower():
                label_col = c
                break

    if label_col is None:
        st.error("⚠️ Target column not found in dataset.")
    else:
        st.success(f"Detected target column: **{label_col}**")

        if pd.api.types.is_numeric_dtype(df[label_col]):
            st.info("Detected numeric target → Binning into Low/Medium/High")
            df["__temp_target__"] = pd.qcut(
                df[label_col], q=3, labels=["Low", "Medium", "High"]
            )
            plot_col = "__temp_target__"
        else:
            plot_col = label_col

        class_counts = df[plot_col].value_counts()

        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            class_counts,
            labels=class_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.8,
            wedgeprops=dict(width=0.35)
        )
        ax.set_title("Class Distribution")
        st.pyplot(fig)

# ====================================================
# TAB 2 — CORRELATION
# ====================================================
with tabs[1]:
    st.header("📊 Correlation Analysis")

    num_df = df.select_dtypes(include=['int64', 'float64'])
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(corr, cmap="viridis", square=False)
    st.pyplot(fig)

# ====================================================
# TAB 3 — FEATURE IMPORTANCE
# ====================================================
with tabs[2]:
    st.header("📈 Feature Importance")

    rf = model.named_steps["clf"]
    importances = rf.feature_importances_

    ohe = preproc.named_transformers_["cat"]["onehot"]
    ohe_cols = ohe.get_feature_names_out(categorical_features)

    final_features = list(numeric_features) + list(ohe_cols)

    df_importance = pd.DataFrame({
        "Feature": final_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(df_importance.head(20))

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(
        data=df_importance.head(10),
        x="Importance", y="Feature",
        palette="Blues_r"
    )
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)

# ====================================================
# TAB 4 — MODEL EVALUATION
# ====================================================
with tabs[3]:
    st.header("🧮 Model Evaluation")

    try:
        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.metric("Model Accuracy", f"{acc*100:.2f}%")

        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(
            cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=model.classes_,
            yticklabels=model.classes_
        )
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ Could not load X_test.pkl or y_test.pkl")
        st.error(e)

# ====================================================
# TAB 5 — PREDICTION PANEL
# ====================================================
with tabs[4]:
    st.header("🤖 Prediction Panel")
    st.write("Enter your information below:")

    age = st.number_input("Age", 1, 100, 21)
    tech = st.number_input("Daily Technology Usage (hours)", 0.0, 24.0, 3.0)
    social = st.number_input("Social Media Usage (hours)", 0.0, 24.0, 2.0)
    gaming = st.number_input("Gaming Usage (hours)", 0.0, 24.0, 1.0)
    screen = st.number_input("Daily Screen Time (hours)", 0.0, 24.0, 6.0)
    stress = st.number_input("Stress Level (0–10)", 0.0, 10.0, 5.0)
    sleep = st.number_input("Sleep Duration (hours)", 0.0, 24.0, 7.0)
    activity = st.number_input("Physical Activity (hours)", 0.0, 24.0, 1.0)

    user_input = {
        "Age": age,
        "Technology_Usage_Hours": tech,
        "Social_Media_Usage_Hours": social,
        "Gaming_Usage_Hours": gaming,
        "Daily_Screen_Time_Hours": screen,
        "Stress_Level": stress,
        "Sleep_Hours": sleep,
        "Physical_Activity_Hours": activity
    }

    df_user = pd.DataFrame([user_input])

    # Column alignment
    def align(df, cols):
        new = df.copy()
        for c in cols:
            if c not in new:
                new[c] = 0
        return new[cols]

    df_aligned = align(df_user, ALL_FEATURES)

    if st.button("Predict Risk Level"):
        try:
            pred = model.predict(df_aligned)[0]
            probs = model.predict_proba(df_aligned)[0]
            confidence = max(probs) * 100

            st.success(f"### 🧠 Predicted Risk Level: **{pred}**")

            if hasattr(model, "predict_proba"):
                st.subheader("📊 Probability Breakdown")
                for cls, p in zip(model.classes_, probs):
                    st.write(f"**{cls}** → {p*100:.2f}%")

            st.subheader("📝 Personalized Recommendations")

            if pred.lower() == "low":
                st.info("""
                ✔ You are in **Low Risk**.
                - Maintain healthy sleep routines (7–9 hours daily).
                - Continue balancing digital usage with physical activities.
                - Keep positive social interactions both online & offline.
                """)

            elif pred.lower() == "medium":
                st.warning("""
                ⚠ You are in **Moderate Risk**.
                - Reduce screen time, especially at night.
                - Exercise at least 30 minutes a day.
                - Take digital breaks: 10 minutes every hour.
                - Monitor your mood and stress regularly.
                """)

            elif pred.lower() == "high":
                st.error("""
                🔥 You are in **High Risk**.
                - Seek support from a counsellor or mental health professional.
                - Limit gaming & social media to reduce digital overload.
                - Practice mindfulness (breathing, meditation, journaling).
                - Build a consistent sleep routine.
                - Engage in outdoor physical activities daily.
                """)

            # ====================================================
            # SECURE BACKEND LOGGING (NEW)
            # ====================================================
            proba_dict = dict(zip(model.classes_, probs))

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **df_user.iloc[0].to_dict(),
                "prediction": pred,
                "confidence_pct": round(confidence, 2)
            }

            for cls, p in proba_dict.items():
                record[f"proba_{cls}"] = round(float(p), 6)

            
            # ====================================================

        except Exception as e:
            st.error("Prediction failed.")
            st.error(e)

