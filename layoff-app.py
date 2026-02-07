import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Layoff Risk Predictor",
    layout="centered"
)

st.title("📉 Employee Layoff Risk Predictor")
st.write(
    "Predicts **layoff vulnerability** using employee-only data.\n\n"
    "⚠️ This is a **risk estimation**, not a layoff decision."
)

st.divider()

# -----------------------------
# Load Model & Encoder
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("layoff_xgboost_model.pkl")
    encoder = joblib.load("job_level_encoder.pkl")
    explainer = shap.TreeExplainer(model)
    return model, encoder, explainer

model, encoder, explainer = load_model()

# -----------------------------
# Input Form
# -----------------------------
st.subheader("🧾 Employee Details")

age = st.number_input("Age", 18, 65, 30)
tenure_months = st.number_input("Tenure (months)", 0, 240, 24)

job_level = st.selectbox("Job Level", ["junior", "mid", "senior"])
job_level_encoded = encoder.transform([job_level])[0]

skill_overlap_score = st.slider(
    "Skill Overlap Score (replaceability)", 0.0, 1.0, 0.6
)

avg_performance_rating = st.slider(
    "Average Performance Rating", 1.0, 5.0, 3.5
)

performance_trend = st.selectbox(
    "Performance Trend", [-1, 0, 1],
    help="-1 = declining, 0 = stable, 1 = improving"
)

absenteeism_rate = st.slider(
    "Absenteeism Rate", 0.0, 0.5, 0.05
)

salary_to_role_avg_ratio = st.slider(
    "Salary vs Role Average Ratio", 0.7, 1.5, 1.0
)

training_hours_last_year = st.number_input(
    "Training Hours (last year)", 0, 200, 20
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔮 Predict Layoff Risk"):

    input_df = pd.DataFrame([[
        age,
        tenure_months,
        job_level_encoded,
        skill_overlap_score,
        avg_performance_rating,
        performance_trend,
        absenteeism_rate,
        salary_to_role_avg_ratio,
        training_hours_last_year
    ]], columns=[
        "age",
        "tenure_months",
        "job_level_encoded",
        "skill_overlap_score",
        "avg_performance_rating",
        "performance_trend",
        "absenteeism_rate",
        "salary_to_role_avg_ratio",
        "training_hours_last_year"
    ])

    # -----------------------------
    # Prediction
    # -----------------------------
    probability = model.predict_proba(input_df)[0][1]

    if probability < 0.30:
        risk_label = "🟢 Low Risk"
    elif probability < 0.60:
        risk_label = "🟡 Medium Risk"
    else:
        risk_label = "🔴 High Risk"

    st.subheader("📊 Prediction Result")
    st.metric("Layoff Probability", f"{probability:.2%}")
    st.write("**Risk Category:**", risk_label)

    st.divider()

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    st.subheader("🔍 SHAP Explanation (Why this prediction?)")

    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns
        ),
        show=False
    )

    st.pyplot(fig)

    st.caption(
        "🔴 Red features increase layoff risk\n"
        "🔵 Blue features reduce layoff risk"
    )

# -----------------------------
# Optional Global SHAP Section
# -----------------------------
with st.expander("📌 Show Global Feature Importance (SHAP)"):

    st.write(
        "This view shows which employee attributes influence layoff risk "
        "across **all employees**."
    )

    # Dummy background sample for global plot
    background = shap.sample(
        pd.DataFrame(
            np.random.rand(200, 9),
            columns=[
                "age",
                "tenure_months",
                "job_level_encoded",
                "skill_overlap_score",
                "avg_performance_rating",
                "performance_trend",
                "absenteeism_rate",
                "salary_to_role_avg_ratio",
                "training_hours_last_year"
            ]
        ),
        100
    )

    shap_values_bg = explainer.shap_values(background)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_values_bg,
        background,
        show=False
    )

    st.pyplot(fig)