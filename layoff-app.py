import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model & encoder
model = joblib.load("layoff_xgboost_model.pkl")
encoder = joblib.load("job_level_encoder.pkl")

st.set_page_config(page_title="Layoff Risk Predictor", layout="centered")

st.title("📉 Employee Layoff Risk Predictor")
st.write("Predicts layoff vulnerability using **employee-only data**.")

st.divider()

# -------- Input Form --------
age = st.number_input("Age", min_value=18, max_value=65, value=30)
tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=240, value=24)

job_level = st.selectbox("Job Level", ["junior", "mid", "senior"])
job_level_encoded = encoder.transform([job_level])[0]

skill_overlap_score = st.slider("Skill Overlap Score (replaceability)", 0.0, 1.0, 0.6)
avg_performance_rating = st.slider("Avg Performance Rating", 1.0, 5.0, 3.5)
performance_trend = st.selectbox("Performance Trend", [-1, 0, 1])

absenteeism_rate = st.slider("Absenteeism Rate", 0.0, 0.5, 0.05)
salary_to_role_avg_ratio = st.slider("Salary vs Role Avg Ratio", 0.7, 1.5, 1.0)
training_hours_last_year = st.number_input("Training Hours (last year)", 0, 200, 20)

# -------- Prediction --------
if st.button("Predict Layoff Risk"):
    
    input_data = pd.DataFrame([[
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

    probability = model.predict_proba(input_data)[0][1]

    # Risk bucket
    if probability < 0.30:
        risk = "🟢 Low Risk"
    elif probability < 0.60:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    st.subheader("Prediction Result")
    st.metric("Layoff Probability", f"{probability:.2%}")
    st.write("Risk Category:", risk)

    st.caption("⚠️ This is a statistical risk estimate, not a decision.")
