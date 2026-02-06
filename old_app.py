import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and columns
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("ML_MODELS/xgboost_attrition_model.pkl")

@st.cache_resource
def load_columns():
    return joblib.load("ML_MODELS/training_columns.pkl")

model = load_model()
TRAIN_COLS = load_columns()

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("🔮 Employee Attrition Prediction")
st.caption("Simple, correct, and explainable HR attrition model")

# -----------------------------
# Sample profiles
# -----------------------------
def high_risk_sample():
    return {
        "Age": 25,
        "MonthlyIncome": 25000,
        "DailyRate": 800,
        "HourlyRate": 150,
        "TotalWorkingYears": 2,
        "YearsAtCompany": 1,
        "JobLevel": 1,
        "JobSatisfaction": 1,
        "WorkLifeBalance": 1,
        "BusinessTravel": "Travel_Frequently",
        "OverTime": "Yes",
        "MaritalStatus": "Single"
    }

def low_risk_sample():
    return {
        "Age": 40,
        "MonthlyIncome": 95000,
        "DailyRate": 1200,
        "HourlyRate": 80,
        "TotalWorkingYears": 15,
        "YearsAtCompany": 8,
        "JobLevel": 4,
        "JobSatisfaction": 4,
        "WorkLifeBalance": 4,
        "BusinessTravel": "Travel_Rarely",
        "OverTime": "No",
        "MaritalStatus": "Married"
    }

st.subheader("📌 Load Sample Employee")
col1, col2 = st.columns(2)

if col1.button("🔴 High Attrition Risk Sample"):
    st.session_state.sample = high_risk_sample()

if col2.button("🟢 Low Attrition Risk Sample"):
    st.session_state.sample = low_risk_sample()

sample = st.session_state.get("sample", {})

# -----------------------------
# Input form
# -----------------------------
with st.form("form"):
    Age = st.number_input("Age", 18, 60, sample.get("Age", 30))
    MonthlyIncome = st.number_input("Monthly Salary (₹)", 15000, 200000, sample.get("MonthlyIncome", 40000))
    DailyRate = st.number_input("Daily Rate (₹)", 500, 8000, sample.get("DailyRate", 1000))
    HourlyRate = st.number_input("Hourly Rate (₹)", 100, 1500, sample.get("HourlyRate", 300))
    TotalWorkingYears = st.number_input("Total Working Years", 0, 40, sample.get("TotalWorkingYears", 8))
    YearsAtCompany = st.number_input("Years at Company", 0, 40, sample.get("YearsAtCompany", 4))
    JobLevel = st.selectbox("Job Level", [1,2,3,4,5], index=sample.get("JobLevel",3)-1)
    JobSatisfaction = st.selectbox("Job Satisfaction", [1,2,3,4], index=sample.get("JobSatisfaction",3)-1)
    WorkLifeBalance = st.selectbox("Work Life Balance", [1,2,3,4], index=sample.get("WorkLifeBalance",3)-1)
    BusinessTravel = st.selectbox("Business Travel", ["Non-Travel","Travel_Rarely","Travel_Frequently"],
                                  index=["Non-Travel","Travel_Rarely","Travel_Frequently"].index(
                                      sample.get("BusinessTravel","Travel_Rarely")))
    OverTime = st.selectbox("OverTime", ["Yes","No"], index=0 if sample.get("OverTime","No")=="Yes" else 1)
    MaritalStatus = st.selectbox("Marital Status", ["Single","Married","Divorced"],
                                 index=["Single","Married","Divorced"].index(sample.get("MaritalStatus","Married")))

    submit = st.form_submit_button("🚀 Predict")

# -----------------------------
# Prediction
# -----------------------------
if submit:
    # 🔑 Convert Indian salaries to training scale
    data = pd.DataFrame([{
        "Age": Age,
        "MonthlyIncome": MonthlyIncome / 10,
        "DailyRate": DailyRate / 5,
        "HourlyRate": HourlyRate / 3,
        "TotalWorkingYears": TotalWorkingYears,
        "YearsAtCompany": YearsAtCompany,
        "JobLevel": JobLevel,
        "JobSatisfaction": JobSatisfaction,
        "WorkLifeBalance": WorkLifeBalance,
        "BusinessTravel": BusinessTravel,
        "OverTime": OverTime,
        "MaritalStatus": MaritalStatus
    }])

    data = pd.get_dummies(data, drop_first=True)

    for col in TRAIN_COLS:
        if col not in data.columns:
            data[col] = 0

    data = data[TRAIN_COLS]

    prob = model.predict_proba(data)[0][1]
    pred = 1 if prob >= 0.35 else 0   # adjusted threshold

    st.subheader("📊 Prediction Result")
    st.write(f"Attrition Probability: **{prob:.2%}**")

    if pred == 1:
        st.error("⚠️ High Attrition Risk (Yes)")
    else:
        st.success("✅ Low Attrition Risk (No)")
