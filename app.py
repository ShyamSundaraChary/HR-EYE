import streamlit as st
import pandas as pd
import joblib
from google import genai
import time
import random
from google.genai import errors as genai_errors
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

# -----------------------------
# Gemini setup
# -----------------------------
@st.cache_resource
def load_gemini_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

gemini_client = load_gemini_client()


def _safe_generate_content(prompt: str, model: str = "gemini-3-flash-preview", max_retries: int = 5):
    """Call Gemini with exponential backoff and jitter on ServerError (503).

    Returns the response object on success or raises the last exception on final failure.
    """
    base_delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            return gemini_client.models.generate_content(
                model=model,
                contents=prompt,
            )
        except genai_errors.ServerError as e:
            # Model overloaded / 503 — retry with exponential backoff + jitter
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            # add some jitter so retries across clients don't synchronize
            delay = delay * (0.8 + random.random() * 0.4)
            time.sleep(delay)
        except Exception:
            # For other errors, re-raise immediately
            raise


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
st.markdown("Decision support tool for **HR managers** with clear explanations.")

# -----------------------------
# Helper mappings (REAL values)
# -----------------------------

satisfaction_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

worklife_map = {
    "Bad": 1,
    "Good": 2,
    "Better": 3,
    "Best": 4
}

education_map = {
    "Below College": 1,
    "College": 2,
    "Bachelor": 3,
    "Master": 4,
    "Doctor": 5
}

joblevel_map = {
    "Entry Level": 1,
    "Junior": 2,
    "Mid-Level": 3,
    "Senior": 4,
    "Director / VP": 5
}

# -----------------------------
# Sample Data Functions (Modular)
# -----------------------------
def get_high_risk_sample():
    """Returns sample data for High Attrition Risk employee"""
    return {
        "Age": 25,
        "DailyRate": 800,
        "HourlyRate": 150,
        "MonthlyIncome": 25000,
        "MonthlyRate": 12000,
        "DistanceFromHome": 25,
        "TotalWorkingYears": 2,
        "YearsAtCompany": 1,
        "YearsInCurrentRole": 0,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0,
        "NumCompaniesWorked": 4,
        "PercentSalaryHike": 12,
        "TrainingTimesLastYear": 0,
        "Education": "College",
        "EnvironmentSatisfaction": "Low",
        "JobInvolvement": "Low",
        "JobLevel": "Entry Level",
        "JobSatisfaction": "Low",
        "PerformanceRating": 3,
        "RelationshipSatisfaction": "Low",
        "StockOptionLevel": 0,
        "WorkLifeBalance": "Bad",
        "BusinessTravel": "Travel_Frequently",
        "Department": "Sales",
        "EducationField": "Technical Degree",
        "Gender": "Male",
        "JobRole": "Sales Representative",
        "MaritalStatus": "Single",
        "OverTime": "Yes"
    }

def get_low_risk_sample():
    """Returns sample data for Low Attrition Risk employee"""
    return {
        "Age": 38,
        "DailyRate": 1200,
        "HourlyRate": 80,
        "MonthlyIncome": 95000,
        "MonthlyRate": 22000,
        "DistanceFromHome": 5,
        "TotalWorkingYears": 15,
        "YearsAtCompany": 8,
        "YearsInCurrentRole": 5,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 4,
        "NumCompaniesWorked": 2,
        "PercentSalaryHike": 18,
        "TrainingTimesLastYear": 3,
        "Education": "Master",
        "EnvironmentSatisfaction": "High",
        "JobInvolvement": "High",
        "JobLevel": "Senior",
        "JobSatisfaction": "Very High",
        "PerformanceRating": 4,
        "RelationshipSatisfaction": "High",
        "StockOptionLevel": 2,
        "WorkLifeBalance": "Best",
        "BusinessTravel": "Travel_Rarely",
        "Department": "Research & Development",
        "EducationField": "Life Sciences",
        "Gender": "Female",
        "JobRole": "Manager",
        "MaritalStatus": "Married",
        "OverTime": "No"
    }

# -----------------------------
# Sample Data Loader UI
# -----------------------------
st.divider()
st.subheader("📋 Try Sample Employee Profiles")
st.caption("Load pre-filled data to see how the model works with different scenarios")

col_sample, col_button = st.columns([3, 1])

with col_sample:
    sample_choice = st.radio(
        "Choose a sample profile:",
        ["🔴 High Attrition Risk", "🟢 Low Attrition Risk"],
        horizontal=True
    )

with col_button:
    st.write("")  # Spacing
    if st.button("📥 Load Sample", type="primary"):
        if "High" in sample_choice:
            sample_data = get_high_risk_sample()
            st.session_state['sample_loaded'] = True
            st.session_state['sample_type'] = "High Risk"
        else:
            sample_data = get_low_risk_sample()
            st.session_state['sample_loaded'] = True
            st.session_state['sample_type'] = "Low Risk"
        
        # Store sample data in session state
        for key, value in sample_data.items():
            st.session_state[f'input_{key}'] = value
        
        st.success(f"✅ {st.session_state['sample_type']} sample loaded!")
        st.rerun()

if st.session_state.get('sample_loaded'):
    st.info(f"ℹ️ Currently showing: **{st.session_state.get('sample_type')}** sample data")

st.divider()

# -----------------------------
# Input form
# -----------------------------
def default_num(key, default, min_value=None, max_value=None):
    """Get numeric default from session_state and clamp to min/max if provided."""
    v = st.session_state.get(f'input_{key}', default)
    try:
        v = int(v)
    except Exception:
        try:
            v = int(float(v))
        except Exception:
            return default
    if min_value is not None and v < min_value:
        return min_value
    if max_value is not None and v > max_value:
        return max_value
    return v

with st.form("attrition_form"):
    st.subheader("Employee Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        Age = st.number_input(
            "Age (years)", 18, 60,
            default_num('Age', 32, 18, 60)
        )

        DailyRate = st.number_input(
            "Daily Pay (₹)",
            min_value=500,
            max_value=8000,
            value=default_num('DailyRate', 2500, 500, 8000),
            step=100
        )

        HourlyRate = st.number_input(
            "Hourly Pay (₹)",
            min_value=100,
            max_value=1500,
            value=default_num('HourlyRate', 400, 100, 1500),
            step=50
        )

        MonthlyIncome = st.number_input(
            "Monthly Salary (₹)",
            min_value=15000,
            max_value=200000,
            value=default_num('MonthlyIncome', 45000, 15000, 200000),
            step=1000
        )

        MonthlyRate = st.number_input(
            "Monthly Fixed Cost (₹)",
            min_value=10000,
            max_value=100000,
            value=default_num('MonthlyRate', 30000, 10000, 100000),
            step=1000
        )

        DistanceFromHome = st.number_input(
            "Distance From Home (km)",
            min_value=1,
            max_value=50,
            value=default_num('DistanceFromHome', 10, 1, 50)
        )

        TotalWorkingYears = st.number_input(
            "Total Experience (years)",
            min_value=0,
            max_value=40,
            value=default_num('TotalWorkingYears', 8, 0, 40)
        )


    with c2:
        YearsAtCompany = st.number_input(
            "Years at Company", 0, 40,
            default_num('YearsAtCompany', 3, 0, 40)
        )
        YearsInCurrentRole = st.number_input(
            "Years in Current Role", 0, 18,
            default_num('YearsInCurrentRole', 2, 0, 18)
        )
        YearsSinceLastPromotion = st.number_input(
            "Years Since Last Promotion", 0, 15,
            default_num('YearsSinceLastPromotion', 2, 0, 15)
        )
        YearsWithCurrManager = st.number_input(
            "Years with Current Manager", 0, 17,
            default_num('YearsWithCurrManager', 2, 0, 17)
        )
        NumCompaniesWorked = st.number_input(
            "Companies Worked At", 0, 9,
            default_num('NumCompaniesWorked', 2, 0, 9)
        )
        PercentSalaryHike = st.number_input(
            "Last Salary Hike (%)", 11, 25,
            default_num('PercentSalaryHike', 14, 11, 25)
        )
        TrainingTimesLastYear = st.number_input(
            "Trainings Last Year", 0, 6,
            default_num('TrainingTimesLastYear', 2, 0, 6)
        )

    with c3:
        edu_default = st.session_state.get('input_Education', 'Bachelor')
        edu_index = list(education_map.keys()).index(edu_default) if edu_default in education_map else 2
        Education = education_map[
            st.selectbox("Education Level", list(education_map.keys()), index=edu_index)
        ]
        env_default = st.session_state.get('input_EnvironmentSatisfaction', 'Medium')
        env_index = list(satisfaction_map.keys()).index(env_default) if env_default in satisfaction_map else 1
        EnvironmentSatisfaction = satisfaction_map[
            st.selectbox("Environment Satisfaction", satisfaction_map.keys(), index=env_index)
        ]
        job_inv_default = st.session_state.get('input_JobInvolvement', 'Medium')
        job_inv_index = list(satisfaction_map.keys()).index(job_inv_default) if job_inv_default in satisfaction_map else 1
        JobInvolvement = satisfaction_map[
            st.selectbox("Job Involvement", satisfaction_map.keys(), index=job_inv_index)
        ]
        job_lvl_default = st.session_state.get('input_JobLevel', 'Mid-Level')
        job_lvl_index = list(joblevel_map.keys()).index(job_lvl_default) if job_lvl_default in joblevel_map else 2
        JobLevel = joblevel_map[
            st.selectbox("Job Level", joblevel_map.keys(), index=job_lvl_index)
        ]
        job_sat_default = st.session_state.get('input_JobSatisfaction', 'Medium')
        job_sat_index = list(satisfaction_map.keys()).index(job_sat_default) if job_sat_default in satisfaction_map else 1
        JobSatisfaction = satisfaction_map[
            st.selectbox("Job Satisfaction", satisfaction_map.keys(), index=job_sat_index)
        ]
        perf_default = st.session_state.get('input_PerformanceRating', 3)
        perf_index = [3, 4].index(perf_default) if perf_default in [3, 4] else 0
        PerformanceRating = st.selectbox("Performance Rating", [3, 4], index=perf_index)
        rel_sat_default = st.session_state.get('input_RelationshipSatisfaction', 'Medium')
        rel_sat_index = list(satisfaction_map.keys()).index(rel_sat_default) if rel_sat_default in satisfaction_map else 1
        RelationshipSatisfaction = satisfaction_map[
            st.selectbox("Relationship Satisfaction", satisfaction_map.keys(), index=rel_sat_index)
        ]
        stock_default = st.session_state.get('input_StockOptionLevel', 0)
        stock_index = [0, 1, 2, 3].index(stock_default) if stock_default in [0, 1, 2, 3] else 0
        StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=stock_index)
        wlb_default = st.session_state.get('input_WorkLifeBalance', 'Good')
        wlb_index = list(worklife_map.keys()).index(wlb_default) if wlb_default in worklife_map else 1
        WorkLifeBalance = worklife_map[
            st.selectbox("Work-Life Balance", worklife_map.keys(), index=wlb_index)
        ]

    st.subheader("Job Details")

    travel_options = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    travel_default = st.session_state.get('input_BusinessTravel', 'Travel_Rarely')
    travel_index = travel_options.index(travel_default) if travel_default in travel_options else 1
    BusinessTravel = st.selectbox(
        "Business Travel",
        travel_options,
        index=travel_index
    )
    dept_options = ["Sales", "Research & Development", "Human Resources"]
    dept_default = st.session_state.get('input_Department', 'Research & Development')
    dept_index = dept_options.index(dept_default) if dept_default in dept_options else 1
    Department = st.selectbox(
        "Department",
        dept_options,
        index=dept_index
    )
    edu_field_options = ["Life Sciences", "Medical", "Marketing",
                        "Technical Degree", "Human Resources", "Other"]
    edu_field_default = st.session_state.get('input_EducationField', 'Life Sciences')
    edu_field_index = edu_field_options.index(edu_field_default) if edu_field_default in edu_field_options else 0
    EducationField = st.selectbox(
        "Education Field",
        edu_field_options,
        index=edu_field_index
    )
    gender_options = ["Female", "Male"]
    gender_default = st.session_state.get('input_Gender', 'Female')
    gender_index = gender_options.index(gender_default) if gender_default in gender_options else 0
    Gender = st.selectbox("Gender", gender_options, index=gender_index)
    job_role_options = [
        "Sales Executive", "Research Scientist",
        "Laboratory Technician", "Manufacturing Director",
        "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director",
        "Human Resources"
    ]
    job_role_default = st.session_state.get('input_JobRole', 'Sales Executive')
    job_role_index = job_role_options.index(job_role_default) if job_role_default in job_role_options else 0
    JobRole = st.selectbox(
        "Job Role",
        job_role_options,
        index=job_role_index
    )
    marital_options = ["Single", "Married", "Divorced"]
    marital_default = st.session_state.get('input_MaritalStatus', 'Single')
    marital_index = marital_options.index(marital_default) if marital_default in marital_options else 0
    MaritalStatus = st.selectbox("Marital Status", marital_options, index=marital_index)
    overtime_options = ["Yes", "No"]
    overtime_default = st.session_state.get('input_OverTime', 'No')
    overtime_index = overtime_options.index(overtime_default) if overtime_default in overtime_options else 1
    OverTime = st.selectbox("Works Overtime?", overtime_options, index=overtime_index)

    submit = st.form_submit_button("🚀 Predict Attrition")

# -----------------------------
# Gemini explanation
# -----------------------------
def generate_hr_reasoning(df, pred):
    status = (
        "likely to leave the organization"
        if pred == 1
        else "likely to stay with the organization"
    )
    # we need make it back original scale for the explanation to be meaningful to HR managers, so we reverse the transformations we applied to the input features before prediction.
        # "DailyRate": DailyRate/5,
        # "HourlyRate": HourlyRate/3,
        # "MonthlyIncome": MonthlyIncome/10,
    
    df = df.copy()
    df['DailyRate'] = df['DailyRate'] * 5
    df['HourlyRate'] = df['HourlyRate'] * 3
    df['MonthlyIncome'] = df['MonthlyIncome'] * 10
    
    prompt = f"""
        You are an HR analytics assistant.

        An employee is {status}.

        Employee profile:
        {df.to_string(index=False)}

        Explain the reason in 4-6 bullet points.

        Rules:
        - Use simple, professional HR language
        - Focus on compensation, workload, growth, satisfaction, tenure, and work-life balance
        - Do NOT mention AI, ML, models, or probabilities
        - Make the explanation actionable for HR
        """

    try:
        response = _safe_generate_content(prompt)
        if response and hasattr(response, 'text'):
            return response.text
        return "Unable to generate explanation from the response."
    except genai_errors.ServerError:
        # Friendly fallback for overloaded model
        return (
            "The explanation service is temporarily unavailable (503). "
            "Please try again in a few moments."
        )
    except Exception:
        return "Failed to generate explanation due to an unexpected error."

# -----------------------------
# Prediction
# -----------------------------
if submit:
    raw_df = pd.DataFrame([{
        # 🔑 Convert Indian salaries to training scale
        "Age": Age,
        "DailyRate": DailyRate/5,
        "HourlyRate": HourlyRate/3,
        "MonthlyIncome": MonthlyIncome/10,
        "MonthlyRate": MonthlyRate,
        "TotalWorkingYears": TotalWorkingYears,
        "YearsAtCompany": YearsAtCompany,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobSatisfaction": JobSatisfaction,
        "NumCompaniesWorked": NumCompaniesWorked,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
        "BusinessTravel": BusinessTravel,
        "Department": Department,
        "EducationField": EducationField,
        "Gender": Gender,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
        "OverTime": OverTime
    }])

    encoded = pd.get_dummies(raw_df, drop_first=True)

    # Add any missing training columns in one operation to avoid
    # DataFrame fragmentation from many per-column inserts.
    
    missing = [col for col in TRAIN_COLS if col not in encoded.columns]
    if missing:
        zeros = pd.DataFrame(0, index=encoded.index, columns=missing)
        encoded = pd.concat([encoded, zeros], axis=1)

    # Ensure column order matches training columns and return a compact copy
    encoded = encoded[TRAIN_COLS].copy()

    pred = int(model.predict(encoded)[0])
    prob = float(model.predict_proba(encoded)[0][1])

    # Persist latest prediction and input so explanation can be
    # generated on demand (button click) without losing state.
    st.session_state['last_raw_df'] = raw_df
    st.session_state['last_pred'] = pred
    st.session_state['last_prob'] = prob
    st.session_state['prediction_done'] = True

# If a prediction exists in session state, show results and allow
# the user to request the explanation explicitly via a button.
if st.session_state.get('prediction_done'):
    st.subheader("📊 Prediction Result")
    st.write(f"Attrition Probability: **{st.session_state.get('last_prob', 0):.2%}**")
    if st.session_state.get('last_pred') == 1:
        st.error("⚠️ High Attrition Risk (YES)")
    else:
        st.success("✅ Low Attrition Risk (NO)")

    # Button triggers explanation generation using stored input
    if st.button("🧠 Generate HR Explanation"):
        with st.spinner("🧠 Generating HR explanation..."):
            explanation = generate_hr_reasoning(
                st.session_state['last_raw_df'], st.session_state['last_pred']
            )
            st.session_state['last_explanation'] = explanation

    st.subheader("📝 Explanation for HR Managers")
    if st.session_state.get('last_explanation'):
        st.markdown(st.session_state['last_explanation'])
    else:
        st.info("Click '🧠 Generate HR Explanation' to get actionable reasons.")
