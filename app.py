import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = "models/attrition_model.joblib"
DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

DROP_COLUMNS = [
    "Attrition", "EmployeeNumber", "EmployeeCount",
    "Over18", "StandardHours"
]

X = df.drop(columns=DROP_COLUMNS)

# Typical values provide defaults for fields not shown in the form.
baseline = {}
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        baseline[col] = X[col].median()
    else:
        baseline[col] = X[col].mode()[0]

st.markdown("""
<style>
.main-title {
    font-size: 2.35rem;
    font-weight: 750;
    color: #243b75;
}
.subtitle {
    color: #64748b;
    font-size: 1rem;
}
div.stButton > button {
    background-color: #354fc4;
    color: white;
    border-radius: 9px;
    width: 100%;
    height: 3rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">Employee Attrition Prediction</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Explore workforce patterns with machine learning.</div>',
    unsafe_allow_html=True
)
st.write("")

total_employees = len(df)
attrition_rate = (df["Attrition"] == "Yes").mean() * 100

c1, c2, c3 = st.columns(3)
c1.metric("Employees in dataset", f"{total_employees:,}")
c2.metric("Historical attrition rate", f"{attrition_rate:.2f}%")
c3.metric("Model", "Logistic Regression / trained model")

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Employee details")
    st.caption("Change the example employee's details below.")

    with st.form("prediction_form"):
        age = st.slider("Age", 18, 60, 30)
        income = st.number_input(
            "Monthly income", min_value=1000,
            max_value=20000, value=5000, step=500
        )
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        job_role = st.selectbox(
            "Job role", sorted(df["JobRole"].unique())
        )
        years = st.slider("Years at company", 0, 40, 3)
        satisfaction = st.slider(
            "Job satisfaction (1 = low, 4 = high)", 1, 4, 3
        )
        submitted = st.form_submit_button("Predict Attrition")

with right:
    st.subheader("Prediction result")

    if submitted:
        employee = baseline.copy()
        employee["Age"] = age
        employee["MonthlyIncome"] = income
        employee["OverTime"] = overtime
        employee["JobRole"] = job_role
        employee["YearsAtCompany"] = years
        employee["JobSatisfaction"] = satisfaction

        employee_df = pd.DataFrame([employee])[X.columns]
        prediction = model.predict(employee_df)[0]
        classes = list(model.classes_)
        probability = model.predict_proba(employee_df)[
            0, classes.index("Yes")
        ]

        st.metric("Estimated attrition probability", f"{probability:.1%}")

        if prediction == "Yes":
            st.warning("The model predicts: Attrition")
        else:
            st.success("The model predicts: No Attrition")

        st.progress(float(probability))
        st.caption(
            "This probability is a model estimate, not a certainty."
        )
    else:
        st.info("Enter employee details and click Predict Attrition.")

st.divider()

st.subheader("Explore the dataset")

chart1, chart2 = st.columns(2)

with chart1:
    st.write("**Employee attrition distribution**")
    counts = df["Attrition"].value_counts().reindex(
        ["No", "Yes"], fill_value=0
    )
    st.bar_chart(counts)

with chart2:
    st.write("**Attrition rate by job role**")
    rates = (
        df.assign(Left=(df["Attrition"] == "Yes").astype(int))
          .groupby("JobRole")["Left"]
          .mean()
          .mul(100)
          .sort_values(ascending=False)
    )
    st.bar_chart(rates)

st.divider()
st.caption(
    "Educational machine learning project using the IBM HR Analytics "
    "Employee Attrition dataset. Predictions should not be used alone "
    "to make employment decisions."
)
