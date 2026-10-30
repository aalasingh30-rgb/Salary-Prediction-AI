
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_prediction_model.pkl")
data = pd.read_csv("Salary_Data.csv")

st.set_page_config(
    page_title="Salary Prediction AI",
    page_icon="💰",
    layout="centered"
)

st.title("💰 AI Salary Prediction")
st.write("Predict your estimated annual salary using Machine Learning.")
st.divider()

st.subheader("Enter Your Details")

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=60.0,
    value=2.0,
    step=0.5
)

education = st.selectbox(
    "Education Level",
    sorted(data["Education Level"].dropna().unique())
)

job_role = st.selectbox(
    "Job Role",
    sorted(data["Job Title"].dropna().unique())
)

if st.button("Predict Salary", use_container_width=True):

    input_data = pd.DataFrame({
        "Years of Experience": [experience],
        "Education Level": [education],
        "Job Title": [job_role]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction Generated!")
    st.subheader("Your Estimated Annual Salary")
    st.markdown(f"## ₹ {prediction:,.2f}")

    st.caption("This is an estimated prediction, not a guaranteed salary.")

st.divider()
st.caption("AI/ML Project | Salary Prediction System")
