import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("salary_prediction_model.pkl")

st.title("Employee Salary Prediction")

st.write(
    "Predict an employee's salary based on years of experience."
)

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=2.0,
    step=0.5
)

if st.button("Predict Salary"):

    prediction = model.predict(np.array([[experience]]))

    st.success(
        f"Predicted Salary: ${prediction[0]:,.2f}"
    )