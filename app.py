import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💼",
    layout="centered"
)

model = joblib.load("salary_prediction_model.pkl")

st.sidebar.header("About")

st.sidebar.write("""
This application predicts employee salaries using a **Simple Linear Regression** model trained with **Scikit-Learn**.

**Input**
- Years of Experience

**Output**
- Predicted Salary
""")

st.title("💼 Employee Salary Prediction")
st.markdown(
    "Predict an employee's salary based on **years of experience** using a Machine Learning model."
)

st.write(
    "Predict an employee's salary based on years of experience."
)

experience = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.5
)

st.write(f"Experience Entered: **{experience} years**")

if st.button("Predict Salary"):

    prediction = model.predict(np.array([[experience]]))

    st.success(
        st.metric(
    label="Predicted Salary",
    value=f"${prediction[0]:,.2f}"
            )
    )