import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# ==========================================
# Load Trained Model
# ==========================================
try:
    model = joblib.load("salary_prediction_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'salary_prediction_model.pkl' not found.")
    st.stop()

# ==========================================
# Load Dataset
# ==========================================
try:
    df = pd.read_csv("Salary_Data.csv")
except FileNotFoundError:
    st.error("❌ Dataset 'Salary_Data.csv' not found.")
    st.stop()

# ==========================================
# Extract Data
# ==========================================
X = df["YearsExperience"].values
y = df["Salary"].values

# ==========================================
# Model Parameters
# ==========================================
intercept = model.intercept_
coefficient = model.coef_[0]

# ==========================================
# Model Evaluation Metrics
# ==========================================
y_pred = model.predict(X.reshape(-1, 1))

MAE = mean_absolute_error(y, y_pred)
RMSE = np.sqrt(mean_squared_error(y, y_pred))
R2 = r2_score(y, y_pred)

# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("📖 Model Information")

st.sidebar.markdown("### 🤖 Algorithm")
st.sidebar.info("Simple Linear Regression")

st.sidebar.markdown("### 📥 Input Feature")
st.sidebar.write("**Years of Experience**")

st.sidebar.markdown("### 🎯 Target Variable")
st.sidebar.write("**Salary**")

st.sidebar.divider()

st.sidebar.markdown("### 📐 Regression Equation")

st.sidebar.latex(
    rf"Salary = {intercept:.2f} + ({coefficient:.2f}\times Experience)"
)

st.sidebar.caption(
    "The equation above is learned from the training dataset."
)
st.sidebar.divider()

st.sidebar.markdown("### 📈 Dataset Statistics")

st.sidebar.write(f"**Records:** {len(df)}")
st.sidebar.write(f"**Features:** 1")
st.sidebar.write(f"**Target:** Salary")

st.sidebar.divider()

st.sidebar.markdown("### 📊 Model Performance")

st.sidebar.metric("MAE", f"{MAE:.2f}")
st.sidebar.metric("RMSE", f"{RMSE:.2f}")
st.sidebar.metric("R² Score", f"{R2:.3f}")

st.sidebar.divider()

st.sidebar.markdown("### 🛠️ Technologies Used")

st.sidebar.write("""
- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Joblib
""")

# ==========================================
# Main Page
# ==========================================
st.title("💼 Employee Salary Prediction")

st.markdown("""
Predict an employee's salary based on **Years of Experience**
using a **Simple Linear Regression** model trained with
**Scikit-Learn**.
""")

st.divider()

# ==========================================
# User Input
# ==========================================
experience = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.5
)

st.write(f"### Experience Entered: **{experience:.1f} years**")

st.divider()

# ==========================================
# Prediction
# ==========================================
if st.button("🔍 Predict Salary", use_container_width=True):

    prediction = model.predict(np.array([[experience]]))[0]

    # ==========================================
    # Prediction Result
    # ==========================================
    st.success("Prediction completed successfully!")

    st.metric(
        label="💰 Predicted Salary",
        value=f"${prediction:,.2f}"
    )

    # ==========================================
    # Prediction Insights
    # ==========================================
    st.subheader("🧠 Prediction Insights")

    st.write(
        f"""
Based on **{experience:.1f} years** of experience,
the estimated salary is **${prediction:,.2f}**.
"""
    )

    st.info(
        f"""
### Regression Equation

Salary = {intercept:.2f} + ({coefficient:.2f} × Experience)

For **{experience:.1f} years**:

**Salary = {intercept:.2f} + ({coefficient:.2f} × {experience:.1f})**

≈ **${prediction:,.2f}**
"""
    )

    st.markdown("### 📌 Interpretation")

    st.write(
        f"""
- The prediction was generated using a **Simple Linear Regression** model.
- The model assumes a **linear relationship** between years of experience and salary.
- Every additional year of experience increases the predicted salary by approximately **${coefficient:,.2f}**.
- This prediction should be treated as an estimate based on historical training data.
"""
    )

    st.divider()

    # ==========================================
    # Regression Visualization
    # ==========================================
    st.subheader("📈 Regression Visualization")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Training data
    ax.scatter(
        X,
        y,
        color="royalblue",
        alpha=0.75,
        s=60,
        label="Training Data"
    )

    # Regression line
    x_sorted = np.sort(X)
    ax.plot(
        x_sorted,
        model.predict(x_sorted.reshape(-1, 1)),
        color="crimson",
        linewidth=2.5,
        label="Regression Line"
    )

    # User prediction
    ax.scatter(
        experience,
        prediction,
        color="green",
        s=250,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label="Your Prediction"
    )

    ax.set_title("Salary vs Years of Experience")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")

    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend()

    st.pyplot(fig)

    st.divider()

    # ==========================================
    # Dataset Summary
    # ==========================================
    st.subheader("📊 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Records", len(df))
    col2.metric("Min Salary", f"${df['Salary'].min():,.0f}")
    col3.metric("Max Salary", f"${df['Salary'].max():,.0f}")

    col4, col5 = st.columns(2)

    col4.metric(
        "Average Salary",
        f"${df['Salary'].mean():,.0f}"
    )

    col5.metric(
        "Average Experience",
        f"{df['YearsExperience'].mean():.1f} Years"
    )

    st.divider()

    # ==========================================
    # Dataset Preview
    # ==========================================
    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # About Dataset
    # ==========================================
    with st.expander("ℹ️ About the Dataset"):

        st.markdown("""
### Dataset Information

- **Dataset:** Salary Dataset
- **Source:** Kaggle
- **Algorithm:** Simple Linear Regression
- **Input Feature:** Years of Experience
- **Target Variable:** Salary

This dataset is widely used for learning and demonstrating
supervised machine learning and regression analysis.
""")

    # ==========================================
    # Model Limitations
    # ==========================================
    with st.expander("⚠️ Model Limitations"):

        st.markdown("""
- The model assumes a linear relationship between experience and salary.
- Salary is influenced by many additional factors such as education, location, industry, skills, and company size.
- Predictions are estimates based on historical data.
- This application is intended for educational and demonstration purposes.
""")

st.divider()

st.caption(
    "Developed with ❤️ using Python, Scikit-Learn, and Streamlit"
)