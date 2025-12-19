import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import shap

# =====================================================
# PAGE CONFIG (MOBILE FRIENDLY)
# =====================================================
st.set_page_config(
    page_title="Early Stage Diabetic Prediction",
    layout="centered"
)

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
    }
    .cert-box {
        border: 4px double #2c2c2c;
        padding: 25px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOGIN / AUTHENTICATION
# =====================================================
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "diabetes123":
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid username or password")

if "authenticated" not in st.session_state:
    login()
    st.stop()

# =====================================================
# CERTIFICATE STYLE HEADER
# =====================================================
st.markdown(
    """
    <div class="cert-box">
        <h2>Karnataka State Akkamahadevi Women's University</h2>
        <h3>Vijayapur</h3>
        <h4>Department of Computer Science</h4>
        <h3><u>Certificate</u></h3>
        <p>
            This is to certify that the project entitled<br>
            <b>“Early Stage Diabetic Prediction System”</b><br>
            is carried out as part of academic research.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# DYNAMIC DETAILS
# =====================================================
research_scholar = "Pooja Kallappagol"
supervisor = st.text_input("Supervisor Name", value="Dr. Shitalrani Kavale")
academic_year = "2024–25"

st.markdown(
    f"""
    **Research Scholar:** {research_scholar}  
    **Supervisor:** {supervisor}  
    **Academic Year:** {academic_year}
    """
)

st.divider()

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load("Model/lightgbm.pkl")

# =====================================================
# SINGLE PATIENT PREDICTION
# =====================================================
st.subheader("🧑 Patient Information")

Age = st.number_input("Age", 1, 120, 30)

Gender = st.selectbox("Gender", ["Male", "Female"])
Gender = 1 if Gender == "Male" else 0

def yn(label):
    return 1 if st.selectbox(label, ["Yes", "No"]) == "Yes" else 0

Polyuria = yn("Polyuria")
Sudden_weight_loss = yn("Sudden Weight Loss")
Polyphagia = yn("Polyphagia")
Visual_blurring = yn("Visual Blurring")
Irritability = yn("Irritability")
Delayed_healing = yn("Delayed Healing")
Polydipsia = yn("Polydipsia")
Obesity = yn("Obesity")

if st.button("🔍 Predict Diabetes"):
    input_data = np.array([[
        Age,
        Delayed_healing,
        Sudden_weight_loss,
        Visual_blurring,
        Obesity,
        Polyphagia,
        Polyuria,
        Gender,
        Polydipsia,
        Irritability
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"🩺 **Diabetic** (Probability: {probability*100:.2f}%)")
    else:
        st.success(f"✅ **Non-Diabetic** (Probability: {(1-probability)*100:.2f}%)")

# =====================================================
# DATASET UPLOAD & BATCH PREDICTION
# =====================================================
st.divider()
st.subheader("📂 Dataset Upload (Batch Prediction)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    if st.button("Run Batch Prediction"):
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        df["Prediction"] = ["Diabetic" if p == 1 else "Non-Diabetic" for p in preds]
        df["Probability"] = probs

        st.success("Batch prediction completed")
        st.dataframe(df)

        st.download_button(
            "⬇ Download Results",
            df.to_csv(index=False),
            "diabetes_predictions.csv",
            "text/csv"
        )

# =====================================================
# ACCURACY & ROC CURVE
# =====================================================
st.divider()
st.subheader("📊 Model Performance (Accuracy & ROC)")

try:
    test_df = pd.read_csv("test.csv")   # must contain target column: class

    X_test = test_df.drop("class", axis=1)
    y_test = test_df["class"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {acc*100:.2f}%")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    st.pyplot(fig)

except:
    st.warning("Add test.csv with target column 'class' to view accuracy & ROC curve")

# =====================================================
# SHAP MODEL EXPLANATION
# =====================================================
st.divider()
st.subheader("🧠 Model Explanation (SHAP)")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

except:
    st.warning("SHAP explanation requires test dataset")

# =====================================================
# FOOTER
# =====================================================
current_year = datetime.now().year

st.markdown(
    f"""
    <hr>
    <div style="text-align:center; font-size:13px;">
        Early Stage Diabetic Prediction System <br>
        By <b>{research_scholar}</b>, Research Scholar <br>
        © {current_year}
    </div>
    """,
    unsafe_allow_html=True
)
