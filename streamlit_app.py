import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import sqlite3
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Early Stage Diabetic Prediction",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("Model/lightgbm.pkl")

# ---------------- FEATURE NAMES (MODEL ORDER) ----------------
FEATURE_NAMES = [
    "Age",
    "DelayedHealing",
    "SuddenWeightLoss",
    "VisualBlurring",
    "Obesity",
    "Polyphagia",
    "Polyuria",
    "Gender",
    "Polydipsia",
    "Irritability"
]

# ---------------- COLUMN MAPPING ----------------
COLUMN_MAPPING = {
    "age": "Age",
    "delayed healing": "DelayedHealing",
    "sudden weight loss": "SuddenWeightLoss",
    "visual blurring": "VisualBlurring",
    "obesity": "Obesity",
    "polyphagia": "Polyphagia",
    "polyuria": "Polyuria",
    "gender": "Gender",
    "polydipsia": "Polydipsia",
    "irritability": "Irritability"
}

# ---------------- DATABASE ----------------
conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    age INTEGER,
    result TEXT,
    probability REAL,
    timestamp TEXT
)
""")
conn.commit()

# ---------------- SESSION ----------------
if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.role = None

# ---------------- LOGIN ----------------
if not st.session_state.auth:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.auth = True
            st.session_state.role = "Admin"
            st.success("Logged in as Admin")
            st.rerun()
        elif username == "user" and password == "user123":
            st.session_state.auth = True
            st.session_state.role = "User"
            st.success("Logged in as User")
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# ---------------- HEADER ----------------
st.markdown(
    """
    <div style='text-align:center; padding:15px; border:2px solid #444; border-radius:15px;'>
        <h2>Karnataka State Akkamahadevi Women's University, Vijayapur</h2>
        <h4>Department of Computer Science</h4>
        <hr>
        <h3>Early Stage Diabetic Prediction System</h3>
    </div>
    """,
    unsafe_allow_html=True
)

supervisor = st.text_input("Supervisor Name", "Dr. Shitalrani Kavale")
researcher = "Pooja Kallappagol"

# ---------------- SINGLE INPUT ----------------
st.subheader("🧍 Patient Details")

with st.form("predict_form"):
    Age = st.number_input("Age", 1, 120, 30)
    Gender = st.selectbox("Gender", ["Female", "Male"])
    Polyuria = st.selectbox("Polyuria", [0, 1])
    Polydipsia = st.selectbox("Polydipsia", [0, 1])
    SuddenWeightLoss = st.selectbox("Sudden Weight Loss", [0, 1])
    Polyphagia = st.selectbox("Polyphagia", [0, 1])
    VisualBlurring = st.selectbox("Visual Blurring", [0, 1])
    Irritability = st.selectbox("Irritability", [0, 1])
    DelayedHealing = st.selectbox("Delayed Healing", [0, 1])
    Obesity = st.selectbox("Obesity", [0, 1])

    submit = st.form_submit_button("Predict")

if submit:
    Gender = 1 if Gender == "Male" else 0

    X_single = pd.DataFrame([[Age, DelayedHealing, SuddenWeightLoss, VisualBlurring,
                              Obesity, Polyphagia, Polyuria, Gender,
                              Polydipsia, Irritability]],
                            columns=FEATURE_NAMES)

    prediction = model.predict(X_single)[0]
    probability = model.predict_proba(X_single)[0][1]

    if prediction == 1:
        st.error(f"🩺 Diabetic (Probability: {probability*100:.2f}%)")
        result = "Diabetic"
    else:
        st.success(f"✅ Non-Diabetic (Probability: {(1-probability)*100:.2f}%)")
        result = "Non-Diabetic"

    cursor.execute(
        "INSERT INTO predictions VALUES (?, ?, ?, ?)",
        (Age, result, probability, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()

# ---------------- CSV TEMPLATE ----------------
st.subheader("📥 Download CSV Template")
template_df = pd.DataFrame(columns=FEATURE_NAMES)
st.download_button(
    "⬇ Download CSV Template",
    template_df.to_csv(index=False),
    "diabetes_input_template.csv"
)

# ---------------- BATCH PREDICTION ----------------
st.subheader("📂 Batch Prediction (CSV Upload)")
file = st.file_uploader("Upload CSV file", type=["csv"])

X_batch, preds, probs = None, None, None

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):
        df.columns = df.columns.str.lower().str.strip()
        df = df.rename(columns=COLUMN_MAPPING)

        # Auto gender encoding
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].replace(
                {"Male": 1, "male": 1, "Female": 0, "female": 0}
            )

        # Missing & extra columns
        missing = set(FEATURE_NAMES) - set(df.columns)
        extra = set(df.columns) - set(FEATURE_NAMES)

        if missing:
            st.error(f"❌ Missing required columns: {missing}")
            st.stop()

        if extra:
            st.warning(f"⚠ Extra columns will be ignored: {extra}")

        # Feature order verification
        X_batch = df[FEATURE_NAMES]

        preds = model.predict(X_batch)
        probs = model.predict_proba(X_batch)[:, 1]

        df["Prediction"] = ["Diabetic" if p == 1 else "Non-Diabetic" for p in preds]
        df["Probability"] = probs

        st.success("Batch prediction completed")
        st.dataframe(df)

        st.download_button(
            "⬇ Download Results",
            df.to_csv(index=False),
            "diabetes_predictions.csv"
        )

# ---------------- ADMIN PANEL ----------------
if st.session_state.role == "Admin":
    st.subheader("📊 Model Evaluation")

    if X_batch is not None and "class" in df.columns:
        acc = accuracy_score(df["class"], preds)
        st.write(f"Accuracy: {acc:.2f}")

        fpr, tpr, _ = roc_curve(df["class"], probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.legend()
        st.pyplot(fig)

    st.subheader("🔍 SHAP Explanation")

    if X_batch is not None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_batch)

        fig2 = plt.figure()
        shap.summary_plot(shap_values, X_batch, show=False)
        st.pyplot(fig2)
    else:
        st.info("Run batch prediction to view SHAP explanation.")

    st.subheader("🗄 Prediction Logs")
    logs = pd.read_sql("SELECT * FROM predictions", conn)
    st.dataframe(logs)

# ---------------- FOOTER ----------------
st.markdown(
    f"""
    <hr>
    <div style='text-align:center;'>
        <b>Early Stage Diabetic Prediction</b><br>
        By {researcher}, Research Scholar<br>
        Supervisor: {supervisor}
    </div>
    """,
    unsafe_allow_html=True
)
