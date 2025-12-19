# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 18:31:02 2025

@author: Admin
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import os

# ---------- SAFE PDF IMPORT ----------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Early Stage Diabetic Prediction", layout="centered")

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")
MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm.pkl")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load(MODEL_PATH)

# ---------------- FEATURE NAMES ----------------
FEATURE_NAMES = [
    "Age","DelayedHealing","SuddenWeightLoss","VisualBlurring",
    "Obesity","Polyphagia","Polyuria","Gender","Polydipsia","Irritability"
]

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

# ---------------- FEEDBACK FILE ----------------
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=FEATURE_NAMES + ["label"]).to_csv(FEEDBACK_FILE, index=False)

# ---------------- LOGIN SESSION ----------------
if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.role = None

# ---------------- LOGIN ----------------
if not st.session_state.auth:
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "admin123":
            st.session_state.auth = True
            st.session_state.role = "Admin"
            st.rerun()
        elif u == "user" and p == "user123":
            st.session_state.auth = True
            st.session_state.role = "User"
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# ---------------- HEADER ----------------
st.markdown("""
<div style='text-align:center; padding:15px; border:2px solid #444; border-radius:15px;'>
<h2>Karnataka State Akkamahadevi Women's University, Vijayapur</h2>
<h4>Department of Computer Science</h4><hr>
<h3>Early Stage Diabetic Prediction System</h3>
</div>
""", unsafe_allow_html=True)

supervisor = "Dr. Shitalrani Kavale"
researcher = "Pooja Kallappagol"

# ---------------- PDF FUNCTION ----------------
def generate_pdf(age, gender, result, prob):
    if not PDF_AVAILABLE:
        return None

    filename = "Diabetes_Report.pdf"
    c = canvas.Canvas(filename, pagesize=A4)

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(300, 800, "Diabetes Medical Report")

    c.setFont("Helvetica", 11)
    c.drawString(50, 760, f"Age: {age}")
    c.drawString(50, 740, f"Gender: {'Male' if gender==1 else 'Female'}")
    c.drawString(50, 720, f"Result: {result}")
    c.drawString(50, 700, f"Risk Probability: {prob*100:.2f}%")

    c.drawString(50, 660, "ML-based Clinical Assessment")
    c.drawString(50, 640, "Consult physician for confirmation")

    c.drawString(50, 600, f"Generated: {datetime.now()}")
    c.save()
    return filename

# ---------------- PATIENT INPUT ----------------
st.subheader("🧍 Patient Details")

with st.form("predict"):
    Age = st.number_input("Age", 1, 120, 30)
    Gender = st.selectbox("Gender", ["Female", "Male"])
    Polyuria = st.selectbox("Polyuria", [0,1])
    Polydipsia = st.selectbox("Polydipsia", [0,1])
    SuddenWeightLoss = st.selectbox("Sudden Weight Loss", [0,1])
    Polyphagia = st.selectbox("Polyphagia", [0,1])
    VisualBlurring = st.selectbox("Visual Blurring", [0,1])
    Irritability = st.selectbox("Irritability", [0,1])
    DelayedHealing = st.selectbox("Delayed Healing", [0,1])
    Obesity = st.selectbox("Obesity", [0,1])
    submit = st.form_submit_button("Predict")

if submit:
    Gender = 1 if Gender=="Male" else 0

    X_single = pd.DataFrame([[Age,DelayedHealing,SuddenWeightLoss,
        VisualBlurring,Obesity,Polyphagia,Polyuria,Gender,Polydipsia,Irritability]],
        columns=FEATURE_NAMES)

    pred = model.predict(X_single)[0]
    prob = model.predict_proba(X_single)[0][1]

    result = "Diabetic" if pred==1 else "Non-Diabetic"

    st.error(result) if pred==1 else st.success(result)

    cursor.execute("INSERT INTO predictions VALUES (?,?,?,?)",
        (Age,result,prob,datetime.now()))
    conn.commit()

    # -------- ROC POINT --------
    st.subheader("📈 Single Patient ROC")
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1],'--')
    ax.scatter(prob,prob,c='red')
    st.pyplot(fig)

    # -------- PDF --------
    if PDF_AVAILABLE:
        pdf = generate_pdf(Age,Gender,result,prob)
        with open(pdf,"rb") as f:
            st.download_button("⬇ Download Medical PDF", f, pdf)

    else:
        st.warning("PDF disabled (reportlab not installed)")

# ---------------- FEEDBACK ----------------
st.subheader("🔁 Doctor Feedback")
if submit:
    fb = st.radio("Doctor Confirmation",["Correct Prediction","Incorrect Prediction"])
    if st.button("Save Feedback"):
        label = 1 if fb=="Correct Prediction" else 0
        row = X_single.copy()
        row["label"] = label
        row.to_csv(FEEDBACK_FILE,mode="a",header=False,index=False)
        st.success("Feedback saved")

# ---------------- RETRAIN ----------------
if st.session_state.role=="Admin":
    st.subheader("🔄 Retrain Model")
    if st.button("Retrain"):
        fb = pd.read_csv(FEEDBACK_FILE)
        if len(fb)<20:
            st.warning("Need minimum 20 samples")
        else:
            model.fit(fb[FEATURE_NAMES], fb["label"])
            joblib.dump(model, MODEL_PATH)
            st.success("Model retrained")

# ---------------- FOOTER ----------------
st.markdown(f"""
<hr><div style='text-align:center'>
<b>Early Stage Diabetic Prediction</b><br>
By {researcher}<br>
Supervisor: {supervisor}
</div>
""", unsafe_allow_html=True)
