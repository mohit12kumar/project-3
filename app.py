# ==============================
# IMPORTS
# ==============================
import streamlit as st
import numpy as np
import joblib
import os
import gdown
import pandas as pd

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Flight Prediction System", layout="centered")
st.title("✈ Flight Delay & Cancellation Prediction")

st.info("⏳ Loading models (first time only)...")

# ==============================
# DOWNLOAD FUNCTION
# ==============================
def download(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=True)

# ==============================
# LOAD ALL MODELS (CACHED)
# ==============================
@st.cache_resource
def load_all_models():

    # ===== DELAY MODELS =====
    download("1kcjKFn-59lK1S8QHv1rHzcm4sL2VLTSU", "rf_delay.pkl")
    download("1PZdtmAnt15nj1PC1rB8aW0kZIssgU8IM", "dt_delay.pkl")
    download("1cL9xaBH6WU_UlXAMpFlU8zenIpY7jgNf", "lr_delay.pkl")
    download("1hAMdiSjssNoXRGmzcLcnm8RsivyKStA2", "knn_delay.pkl")
    download("1pw_1yVInCY_N5prysDQT7_i78v2LblBU", "svm_delay.pkl")
    download("1B6gZvXZCizgN9j8C9sBxpeZhLc7UeJI4", "xgb_delay.pkl")

    # ===== CANCEL MODELS =====
    download("1AJxhnPsOL5VRtXqB8TO52RFAAzKQa_dI", "rf_cancel.pkl")
    download("1VGat3BhFmQwkjrQKDUDPWW12_FHndjVv", "dt_cancel.pkl")
    download("16k7XQcInCTNuveWDSPiTLbhfgRPH2bUi", "lr_cancel.pkl")
    download("1qnC3xUyeJ8SDi455THh2_IbSmc4BVQgi", "knn_cancel.pkl")
    download("1ppy1emNTbhbi0YP0CxWu-cAJXXhNorNV", "svm_cancel.pkl")
    download("1SJa04KaD6Gjx8TwOjT_2C2_Q5br3gXAW", "xgb_cancel.pkl")

    # ===== LOAD MODELS =====
    def safe_load(path):
        try:
            return joblib.load(path)
        except:
            return None

    delay_models = {
        "Random Forest": safe_load("rf_delay.pkl"),
        "Decision Tree": safe_load("dt_delay.pkl"),
        "Logistic Regression": safe_load("lr_delay.pkl"),
        "KNN": safe_load("knn_delay.pkl"),
        "SVM": safe_load("svm_delay.pkl"),
        "XGBoost": safe_load("xgb_delay.pkl")
    }

    cancel_models = {
        "Random Forest": safe_load("rf_cancel.pkl"),
        "Decision Tree": safe_load("dt_cancel.pkl"),
        "Logistic Regression": safe_load("lr_cancel.pkl"),
        "KNN": safe_load("knn_cancel.pkl"),
        "SVM": safe_load("svm_cancel.pkl"),
        "XGBoost": safe_load("xgb_cancel.pkl")
    }

    return delay_models, cancel_models


# ==============================
# LOAD MODELS
# ==============================
delay_models, cancel_models = load_all_models()

# ==============================
# MODE
# ==============================
mode = st.selectbox("Select Prediction Type", ["Delay", "Cancellation"])
models_dict = delay_models if mode == "Delay" else cancel_models

# ==============================
# INPUT
# ==============================
st.header("🧍 Single Prediction")

airline = st.number_input("Airline", step=1)
origin = st.number_input("Origin", step=1)
dest = st.number_input("Destination", step=1)
dep_delay = st.number_input("Departure Delay")
distance = st.number_input("Distance")
crs_dep_time = st.number_input("CRS Dep Time")
month = st.number_input("Month", 1, 12)
day_of_week = st.number_input("Day of Week", 0, 6)
weekend = st.selectbox("Weekend", [0, 1])

model_choice = st.selectbox("Select Model", list(models_dict.keys()))

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):

    data = np.array([[airline, origin, dest, dep_delay,
                      distance, crs_dep_time, month,
                      day_of_week, weekend]])

    model = models_dict[model_choice]

    if model is None:
        st.error("❌ Model failed to load")
    else:
        pred = model.predict(data)[0]

        if mode == "Delay":
            st.success("⚠ Delayed" if pred == 1 else "✅ On Time")
        else:
            st.success("⚠ Cancelled" if pred == 1 else "✅ Not Cancelled")
