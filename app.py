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

# ==============================
# MODEL LOADER
# ==============================
def load_model(mode, model_name):

    def download(file_id, output):
        if os.path.exists(output):
            os.remove(output)

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=True)

    # DELAY MODELS
    delay_links = {
        "Random Forest": ("1kcjKFn-59lK1S8QHv1rHzcm4sL2VLTSU", "rf_delay.pkl"),
        "Decision Tree": ("1PZdtmAnt15nj1PC1rB8aW0kZIssgU8IM", "dt_delay.pkl"),
        "Logistic Regression": ("1cL9xaBH6WU_UlXAMpFlU8zenIpY7jgNf", "lr_delay.pkl"),
        "KNN": ("1hAMdiSjssNoXRGmzcLcnm8RsivyKStA2", "knn_delay.pkl"),
        "SVM": ("1pw_1yVInCY_N5prysDQT7_i78v2LblBU", "svm_delay.pkl"),
        "XGBoost": ("1B6gZvXZCizgN9j8C9sBxpeZhLc7UeJI4", "xgb_delay.pkl")
    }

    # CANCELLATION MODELS
    cancel_links = {
        "Random Forest": ("1AJxhnPsOL5VRtXqB8TO52RFAAzKQa_dI", "rf_cancel.pkl"),
        "Decision Tree": ("1VGat3BhFmQwkjrQKDUDPWW12_FHndjVv", "dt_cancel.pkl"),
        "Logistic Regression": ("16k7XQcInCTNuveWDSPiTLbhfgRPH2bUi", "lr_cancel.pkl"),
        "KNN": ("1qnC3xUyeJ8SDi455THh2_IbSmc4BVQgi", "knn_cancel.pkl"),
        "SVM": ("1ppy1emNTbhbi0YP0CxWu-cAJXXhNorNV", "svm_cancel.pkl"),
        "XGBoost": ("1SJa04KaD6Gjx8TwOjT_2C2_Q5br3gXAW", "xgb_cancel.pkl")
    }

    links = delay_links if mode == "Delay" else cancel_links
    file_id, filename = links[model_name]

    download(file_id, filename)

    try:
        model = joblib.load(filename)

        # Fix old XGBoost issue
        if hasattr(model, "use_label_encoder"):
            model.use_label_encoder = False

        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None


# ==============================
# ACCURACY DATA
# ==============================
delay_accuracy = {
    "Random Forest": 91.90,
    "SVM": 91.89,
    "Decision Tree": 91.28,
    "Logistic Regression": 91.05,
    "KNN": 92.71,
    "XGBoost": 89.45
}

cancel_accuracy = {
    "Random Forest": 96.73,
    "XGBoost": 95.21,
    "Decision Tree": 97.11,
    "KNN": 97.13,
    "SVM": 43.80,
    "Logistic Regression": 43.10
}

# ==============================
# MODE + MODEL
# ==============================
mode = st.selectbox("Select Prediction Type", ["Delay", "Cancellation"])

model_list = ["Random Forest","Decision Tree","Logistic Regression","KNN","SVM","XGBoost"]
model_choice = st.selectbox("Select Model", model_list)

acc = delay_accuracy.get(model_choice) if mode == "Delay" else cancel_accuracy.get(model_choice)

st.success(f"✅ Using Model: {model_choice} ({mode})")
st.info(f"📊 Model Accuracy: {acc}%")

model = load_model(mode, model_choice)

# ==============================
# SINGLE PREDICTION
# ==============================
st.header("🧍 Single Prediction")

col1, col2 = st.columns(2)

with col1:
    airline = st.number_input("Airline", step=1)
    origin = st.number_input("Origin", step=1)
    dest = st.number_input("Destination", step=1)
    dep_delay = st.number_input("Departure Delay")

with col2:
    distance = st.number_input("Distance")
    crs_dep_time = st.number_input("CRS Dep Time")
    month = st.number_input("Month", 1, 12)
    day_of_week = st.number_input("Day of Week", 0, 6)

weekend = st.selectbox("Weekend", [0, 1])

if st.button("Predict"):
    if model is not None:

        data = pd.DataFrame([[
            airline, origin, dest, dep_delay,
            distance, crs_dep_time, month,
            day_of_week, weekend
        ]], columns=[
            "airline","origin","dest","dep_delay",
            "distance","crs_dep_time","month",
            "day_of_week","is_weekend"
        ])

        pred = model.predict(data)[0]

        if mode == "Delay":
            st.success("⚠ Flight Delayed" if pred == 1 else "✅ Flight On Time")
        else:
            st.success("⚠ Flight Cancelled" if pred == 1 else "✅ Not Cancelled")

    else:
        st.error("Model not loaded")


# ==============================
# BATCH PREDICTION
# ==============================
st.header("📂 Batch Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):

        # 🔥 FIX COLUMN CASE (LOWERCASE)
        df.columns = [col.strip().lower() for col in df.columns]

        # Fix naming
        df = df.rename(columns={"weekend": "is_weekend"})

        # Create missing column
        if "is_weekend" not in df.columns:
            df["is_weekend"] = df["day_of_week"].apply(
                lambda x: 1 if x in [5, 6] else 0
            )

        features = [
            "airline","origin","dest","dep_delay",
            "distance","crs_dep_time","month",
            "day_of_week","is_weekend"
        ]

        df_model = df[features]

        preds = model.predict(df_model)

        if mode == "Delay":
            df["Result"] = ["Delayed" if x==1 else "On Time" for x in preds]
        else:
            df["Result"] = ["Cancelled" if x==1 else "Not Cancelled" for x in preds]

        st.success("✅ Batch Prediction Done")
        st.dataframe(df)

        st.download_button("Download CSV", df.to_csv(index=False), "output.csv")
