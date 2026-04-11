# ==============================
# IMPORTS
# ==============================
import streamlit as st
import numpy as np
import joblib
import os
import gdown
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Flight Prediction System", layout="centered")
st.title("✈ Flight Delay & Cancellation Prediction")

# ==============================
# ALIGN FEATURES (FIXED)
# ==============================
def align_features(model, df):
    try:
        if hasattr(model, "feature_names_in_"):
            expected = model.feature_names_in_
            df = df[expected]   # strict match
    except:
        pass
    return df

# ==============================
# SAFE PREDICT (FIXED)
# ==============================
def safe_predict(model, data):

    if model is None:
        return np.zeros(len(data))

    try:
        # ALWAYS use numpy → avoids feature mismatch
        data = data.values

        return model.predict(data)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return np.zeros(len(data))

# ==============================
# MODEL LOADER (CACHED)
# ==============================
@st.cache_resource
def load_model(mode, model_name):

    delay_links = {
        "Random Forest": ("1kcjKFn-59lK1S8QHv1rHzcm4sL2VLTSU", "rf_delay.pkl"),
        "Decision Tree": ("1PZdtmAnt15nj1PC1rB8aW0kZIssgU8IM", "dt_delay.pkl"),
        "Logistic Regression": ("1cL9xaBH6WU_UlXAMpFlU8zenIpY7jgNf", "lr_delay.pkl"),
        "KNN": ("1hAMdiSjssNoXRGmzcLcnm8RsivyKStA2", "knn_delay.pkl"),
        "SVM": ("1pw_1yVInCY_N5prysDQT7_i78v2LblBU", "svm_delay.pkl"),
        "XGBoost": ("1B6gZvXZCizgN9j8C9sBxpeZhLc7UeJI4", "xgb_delay.pkl")
    }

    cancel_links = {
        "Random Forest": ("1AJxhnPsOL5VRtXqB8TO52RFAAzKQa_dI", "rf_cancel.pkl"),
        "Decision Tree": ("1VGat3BhFmQwkjrQKDUDPWW12_FHndjVv", "dt_cancel.pkl"),
        "Logistic Regression": ("16k7XQcInCTNuveWDSPiTLbhfgRPH2bUi", "lr_cancel.pkl"),
        "KNN": ("1qnC3xUyeJ8SDi455THh2_IbSmc4BVQgi", "knn_cancel.pkl"),
        "SVM": ("1ppy1emNTbhbi0YP0CxWu-cAJXXhNorNV", "svm_cancel.pkl"),
        "XGBoost": ("1pKYOfMvNr2qoVkOrbFin31MCX3jAkerf", "xgb_cancel.pkl")
    }

    def download(file_id, output):
        if not os.path.exists(output):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=True)

    try:
        links = delay_links if mode == "Delay" else cancel_links
        file_id, filename = links[model_name]

        download(file_id, filename)

        model = joblib.load(filename)
        return model

    except:
        return None

# ==============================
# ACCURACY
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

model_choice = st.selectbox("Select Model",
    ["Random Forest","Decision Tree","Logistic Regression","KNN","SVM","XGBoost"])

acc = delay_accuracy.get(model_choice) if mode=="Delay" else cancel_accuracy.get(model_choice)

st.success(f"{model_choice} ({mode})")
st.info(f"Accuracy: {acc}%")

model = load_model(mode, model_choice)

if model is None:
    st.error("❌ Model not available")

# ==============================
# SINGLE PREDICTION
# ==============================
st.header("🧍 Single Prediction")

airline = st.number_input("Airline")
origin = st.number_input("Origin")
dest = st.number_input("Destination")
dep_delay = st.number_input("Departure Delay")
distance = st.number_input("Distance")
crs_dep_time = st.number_input("CRS Dep Time")
month = st.number_input("Month",1,12)
day_of_week = st.number_input("Day of Week",0,6)
weekend = st.selectbox("Weekend",[0,1])

if st.button("Predict"):

    df = pd.DataFrame([[airline, origin, dest, dep_delay,
                        distance, crs_dep_time, month,
                        day_of_week, weekend]],
                      columns=["airline","origin","dest","dep_delay",
                               "distance","crs_dep_time","month",
                               "day_of_week","is_weekend"])

    df = align_features(model, df)

    pred = safe_predict(model, df)[0]

    if mode=="Delay":
        st.success("⚠ Delayed" if pred==1 else "✅ On Time")
    else:
        st.success("⚠ Cancelled" if pred==1 else "✅ Not Cancelled")

# ==============================
# CSV PREDICTION
# ==============================
st.header("📂 CSV Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("Run Prediction"):

        df.columns = [c.strip().lower() for c in df.columns]

        df_new = df.copy()

        if "is_weekend" not in df_new.columns:
            if "day_of_week" in df_new.columns:
                df_new["is_weekend"] = df_new["day_of_week"].apply(
                    lambda x: 1 if x in [5,6] else 0
                )
            else:
                df_new["is_weekend"] = 0

        df_new = align_features(model, df_new)

        preds = safe_predict(model, df_new)

        df["Result"] = ["Delayed" if x==1 else "On Time" for x in preds] if mode=="Delay" \
                       else ["Cancelled" if x==1 else "Not Cancelled" for x in preds]

        st.success("Prediction Complete")
        st.dataframe(df)

        st.download_button("Download CSV", df.to_csv(index=False), "output.csv")
