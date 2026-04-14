# ==============================
# IMPORTS
# ==============================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import gdown
from tensorflow.keras.models import load_model as load_keras_model
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Flight Prediction System", layout="wide")
st.title("✈ Flight Delay & Cancellation Prediction")

# ==============================
# ACCURACY
# ==============================
model_accuracy = {
    "Delay": {
        "Deep Learning": 0.9351,
        "XGBoost": 0.91,
        "Random Forest": 0.89,
        "Decision Tree": 0.85,
        "Logistic Regression": 0.88,
        "SVM": 0.87,
        "KNN": 0.86
    },
    "Cancellation": {
        "Deep Learning": 0.9680,
        "XGBoost": 0.95,
        "Random Forest": 0.94,
        "Decision Tree": 0.90,
        "Logistic Regression": 0.92,
        "SVM": 0.91,
        "KNN": 0.89
    }
}

# ==============================
# LOAD SCALER
# ==============================
@st.cache_resource
def load_scaler(mode):
    try:
        return joblib.load("scaler_delay.pkl") if mode=="Delay" else joblib.load("scaler_cancel.pkl")
    except:
        return None

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model(mode, model_name):

    delay_links = {
        "Random Forest": ("1kcjKFn-59lK1S8QHv1rHzcm4sL2VLTSU", "rf_delay.pkl"),
        "Decision Tree": ("1PZdtmAnt15nj1PC1rB8aW0kZIssgU8IM", "dt_delay.pkl"),
        "Logistic Regression": ("1cL9xaBH6WU_UlXAMpFlU8zenIpY7jgNf", "lr_delay.pkl"),
        "KNN": ("1hAMdiSjssNoXRGmzcLcnm8RsivyKStA2", "knn_delay.pkl"),
        "SVM": ("1pw_1yVInCY_N5prysDQT7_i78v2LblBU", "svm_delay.pkl"),
        "XGBoost": ("1GhFI6E5AflX1jPdiz3TRfKAVW1YSt1fm", "xgb_delay.pkl")
    }

    cancel_links = {
        "Random Forest": ("1AJxhnPsOL5VRtXqB8TO52RFAAzKQa_dI", "rf_cancel.pkl"),
        "Decision Tree": ("1VGat3BhFmQwkjrQKDUDPWW12_FHndjVv", "dt_cancel.pkl"),
        "Logistic Regression": ("16k7XQcInCTNuveWDSPiTLbhfgRPH2bUi", "lr_cancel.pkl"),
        "KNN": ("1qnC3xUyeJ8SDi455THh2_IbSmc4BVQgi", "knn_cancel.pkl"),
        "SVM": ("1ppy1emNTbhbi0YP0CxWu-cAJXXhNorNV", "svm_cancel.pkl"),
        "XGBoost": ("1pKYOfMvNr2qoVkOrbFin31MCX3jAkerf", "xgb_cancel.pkl")
    }

    if model_name == "Deep Learning":
        return load_keras_model(
            "deep_delay_model.keras" if mode=="Delay" else "deep_cancel_model.keras",
            compile=False
        )

    links = delay_links if mode=="Delay" else cancel_links

    try:
        file_id, filename = links[model_name]

        if not os.path.exists(filename):
            st.warning(f"⬇ Downloading {model_name}...")
            gdown.download(id=file_id, output=filename, quiet=False)

        return joblib.load(filename)

    except Exception as e:
        st.error(f"❌ {model_name} not available: {e}")
        return None

# ==============================
# PREPROCESS
# ==============================
def preprocess_input(df):
    df.columns = df.columns.str.lower().str.strip()

    new_df = pd.DataFrame()

    new_df["airline"] = df.get("airline", "UNK")
    new_df["origin"] = df.get("origin", "UNK")
    new_df["dest"] = df.get("dest", "UNK")

    new_df["dep_delay"] = pd.to_numeric(df.get("dep_delay",0), errors='coerce')
    new_df["distance"] = pd.to_numeric(df.get("distance",0), errors='coerce')
    new_df["crs_dep_time"] = pd.to_numeric(df.get("crs_dep_time",0), errors='coerce')

    new_df = new_df.fillna(0)

    new_df["month"] = 1
    new_df["day_of_week"] = 0
    new_df["is_weekend"] = 0

    for col in ["airline","origin","dest"]:
        new_df[col] = new_df[col].astype("category").cat.codes

    return new_df

# ==============================
# UI
# ==============================
mode = st.selectbox("Prediction Type", ["Delay","Cancellation"])

model_choice = st.selectbox(
    "Select Model",
    ["Deep Learning","XGBoost","Decision Tree","Logistic Regression","SVM","KNN","Random Forest"]
)

model = load_model(mode, model_choice)

acc = model_accuracy[mode][model_choice]
st.success(f"🎯 Accuracy: {acc*100:.2f}%")

# ==============================
# SINGLE PREDICTION
# ==============================
st.header("📊 Single Prediction")

airline = st.text_input("Airline","AA")
origin = st.text_input("Origin","JFK")
dest = st.text_input("Destination","LAX")
dep_delay = st.text_input("Departure Delay","10")
distance = st.text_input("Distance","1000")
time = st.text_input("Time","1400")

if st.button("Predict"):

    df = pd.DataFrame([[airline,origin,dest,dep_delay,distance,time]],
                      columns=["airline","origin","dest","dep_delay","distance","crs_dep_time"])

    df = preprocess_input(df)

    scaler = load_scaler(mode)

    if model_choice in ["KNN","SVM","Logistic Regression","Deep Learning"] and scaler:
        df = scaler.transform(df)

    if model_choice == "Deep Learning":
        pred = (model.predict(df) > 0.35).astype(int)[0][0]
    else:
        pred = model.predict(df)[0]

    result = "Delayed" if pred==1 else "On Time" if mode=="Delay" else ("Cancelled" if pred==1 else "Not Cancelled")

    st.success(result)

# ==============================
# CSV
# ==============================
st.header("📂 Upload CSV")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower().str.strip()

    if st.button("Run Prediction"):

        df_new = preprocess_input(df)

        scaler = load_scaler(mode)

        if model_choice in ["KNN","SVM","Logistic Regression","Deep Learning"] and scaler:
            df_new = scaler.transform(df_new)

        if model_choice == "Deep Learning":
            preds = (model.predict(df_new) > 0.35).astype(int).flatten()
        else:
            preds = model.predict(df_new)

        flight_no = df["fl_number"] if "fl_number" in df.columns else range(len(df))

        if mode=="Delay":
            result = ["Delayed" if x else "On Time" for x in preds]
            out = pd.DataFrame({"FL_NUMBER":flight_no,"Delay":result})
        else:
            result = ["Cancelled" if x else "Not Cancelled" for x in preds]
            out = pd.DataFrame({"FL_NUMBER":flight_no,"Cancellation":result})

        st.dataframe(out)
        st.download_button("Download", out.to_csv(index=False))
