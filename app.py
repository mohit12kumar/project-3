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
from tensorflow.keras.models import load_model as load_keras_model
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Flight Prediction System", layout="wide")

st.title("✈ Flight Delay & Cancellation Prediction")

# ==============================
# LOAD SCALER
# ==============================
@st.cache_resource
def load_scaler(mode):
    try:
        if mode == "Delay":
            return joblib.load("scaler_delay.pkl")
        else:
            return joblib.load("scaler_cancel.pkl")
    except:
        return None

# ==============================
# ML MODEL LOADER
# ==============================
@st.cache_resource
def load_model(mode, model_name):

    delay_links = {
        "Random Forest": ("1kcjKFn-59lK1S8QHv1rHzcm4sL2VLTSU", "rf_delay.pkl"),
        "Decision Tree": ("1PZdtmAnt15nj1PC1rB8TO52RFAAzKQa_dI", "dt_delay.pkl"),
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

    links = delay_links if mode == "Delay" else cancel_links
    file_id, filename = links[model_name]

    if not os.path.exists(filename):
        st.warning(f"⬇ Downloading {model_name}...")
        gdown.download(id=file_id, output=filename, quiet=False)

    return joblib.load(filename)

# ==============================
# DL MODEL LOADER
# ==============================
@st.cache_resource
def load_dl_model(mode):
    if mode == "Delay":
        return load_keras_model("deep_delay_model.keras", compile=False)
    else:
        return load_keras_model("deep_cancel_model.keras", compile=False)

# ==============================
# SAFE PREDICT
# ==============================
def safe_predict(model, data):
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(data)[:,1]
            return (probs > 0.4).astype(int)
        return model.predict(data)
    except:
        return np.zeros(len(data))

# ==============================
# PREPROCESS
# ==============================
def preprocess_input(df):
    df.columns = [c.strip().lower() for c in df.columns]

    new_df = pd.DataFrame()

    new_df["airline"] = df.get("airline","UNK")
    new_df["origin"] = df.get("origin","UNK")
    new_df["dest"] = df.get("dest","UNK")

    new_df["dep_delay"] = pd.to_numeric(df.get("dep_delay",0), errors='coerce')
    new_df["distance"] = pd.to_numeric(df.get("distance",0), errors='coerce')
    new_df["crs_dep_time"] = pd.to_numeric(df.get("crs_dep_time",0), errors='coerce')

    new_df = new_df.fillna(0)

    if "fl_date" in df.columns:
        dt = pd.to_datetime(df["fl_date"], errors='coerce')
        new_df["month"] = dt.dt.month.fillna(1)
        new_df["day_of_week"] = dt.dt.dayofweek.fillna(0)
    else:
        new_df["month"] = 1
        new_df["day_of_week"] = 0

    new_df["is_weekend"] = new_df["day_of_week"].apply(lambda x:1 if x in [5,6] else 0)

    for col in ["airline","origin","dest"]:
        new_df[col] = new_df[col].astype("category").cat.codes

    return new_df

# ==============================
# UI
# ==============================
mode = st.selectbox("Prediction Type", ["Delay","Cancellation"])
model_type = st.selectbox("Model Type", ["Machine Learning", "Deep Learning"])

scaler_models = ["Logistic Regression", "KNN", "SVM"]

if model_type == "Machine Learning":
    model_choice = st.selectbox(
        "Select Model",
        ["XGBoost","Decision Tree","Logistic Regression","SVM","KNN","Random Forest"]
    )
    model = load_model(mode, model_choice)
else:
    model = load_dl_model(mode)

# ==============================
# CSV PREDICTION + ACCURACY
# ==============================
st.header("📂 Upload Dataset")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("🚀 Run Prediction"):

        df_new = preprocess_input(df)

        # APPLY SCALER IF NEEDED
        if model_type == "Machine Learning" and model_choice in scaler_models:
            scaler = load_scaler(mode)
            if scaler:
                df_new = pd.DataFrame(scaler.transform(df_new), columns=df_new.columns)

        if model_type == "Deep Learning":
            scaler = load_scaler(mode)
            if scaler:
                df_new = scaler.transform(df_new)

        # PREDICTION
        if model_type == "Machine Learning":
            preds = safe_predict(model, df_new)
        else:
            preds = (model.predict(df_new) > 0.5).astype(int).flatten()

        # OUTPUT
        flight_numbers = df.get("FL_NUMBER", range(len(df)))

        if mode == "Delay":
            result = ["Delayed" if x==1 else "On Time" for x in preds]
            output_df = pd.DataFrame({"FL_NUMBER":flight_numbers,"Delay":result})

            if "arr_delay" in df.columns:
                y_true = (df["arr_delay"] > 15).astype(int)
                acc = accuracy_score(y_true, preds)
                st.info(f"🎯 Accuracy: {acc:.2f}")

        else:
            result = ["Cancelled" if x==1 else "Not Cancelled" for x in preds]
            output_df = pd.DataFrame({"FL_NUMBER":flight_numbers,"Cancellation":result})

            if "cancelled" in df.columns:
                y_true = df["cancelled"]
                acc = accuracy_score(y_true, preds)
                st.info(f"🎯 Accuracy: {acc:.2f}")

        st.dataframe(output_df)
        st.download_button("📥 Download", output_df.to_csv(index=False))
