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
st.set_page_config(page_title="Flight Prediction System", layout="wide")

st.markdown(
    "<h2 style='text-align: center; color: #2E86C1;'>✈ Flight Delay & Cancellation Prediction System</h2>",
    unsafe_allow_html=True
)

# ==============================
# ALIGN FEATURES
# ==============================
def align_features(model, df):
    try:
        if hasattr(model, "feature_names_in_"):
            df = df[model.feature_names_in_]
    except:
        pass
    return df

# ==============================
# SAFE PREDICT
# ==============================
def safe_predict(model, data):
    try:
        data = data.values

        if "XGB" in str(type(model)):
            try:
                return model.predict(data)
            except:
                import xgboost as xgb
                dmatrix = xgb.DMatrix(data)
                preds = model.get_booster().predict(dmatrix)
                return (preds > 0.4).astype(int)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(data)[:, 1]
            return (probs > 0.4).astype(int)

        return model.predict(data)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return np.zeros(len(data))

# ==============================
# MODEL LOADER (ONE MODEL ONLY)
# ==============================
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

    links = delay_links if mode == "Delay" else cancel_links
    file_id, filename = links[model_name]

    current_key = f"{mode}_{model_name}"

    if "loaded_model_key" in st.session_state:
        if st.session_state["loaded_model_key"] != current_key:
            st.session_state.pop("model", None)

    if "model" not in st.session_state:

        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            st.warning(f"⬇ Downloading {model_name}...")
            gdown.download(url, filename, quiet=False)
        else:
            st.success(f"⚡ Using cached file: {filename}")

        st.session_state["model"] = joblib.load(filename)
        st.session_state["loaded_model_key"] = current_key

    return st.session_state["model"]

# ==============================
# PREPROCESS INPUT
# ==============================
def preprocess_input(df):

    for col in ["dep_delay","distance","crs_dep_time"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    month_map = {
        "january":1,"february":2,"march":3,"april":4,
        "may":5,"june":6,"july":7,"august":8,
        "september":9,"october":10,"november":11,"december":12
    }

    df["month"] = df["month"].astype(str).str.lower().map(month_map).fillna(1)

    day_map = {
        "monday":0,"tuesday":1,"wednesday":2,
        "thursday":3,"friday":4,"saturday":5,"sunday":6
    }

    df["day_of_week"] = df["day_of_week"].astype(str).str.lower().map(day_map).fillna(0)

    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in [5,6] else 0)

    for col in ["airline","origin","dest"]:
        df[col] = df[col].astype(str).astype("category").cat.codes

    return df

# ==============================
# UI
# ==============================
mode = st.selectbox("Select Prediction Type", ["Delay", "Cancellation"])

model_choice = st.selectbox(
    "Select Model",
    ["Random Forest","Decision Tree","Logistic Regression","KNN","SVM","XGBoost"]
)

# ==============================
# ACCURACY DISPLAY
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

acc = delay_accuracy.get(model_choice) if mode=="Delay" else cancel_accuracy.get(model_choice)

st.metric(label=f"{model_choice} ({mode}) Accuracy", value=f"{acc}%")

# Load model
model = load_model(mode, model_choice)

# ==============================
# INPUT
# ==============================
st.header("📊 Enter Flight Details")

col1, col2 = st.columns(2)

with col1:
    airline = st.text_input("Airline")
    origin = st.text_input("Origin")
    dest = st.text_input("Destination")
    dep_delay = st.text_input("Departure Delay")

with col2:
    distance = st.text_input("Distance")
    crs_dep_time = st.text_input("Scheduled Departure Time")

    month = st.selectbox("Month",
        ["January","February","March","April","May","June",
         "July","August","September","October","November","December"])

    day_of_week = st.selectbox("Day",
        ["Monday","Tuesday","Wednesday","Thursday",
         "Friday","Saturday","Sunday"])

# ==============================
# PREDICTION
# ==============================
if st.button("🚀 Predict Now"):

    df = pd.DataFrame([[airline, origin, dest, dep_delay,
                        distance, crs_dep_time, month,
                        day_of_week]],
                      columns=["airline","origin","dest","dep_delay",
                               "distance","crs_dep_time","month",
                               "day_of_week"])

    df = preprocess_input(df)
    df = align_features(model, df)

    pred = safe_predict(model, df)[0]

    st.subheader("📢 Result")

    if mode == "Delay":
        st.error("✈️ Flight will be DELAYED") if pred==1 else st.success("✈️ Flight ON TIME")
    else:
        st.error("✈️ Flight will be CANCELLED") if pred==1 else st.success("✈️ Flight NOT CANCELLED")

# ==============================
# CSV
# ==============================
st.header("📂 Upload CSV")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("Run Prediction"):

        df.columns = [c.strip().lower() for c in df.columns]

        required = ["airline","origin","dest","dep_delay",
                    "distance","crs_dep_time","month","day_of_week"]

        for col in required:
            if col not in df.columns:
                df[col] = ""

        df_new = df[required]

        df_new = preprocess_input(df_new)
        df_new = align_features(model, df_new)

        preds = safe_predict(model, df_new)

        df["Result"] = ["Delayed" if x==1 else "On Time" for x in preds] if mode=="Delay" \
                       else ["Cancelled" if x==1 else "Not Cancelled" for x in preds]

        st.success("Done")
        st.dataframe(df)

        st.download_button("Download CSV", df.to_csv(index=False), "output.csv")
