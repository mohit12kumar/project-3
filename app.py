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
from sklearn.metrics import accuracy_score
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
        if mode == "Delay":
            return joblib.load("scaler_delay.pkl")
        else:
            return joblib.load("scaler_cancel.pkl")
    except:
        return None

# ==============================
# LOAD ML MODEL
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

    links = delay_links if mode == "Delay" else cancel_links
    file_id, filename = links[model_name]

    if not os.path.exists(filename):
        gdown.download(id=file_id, output=filename, quiet=False)

    return joblib.load(filename)

# ==============================
# LOAD DL MODEL
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
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(data)[:,1]
        return (probs > 0.4).astype(int)
    return model.predict(data)

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

    # month/day mapping
    month_map = {
        "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
        "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
    }

    day_map = {
        "Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
        "Friday":4,"Saturday":5,"Sunday":6
    }

    new_df["month"] = df.get("month","January").map(month_map)
    new_df["day_of_week"] = df.get("day_of_week","Monday").map(day_map)

    new_df["is_weekend"] = new_df["day_of_week"].apply(lambda x:1 if x in [5,6] else 0)

    for col in ["airline","origin","dest"]:
        new_df[col] = new_df[col].astype("category").cat.codes

    return new_df

# ==============================
# UI SELECTION
# ==============================
mode = st.selectbox("Prediction Type", ["Delay","Cancellation"])
model_type = st.selectbox("Model Type", ["Machine Learning","Deep Learning"])

if model_type == "Machine Learning":
    model_choice = st.selectbox("Select Model",
        ["XGBoost","Decision Tree","Logistic Regression","SVM","KNN","Random Forest"]
    )
    model = load_model(mode, model_choice)
    acc = model_accuracy[mode].get(model_choice)
else:
    model = load_dl_model(mode)
    acc = model_accuracy[mode]["Deep Learning"]

st.success(f"🎯 Model Accuracy: {acc*100:.2f}%")

# ==============================
# SINGLE PREDICTION (ALL MODELS)
# ==============================
st.header("📊 Single Prediction")

airline = st.text_input("Airline","AA")
origin = st.text_input("Origin","JFK")
dest = st.text_input("Destination","LAX")
dep_delay = st.text_input("Departure Delay","10")
distance = st.text_input("Distance","1000")
time = st.text_input("Time","1400")

month = st.selectbox("Month", list({
    "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
    "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
}.keys()))

day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

if st.button("🚀 Predict Single"):

    df = pd.DataFrame([[airline,origin,dest,dep_delay,distance,time,month,day]],
                      columns=["airline","origin","dest","dep_delay","distance","crs_dep_time","month","day_of_week"])

    df = preprocess_input(df)

    results = {}

    for m_name in ["XGBoost","Decision Tree","Logistic Regression","SVM","KNN","Random Forest"]:
        try:
            m = load_model(mode, m_name)
            df_temp = df.copy()

            if m_name in ["Logistic Regression","KNN","SVM"]:
                scaler = load_scaler(mode)
                if scaler:
                    df_temp = pd.DataFrame(scaler.transform(df_temp), columns=df_temp.columns)

            pred = safe_predict(m, df_temp)[0]
            results[m_name] = pred
        except:
            results[m_name] = "Error"

    # DL
    try:
        dl = load_dl_model(mode)
        scaler = load_scaler(mode)
        df_dl = scaler.transform(df) if scaler else df.values
        pred = (dl.predict(df_dl) > 0.5).astype(int)[0][0]
        results["Deep Learning"] = pred
    except:
        results["Deep Learning"] = "Error"

    final = {}
    for k,v in results.items():
        if v == "Error":
            final[k] = "Error"
        else:
            if mode == "Delay":
                final[k] = "Delayed" if v==1 else "On Time"
            else:
                final[k] = "Cancelled" if v==1 else "Not Cancelled"

    st.dataframe(pd.DataFrame(final.items(), columns=["Model","Prediction"]))

# ==============================
# CSV UPLOAD
# ==============================
st.header("📂 Upload Dataset")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        df_new = preprocess_input(df)

        if model_type == "Machine Learning" and model_choice in ["Logistic Regression","KNN","SVM"]:
            scaler = load_scaler(mode)
            if scaler:
                df_new = pd.DataFrame(scaler.transform(df_new), columns=df_new.columns)

        if model_type == "Deep Learning":
            scaler = load_scaler(mode)
            if scaler:
                df_new = scaler.transform(df_new)

        preds = safe_predict(model, df_new) if model_type=="Machine Learning" else (model.predict(df_new)>0.5).astype(int).flatten()

        flight_no = df.get("FL_NUMBER", range(len(df)))

        if mode=="Delay":
            result = ["Delayed" if x==1 else "On Time" for x in preds]
            out = pd.DataFrame({"FL_NUMBER":flight_no,"Delay":result})
        else:
            result = ["Cancelled" if x==1 else "Not Cancelled" for x in preds]
            out = pd.DataFrame({"FL_NUMBER":flight_no,"Cancellation":result})

        st.dataframe(out)
        st.download_button("Download", out.to_csv(index=False))
