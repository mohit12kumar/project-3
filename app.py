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
    "Delay": {"Deep Learning": 0.9351},
    "Cancellation": {"Deep Learning": 0.9680}
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
# LOAD DL MODEL
# ==============================
@st.cache_resource
def load_dl_model(mode):
    return load_keras_model("deep_delay_model.keras", compile=False) if mode=="Delay" else load_keras_model("deep_cancel_model.keras", compile=False)

# ==============================
# PREPROCESS
# ==============================
def preprocess_input(df):
    df.columns = df.columns.str.lower().str.strip()

    new_df = pd.DataFrame()

    new_df["airline"] = df.get("airline", df.get("airline_code", "UNK"))
    new_df["origin"] = df.get("origin", "UNK")
    new_df["dest"] = df.get("dest", "UNK")

    new_df["dep_delay"] = pd.to_numeric(df.get("dep_delay", 0), errors="coerce")
    new_df["distance"] = pd.to_numeric(df.get("distance", 0), errors="coerce")
    new_df["crs_dep_time"] = pd.to_numeric(df.get("crs_dep_time", 0), errors="coerce")

    new_df = new_df.fillna(0)

    # DATE
    if "fl_date" in df.columns:
        dt = pd.to_datetime(df["fl_date"], errors="coerce")
        new_df["month"] = dt.dt.month.fillna(1)
        new_df["day_of_week"] = dt.dt.dayofweek.fillna(0)
    else:
        month_map = {
            "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
            "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
        }
        day_map = {
            "Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
            "Friday":4,"Saturday":5,"Sunday":6
        }

        new_df["month"] = df["month"].map(month_map) if "month" in df.columns else 1
        new_df["day_of_week"] = df["day_of_week"].map(day_map) if "day_of_week" in df.columns else 0

    new_df["is_weekend"] = new_df["day_of_week"].apply(lambda x:1 if x in [5,6] else 0)

    for col in ["airline","origin","dest"]:
        new_df[col] = new_df[col].astype("category").cat.codes
        new_df[col] = new_df[col].replace(-1,0)

    return new_df

# ==============================
# UI
# ==============================
mode = st.selectbox("Prediction Type", ["Delay","Cancellation"])

model = load_dl_model(mode)
acc = model_accuracy[mode]["Deep Learning"]
st.success(f"🎯 Model Accuracy: {acc*100:.2f}%")

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

month = st.selectbox("Month", ["January","February","March","April","May","June","July","August","September","October","November","December"])
day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

if st.button("🚀 Predict Single"):

    df = pd.DataFrame([[airline,origin,dest,dep_delay,distance,time,month,day]],
                      columns=["airline","origin","dest","dep_delay","distance","crs_dep_time","month","day_of_week"])

    df = preprocess_input(df)

    # 🔥 FIX DL INPUT
    df = df[["airline","origin","dest","dep_delay","distance","crs_dep_time","month","day_of_week","is_weekend"]]
    df = df.astype(float)

    scaler = load_scaler(mode)
    if scaler:
        df = scaler.transform(df)

    pred = (model.predict(df) > 0.35).astype(int)[0][0]

    result = "Delayed" if pred==1 else "On Time" if mode=="Delay" else ("Cancelled" if pred==1 else "Not Cancelled")

    st.success(f"✈ Prediction: {result}")

# ==============================
# CSV UPLOAD
# ==============================
st.header("📂 Upload Dataset")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower().str.strip()

    st.dataframe(df.head())

    if st.button("Run Prediction"):

        df_new = preprocess_input(df)

        # 🔥 FIX DL INPUT
        df_dl = df_new[["airline","origin","dest","dep_delay","distance","crs_dep_time","month","day_of_week","is_weekend"]]
        df_dl = df_dl.astype(float)

        scaler = load_scaler(mode)
        if scaler:
            df_dl = scaler.transform(df_dl)

        preds = (model.predict(df_dl) > 0.35).astype(int).flatten()

        # FL NUMBER FIX
        flight_no = df["fl_number"] if "fl_number" in df.columns else range(len(df))

        if mode=="Delay":
            result = ["Delayed" if x==1 else "On Time" for x in preds]
            out = pd.DataFrame({"FL_NUMBER":flight_no,"Delay":result})
        else:
            result = ["Cancelled" if x==1 else "Not Cancelled" for x in preds]
            out = pd.DataFrame({"FL_NUMBER":flight_no,"Cancellation":result})

        st.dataframe(out)
        st.download_button("📥 Download", out.to_csv(index=False), file_name="prediction.csv")
