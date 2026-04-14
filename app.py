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
from tensorflow.keras.models import load_model as load_keras_model  # 🔥 FIXED

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Flight Prediction System", layout="wide")

st.markdown("<h2 style='text-align:center;'>✈ Flight Delay & Cancellation Prediction</h2>", unsafe_allow_html=True)

# ==============================
# ML MODEL LOADER (UNCHANGED)
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

    current_key = f"{mode}_{model_name}"

    if "loaded_model_key" in st.session_state:
        if st.session_state["loaded_model_key"] != current_key:
            st.session_state.pop("model", None)

    if "model" not in st.session_state:

        if not os.path.exists(filename):
            st.warning(f"⬇ Downloading {model_name}...")
            gdown.download(id=file_id, output=filename, quiet=False)

        st.session_state["model"] = joblib.load(filename)
        st.session_state["loaded_model_key"] = current_key

    return st.session_state["model"]

# ==============================
# DL MODEL LOADER (FIXED)
# ==============================
@st.cache_resource
def load_dl_model(mode):
    if mode == "Delay":
        return load_keras_model("deep_delay_model.keras")
    else:
        return load_keras_model("deep_cancel_model(1).keras")

# ==============================
# SAFE PREDICT (UNCHANGED)
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
# ALIGN FEATURES (UNCHANGED)
# ==============================
def align_features(model, df):
    try:
        if hasattr(model, "feature_names_in_"):
            df = df[model.feature_names_in_]
    except:
        pass
    return df

# ==============================
# PREPROCESS SINGLE INPUT
# ==============================
def preprocess_single_input(df):

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
        df[col] = df[col].astype("category").cat.codes

    return df

# ==============================
# PREPROCESS CSV
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

if model_type == "Machine Learning":
    model_choice = st.selectbox(
        "Select Model",
        ["XGBoost","Decision Tree","Logistic Regression","SVM","KNN","Random Forest"]
    )
    model = load_model(mode, model_choice)
else:
    model = load_dl_model(mode)

# ==============================
# SINGLE PREDICTION
# ==============================
st.header("📊 Single Prediction")

airline = st.text_input("Airline", placeholder="AA")
origin = st.text_input("Origin", placeholder="JFK")
dest = st.text_input("Destination", placeholder="LAX")
dep_delay = st.text_input("Departure Delay", placeholder="15")
distance = st.text_input("Distance", placeholder="1200")
crs_dep_time = st.text_input("Time", placeholder="1400")

month = st.selectbox("Month", ["January","February","March","April","May","June","July","August","September","October","November","December"])
day_of_week = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

if st.button("🚀 Predict"):

    df = pd.DataFrame([[airline,origin,dest,dep_delay,distance,crs_dep_time,month,day_of_week]],
                      columns=["airline","origin","dest","dep_delay","distance","crs_dep_time","month","day_of_week"])

    df = preprocess_single_input(df)

    if model_type == "Machine Learning":
        df = align_features(model, df)
        pred = safe_predict(model, df)[0]
    else:
        pred = (model.predict(df.values) > 0.5).astype(int)[0][0]

    if mode == "Delay":
        st.error("✈️ Flight DELAYED") if pred==1 else st.success("✈️ Flight ON TIME")
    else:
        st.error("✈️ Flight CANCELLED") if pred==1 else st.success("✈️ Flight NOT CANCELLED")

# ==============================
# CSV PREDICTION
# ==============================
# ==============================
# CSV PREDICTION (UPDATED)
# ==============================
st.header("📂 Upload Dataset")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write("Original Data Preview")
    st.dataframe(df.head())

    if st.button("🚀 Run Prediction"):

        try:
            # ✅ ALWAYS KEEP ORIGINAL FL_NUMBER
            if "FL_NUMBER" in df.columns:
                flight_numbers = df["FL_NUMBER"].copy()
            elif "fl_number" in df.columns:
                flight_numbers = df["fl_number"].copy()
            else:
                flight_numbers = pd.Series(range(1, len(df)+1))

            # ✅ Preprocess (DON’T TOUCH ORIGINAL df)
            df_new = preprocess_input(df)
            
            if model_type == "Machine Learning":
                df_new = align_features(model, df_new)
                preds = safe_predict(model, df_new)
            else:
                preds = (model.predict(df_new.values) > 0.5).astype(int).flatten()

            # ==============================
            # OUTPUT
            # ==============================
            if mode == "Delay":

                result = ["Delayed" if x==1 else "On Time" for x in preds]

                output_df = pd.DataFrame({
                    "FL_NUMBER": flight_numbers.values,
                    "Delay": result
                })

            else:

                result = ["Cancelled" if x==1 else "Not Cancelled" for x in preds]

                output_df = pd.DataFrame({
                    "FL_NUMBER": flight_numbers.values,
                    "Cancellation": result
                })

            st.success("✅ Prediction Complete")

            # ✅ SHOW RESULT
            st.dataframe(output_df)

            # ✅ DOWNLOAD
            st.download_button(
                "📥 Download Result",
                output_df.to_csv(index=False),
                "final_output.csv"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
