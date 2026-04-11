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

st.write("🚀 App Loaded Successfully")

# ==============================
# ALIGN FEATURES
# ==============================
def align_features(model, df):
    try:
        expected = model.feature_names_in_
        df.columns = [c.lower() for c in df.columns]
        expected_lower = [c.lower() for c in expected]
        df = df[expected_lower]
        df.columns = expected
    except:
        pass
    return df

# ==============================
# SAFE PREDICT (NO CRASH)
# ==============================
def safe_predict(model, data):
    import numpy as np

    if model is None:
        return np.zeros(len(data))

    try:
        # XGBoost Safe Mode
        if "XGB" in str(type(model)):
            import xgboost as xgb
            booster = model.get_booster()
            dmatrix = xgb.DMatrix(data)
            preds = booster.predict(dmatrix)
            return (preds > 0.5).astype(int)
        else:
            return model.predict(data)

    except Exception as e:
        st.warning("⚠ Model not compatible in this environment")
        return np.zeros(len(data))

# ==============================
# MODEL LOADER (SAFE)
# ==============================
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
        "XGBoost": ("1SJa04KaD6Gjx8TwOjT_2C2_Q5br3gXAW", "xgb_cancel.pkl")
    }

    try:
        def download(file_id, output):
            if not os.path.exists(output):
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, output, quiet=True)

        links = delay_links if mode == "Delay" else cancel_links
        file_id, filename = links[model_name]

        download(file_id, filename)

        model = joblib.load(filename)
        return model

    except:
        st.warning(f"⚠ {model_name} failed to load")
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
    st.error("❌ Model not available, try another")

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

    df = pd.DataFrame([[
        airline, origin, dest, dep_delay,
        distance, crs_dep_time, month,
        day_of_week, weekend
    ]], columns=[
        "airline","origin","dest","dep_delay",
        "distance","crs_dep_time","month",
        "day_of_week","is_weekend"
    ])

    df = align_features(model, df)

    pred = safe_predict(model, df)[0]

    if mode=="Delay":
        st.success("⚠ Delayed" if pred==1 else "✅ On Time")
    else:
        st.success("⚠ Cancelled" if pred==1 else "✅ Not Cancelled")

# ==============================
# SMART CSV PREDICTION
# ==============================
st.header("📂 Smart CSV Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("Run Prediction"):

        df.columns = [c.strip().lower() for c in df.columns]

        mapping = {
            "airline": ["airline","carrier"],
            "origin": ["origin","source"],
            "dest": ["dest","destination"],
            "dep_delay": ["dep_delay","delay"],
            "distance": ["distance"],
            "crs_dep_time": ["crs_dep_time","time"],
            "month": ["month"],
            "day_of_week": ["day_of_week","day"],
            "is_weekend": ["is_weekend","weekend"]
        }

        new_df = {}

        for key, aliases in mapping.items():
            for col in df.columns:
                if col in aliases:
                    new_df[key] = df[col]
                    break
            else:
                new_df[key] = 0

        df_new = pd.DataFrame(new_df)

        if df_new["is_weekend"].eq(0).all():
            df_new["is_weekend"] = df_new["day_of_week"].apply(
                lambda x: 1 if x in [5,6] else 0
            )

        df_new = align_features(model, df_new)

        preds = safe_predict(model, df_new)

        if mode=="Delay":
            df["Result"] = ["Delayed" if x==1 else "On Time" for x in preds]
        else:
            df["Result"] = ["Cancelled" if x==1 else "Not Cancelled" for x in preds]

        st.success("Prediction Complete")
        st.dataframe(df)

        st.download_button("Download CSV", df.to_csv(index=False), "output.csv")
