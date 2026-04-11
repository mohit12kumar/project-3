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
# DOWNLOAD FUNCTION
# ==============================
def download(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=True)

# ==============================
# DOWNLOAD DELAY MODELS
# ==============================
download("1kcjKFn-59lK1S8QHv1rHzcm4sL2VLTSU", "rf_delay.pkl")
download("1PZdtmAnt15nj1PC1rB8aW0kZIssgU8IM", "dt_delay.pkl")
download("1cL9xaBH6WU_UlXAMpFlU8zenIpY7jgNf", "lr_delay.pkl")
download("1hAMdiSjssNoXRGmzcLcnm8RsivyKStA2", "knn_delay.pkl")
download("1pw_1yVInCY_N5prysDQT7_i78v2LblBU", "svm_delay.pkl")
download("1B6gZvXZCizgN9j8C9sBxpeZhLc7UeJI4", "xgb_delay.pkl")

# ==============================
# DOWNLOAD CANCELLATION MODELS
# ==============================
download("1VGat3BhFmQwkjrQKDUDPWW12_FHndjVv", "dt_cancel.pkl")
download("1qnC3xUyeJ8SDi455THh2_IbSmc4BVQgi", "knn_cancel.pkl")
download("1ppy1emNTbhbi0YP0CxWu-cAJXXhNorNV", "svm_cancel.pkl")
download("16k7XQcInCTNuveWDSPiTLbhfgRPH2bUi", "lr_cancel.pkl")
download("1AJxhnPsOL5VRtXqB8TO52RFAAzKQa_dI", "rf_cancel.pkl")
download("1SJa04KaD6Gjx8TwOjT_2C2_Q5br3gXAW", "xgb_cancel.pkl")

# ==============================
# LOAD MODELS
# ==============================
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

delay_models = {k: v for k, v in delay_models.items() if v is not None}
cancel_models = {k: v for k, v in cancel_models.items() if v is not None}

# ==============================
# METRICS (DELAY)
# ==============================
delay_accuracy = {
    "Deep Learning": 93.51, "Random Forest": 91.90, "SVM": 91.89,
    "Decision Tree": 91.29, "Logistic Regression": 91.05,
    "KNN": 92.71, "XGBoost": 89.45
}

delay_precision = {
    "Deep Learning": 84.89, "Random Forest": 74.22, "SVM": 74.31,
    "Decision Tree": 71.53, "Logistic Regression": 70.65,
    "KNN": 89.95, "XGBoost": 64.85
}

delay_recall = {
    "Deep Learning": 75.87, "Random Forest": 81.25, "SVM": 80.95,
    "Decision Tree": 82.13, "Logistic Regression": 82.26,
    "KNN": 65.01, "XGBoost": 84.69
}

delay_f1 = {
    "Deep Learning": 80.13, "Random Forest": 77.58, "SVM": 77.49,
    "Decision Tree": 76.47, "Logistic Regression": 76.02,
    "KNN": 75.47, "XGBoost": 73.46
}

# ==============================
# METRICS (CANCELLATION)
# ==============================
cancel_accuracy = {
    "Random Forest": 96.73, "XGBoost": 95.21, "Deep Learning": 96.98,
    "Decision Tree": 97.11, "KNN": 97.13, "SVM": 43.80,
    "Logistic Regression": 43.10
}

cancel_precision = {
    "Random Forest": 42.95, "XGBoost": 35.18, "Deep Learning": 44.22,
    "Decision Tree": 45.75, "KNN": 36.04, "SVM": 3.13,
    "Logistic Regression": 3.10
}

cancel_recall = {
    "Random Forest": 74.43, "XGBoost": 98.09, "Deep Learning": 56.75,
    "Decision Tree": 53.58, "KNN": 12.09, "SVM": 68.21,
    "Logistic Regression": 68.59
}

cancel_f1 = {
    "Random Forest": 54.47, "XGBoost": 51.79, "Deep Learning": 49.71,
    "Decision Tree": 49.36, "KNN": 18.10, "SVM": 5.98,
    "Logistic Regression": 5.95
}

# ==============================
# MODE SELECTION
# ==============================
mode = st.selectbox("Select Prediction Type", ["Delay", "Cancellation"])

models_dict = delay_models if mode == "Delay" else cancel_models
accuracy_dict = delay_accuracy if mode == "Delay" else cancel_accuracy
precision_dict = delay_precision if mode == "Delay" else cancel_precision
recall_dict = delay_recall if mode == "Delay" else cancel_recall
f1_dict = delay_f1 if mode == "Delay" else cancel_f1

# ==============================
# INPUT UI
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
# SHOW METRICS
# ==============================
if model_choice in accuracy_dict:
    st.success(f"📊 Accuracy: {accuracy_dict[model_choice]}%")
    st.info(f"🎯 Precision: {precision_dict[model_choice]}%")
    st.warning(f"🔁 Recall: {recall_dict[model_choice]}%")
    st.write(f"📌 F1 Score: {f1_dict[model_choice]}%")

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):
    data = np.array([[airline, origin, dest, dep_delay,
                      distance, crs_dep_time, month,
                      day_of_week, weekend]])

    model = models_dict[model_choice]
    pred = model.predict(data)[0]

    if mode == "Delay":
        st.success("⚠ Delayed" if pred == 1 else "✅ On Time")
    else:
        st.success("⚠ Cancelled" if pred == 1 else "✅ Not Cancelled")

# ==============================
# BATCH PREDICTION
# ==============================
st.header("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if st.button("Predict Batch"):

        df.columns = [col.upper() for col in df.columns]

        required = [
            "AIRLINE", "ORIGIN", "DEST", "DEP_DELAY",
            "DISTANCE", "CRS_DEP_TIME", "MONTH",
            "DAY_OF_WEEK", "WEEKEND"
        ]

        if "WEEKEND" not in df.columns and "DAY_OF_WEEK" in df.columns:
            df["WEEKEND"] = df["DAY_OF_WEEK"].apply(lambda x: 1 if x in [5, 6] else 0)

        df_model = df[required]

        model = models_dict[model_choice]
        preds = model.predict(df_model)

        if mode == "Delay":
            df["RESULT"] = ["Delayed" if x == 1 else "On Time" for x in preds]
        else:
            df["RESULT"] = ["Cancelled" if x == 1 else "Not Cancelled" for x in preds]

        st.dataframe(df)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download", csv, "results.csv")

# ==============================
# PERFORMANCE TABLE
# ==============================
st.subheader(f"📊 {mode} Model Performance")

df_metrics = pd.DataFrame({
    "Model": list(accuracy_dict.keys()),
    "Accuracy": list(accuracy_dict.values()),
    "Precision": list(precision_dict.values()),
    "Recall": list(recall_dict.values()),
    "F1 Score": list(f1_dict.values())
})

st.dataframe(df_metrics)
