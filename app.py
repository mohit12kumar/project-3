# ==============================
# STREAMLIT CONFIG
# ==============================
import streamlit as st
st.set_page_config(page_title="Flight Delay Prediction", layout="centered")

# ==============================
# IMPORTS
# ==============================
import numpy as np
import joblib
import os
import gdown

st.title("✈ Flight Delay Prediction System")

# ==============================
# DOWNLOAD MODELS (ONLY ONCE)
# ==============================
@st.cache_resource
def download_models():
    def download(file_id, output):
        if not os.path.exists(output):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=True)

    download("1kcjKFn-59lK1S8QHv1rHzcm4sL2VLTSU", "random_forest_model.pkl")
    download("1PZdtmAnt15nj1PC1rB8aW0kZIssgU8IM", "decision_tree_model.pkl")
    download("1cL9xaBH6WU_UlXAMpFlU8zenIpY7jgNf", "LogisticRegression.pkl")
    download("1hAMdiSjssNoXRGmzcLcnm8RsivyKStA2", "KNeighborsClassifier.pkl")
    download("1pw_1yVInCY_N5prysDQT7_i78v2LblBU", "LinearSVC.pkl")
    download("1B6gZvXZCizgN9j8C9sBxpeZhLc7UeJI4", "XGBClassifier.pkl")

download_models()

st.success("✅ Models ready")

# ==============================
# SAFE MODEL LOADING
# ==============================
@st.cache_resource
def load_models():
    def safe_load(path):
        try:
            return joblib.load(path)
        except:
            return None

    return {
        "Random Forest": safe_load("random_forest_model.pkl"),
        "Decision Tree": safe_load("decision_tree_model.pkl"),
        "Logistic Regression": safe_load("LogisticRegression.pkl"),
        "KNN": safe_load("KNeighborsClassifier.pkl"),
        "SVM": safe_load("LinearSVC.pkl"),
        "XGBoost": safe_load("XGBClassifier.pkl")
    }

models_dict = load_models()

# Remove failed models
models_dict = {k: v for k, v in models_dict.items() if v is not None}

# ==============================
# MODEL ACCURACY
# ==============================
model_accuracy = {
    "Decision Tree": 93.46,
    "Random Forest": 91.90,
    "SVM": 91.89,
    "Logistic Regression": 91.05,
    "KNN": 92.71,
    "XGBoost": 89.45
}

# ==============================
# INPUT UI
# ==============================
st.header("Enter Flight Details")

col1, col2 = st.columns(2)

with col1:
    airline = st.number_input("Airline", step=1)
    origin = st.number_input("Origin", step=1)
    dest = st.number_input("Destination", step=1)
    dep_delay = st.number_input("Departure Delay")

with col2:
    distance = st.number_input("Distance")
    crs_dep_time = st.number_input("CRS Departure Time")
    month = st.number_input("Month", min_value=1, max_value=12)
    day_of_week = st.number_input("Day of Week", min_value=0, max_value=6)

is_weekend = st.selectbox("Weekend", [0, 1])

# ==============================
# MODEL SELECTION
# ==============================
if len(models_dict) == 0:
    st.error("❌ No models loaded")
else:
    model_choice = st.selectbox("Select Model", list(models_dict.keys()))

    # ==============================
    # PREDICTION
    # ==============================
    if st.button("Predict Delay"):
        try:
            data = np.array([[airline, origin, dest, dep_delay,
                              distance, crs_dep_time, month,
                              day_of_week, is_weekend]])

            model = models_dict[model_choice]
            pred = model.predict(data)[0]

            if pred == 1:
                st.error("⚠ Flight Delayed")
            else:
                st.success("✅ Flight On Time")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ==============================
# PERFORMANCE
# ==============================
st.header("📊 Model Performance")

for model, acc in model_accuracy.items():
    if model in models_dict:
        st.write(f"**{model}** : {acc}%")
