# ==============================
# STREAMLIT CONFIG (MUST BE FIRST)
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
# GOOGLE DRIVE DOWNLOAD FUNCTION
# ==============================
def download_model(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# ==============================
# DOWNLOAD MODELS
# ==============================
download_model("1kcjKFn-59lK1S8QHv1rHzcm4sL2VLTSU", "random_forest_model.pkl")
download_model("1PZdtmAnt15nj1PC1rB8aW0kZIssgU8IM", "decision_tree_model.pkl")
download_model("1cL9xaBH6WU_UlXAMpFlU8zenIpY7jgNf", "LogisticRegression.pkl")
download_model("1hAMdiSjssNoXRGmzcLcnm8RsivyKStA2", "KNeighborsClassifier.pkl")
download_model("1pw_1yVInCY_N5prysDQT7_i78v2LblBU", "LinearSVC.pkl")
download_model("1B6gZvXZCizgN9j8C9sBxpeZhLc7UeJI4", "XGBClassifier.pkl")

st.write("📥 Loading models...")

# ==============================
# SAFE MODEL LOADING
# ==============================
def load_safe(path, name):
    try:
        return joblib.load(path)
    except Exception:
        st.warning(f"⚠ {name} not loaded (skipped)")
        return None

rf_model = load_safe("random_forest_model.pkl", "Random Forest")
dt_model = load_safe("decision_tree_model.pkl", "Decision Tree")
lr_model = load_safe("LogisticRegression.pkl", "Logistic Regression")
knn_model = load_safe("KNeighborsClassifier.pkl", "KNN")
svm_model = load_safe("LinearSVC.pkl", "SVM")
xgb_model = load_safe("XGBClassifier.pkl", "XGBoost")

# ==============================
# FILTER WORKING MODELS
# ==============================
models_dict = {
    "Random Forest": rf_model,
    "Decision Tree": dt_model,
    "Logistic Regression": lr_model,
    "KNN": knn_model,
    "SVM": svm_model,
    "XGBoost": xgb_model
}

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
# INPUT FIELDS
# ==============================
st.header("Enter Flight Details")

airline = st.number_input("Airline", step=1)
origin = st.number_input("Origin", step=1)
dest = st.number_input("Destination", step=1)
dep_delay = st.number_input("Departure Delay")
distance = st.number_input("Distance")
crs_dep_time = st.number_input("CRS Departure Time")
month = st.number_input("Month", min_value=1, max_value=12)
day_of_week = st.number_input("Day of Week", min_value=0, max_value=6)
is_weekend = st.selectbox("Weekend", [0, 1])

# ==============================
# MODEL SELECTION
# ==============================
if len(models_dict) == 0:
    st.error("❌ No models loaded properly")
else:
    model_choice = st.selectbox("Select Model", list(models_dict.keys()))

    # ==============================
    # PREDICT DELAY
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
            st.error(f"Error: {e}")

# ==============================
# MODEL PERFORMANCE
# ==============================
st.header("📊 Model Performance")

for model, acc in model_accuracy.items():
    st.write(f"**{model}** : {acc}%")
