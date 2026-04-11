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
from tensorflow.keras.models import load_model

st.title("✈ Flight Delay Prediction System")

# ==============================
# GOOGLE DRIVE DOWNLOAD FUNCTION
# ==============================
def download_model(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# ==============================
# DOWNLOAD MODELS (YOUR LINKS)
# ==============================
download_model("1kcjKFn-59lK1S8QHv1rHzcm4sL2VLTSU", "random_forest_model.pkl")
download_model("1PZdtmAnt15nj1PC1rB8aW0kZIssgU8IM", "decision_tree_model.pkl")
download_model("1cL9xaBH6WU_UlXAMpFlU8zenIpY7jgNf", "LogisticRegression.pkl")
download_model("1hAMdiSjssNoXRGmzcLcnm8RsivyKStA2", "KNeighborsClassifier.pkl")
download_model("1pw_1yVInCY_N5prysDQT7_i78v2LblBU", "LinearSVC.pkl")
download_model("1B6gZvXZCizgN9j8C9sBxpeZhLc7UeJI4", "XGBClassifier.pkl")
download_model("1hUTkMkXS-_dQwQ4dHja0Sn4N8rU-Vbqi", "deep_learning_model.keras")

# ==============================
# LOAD MODELS
# ==============================
rf_model = joblib.load("random_forest_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
lr_model = joblib.load("LogisticRegression.pkl")
knn_model = joblib.load("KNeighborsClassifier.pkl")
svm_model = joblib.load("LinearSVC.pkl")
xgb_model = joblib.load("XGBClassifier.pkl")
dl_model = load_model("deep_learning_model.keras", compile=False)

# ==============================
# MODEL ACCURACY
# ==============================
model_accuracy = {
    "Deep Learning": 93.46,
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
model_choice = st.selectbox("Select Model", list(model_accuracy.keys()))

def get_model(name):
    return {
        "Random Forest": rf_model,
        "Decision Tree": dt_model,
        "Logistic Regression": lr_model,
        "KNN": knn_model,
        "SVM": svm_model,
        "XGBoost": xgb_model,
        "Deep Learning": dl_model
    }.get(name, None)

# ==============================
# PREDICT DELAY
# ==============================
if st.button("Predict Delay"):
    try:
        data = np.array([[airline, origin, dest, dep_delay,
                          distance, crs_dep_time, month,
                          day_of_week, is_weekend]])

        model = get_model(model_choice)

        if model is None:
            st.error("❌ Model not loaded")
        else:
            if model_choice == "Deep Learning":
                pred = (model.predict(data) > 0.5).astype(int)[0][0]
            else:
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
