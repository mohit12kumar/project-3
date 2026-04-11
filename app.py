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
# RELOAD BUTTON (IMPORTANT)
# ==============================
if st.button("🔄 Reload Models"):
    st.cache_resource.clear()
    st.success("✅ Cache cleared. Reload app.")

# ==============================
# MODEL LOADER (FINAL FIXED)
# ==============================
@st.cache_resource
def load_model(mode, model_name):

    def download(file_id, output):
        # 🔥 DELETE OLD FILE
        if os.path.exists(output):
            os.remove(output)

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

    # ==============================
    # DELAY MODELS
    # ==============================
    delay_links = {
        "Random Forest": ("1kcjKFn-59lK1S8QHv1rHzcm4sL2VLTSU", "random_forest_model.pkl"),
        "Decision Tree": ("1PZdtmAnt15nj1PC1rB8aW0kZIssgU8IM", "decision_tree_model.pkl"),
        "Logistic Regression": ("1cL9xaBH6WU_UlXAMpFlU8zenIpY7jgNf", "LogisticRegression.pkl"),
        "KNN": ("1hAMdiSjssNoXRGmzcLcnm8RsivyKStA2", "KNeighborsClassifier.pkl"),
        "SVM": ("1pw_1yVInCY_N5prysDQT7_i78v2LblBU", "LinearSVC.pkl"),
        "XGBoost": ("YOUR_NEW_FILE_ID", "XGBClassifier.pkl")  # 🔥 UPDATE HERE
    }

    # ==============================
    # CANCELLATION MODELS
    # ==============================
    cancel_links = {
        "Random Forest": ("1AJxhnPsOL5VRtXqB8TO52RFAAzKQa_dI", "rf_cancel.pkl"),
        "Decision Tree": ("1VGat3BhFmQwkjrQKDUDPWW12_FHndjVv", "dt_cancel.pkl"),
        "Logistic Regression": ("16k7XQcInCTNuveWDSPiTLbhfgRPH2bUi", "lr_cancel.pkl"),
        "KNN": ("1qnC3xUyeJ8SDi455THh2_IbSmc4BVQgi", "knn_cancel.pkl"),
        "SVM": ("1ppy1emNTbhbi0YP0CxWu-cAJXXhNorNV", "svm_cancel.pkl"),
        "XGBoost": ("YOUR_NEW_FILE_ID", "XGBClassifier.pkl")  # 🔥 UPDATE HERE
    }

    links = delay_links if mode == "Delay" else cancel_links

    try:
        file_id, filename = links[model_name]
    except:
        st.error("❌ Model not found")
        return None

    # Download fresh model
    download(file_id, filename)

    # Load safely
    try:
        model = joblib.load(filename)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None


# ==============================
# MODE + MODEL SELECTION
# ==============================
mode = st.selectbox("Select Prediction Type", ["Delay", "Cancellation"])

model_list = [
    "Random Forest",
    "Decision Tree",
    "Logistic Regression",
    "KNN",
    "SVM",
    "XGBoost"
]

model_choice = st.selectbox("Select Model", model_list)

st.success(f"✅ Using Model: {model_choice} ({mode})")
st.info("📥 Model will download only once (cached)")

model = load_model(mode, model_choice)

# Debug (optional)
if model is not None:
    st.write("Model type:", type(model))

# ==============================
# INPUT UI
# ==============================
st.header("🧍 Single Prediction")

col1, col2 = st.columns(2)

with col1:
    airline = st.number_input("Airline", step=1)
    origin = st.number_input("Origin", step=1)
    dest = st.number_input("Destination", step=1)
    dep_delay = st.number_input("Departure Delay")

with col2:
    distance = st.number_input("Distance")
    crs_dep_time = st.number_input("CRS Dep Time")
    month = st.number_input("Month", 1, 12)
    day_of_week = st.number_input("Day of Week", 0, 6)

weekend = st.selectbox("Weekend", [0, 1])

# ==============================
# SINGLE PREDICTION
# ==============================
if st.button("Predict"):

    if model is None:
        st.error("❌ Model not loaded properly")
    else:
        try:
            data = np.array([[airline, origin, dest, dep_delay,
                              distance, crs_dep_time, month,
                              day_of_week, weekend]])

            pred = model.predict(data)[0]

            if mode == "Delay":
                st.success("⚠ Flight Delayed" if pred == 1 else "✅ Flight On Time")
            else:
                st.success("⚠ Flight Cancelled" if pred == 1 else "✅ Not Cancelled")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ==============================
# BATCH PREDICTION (FIXED)
# ==============================
st.header("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.write("Preview:")
        st.dataframe(df.head())

        if st.button("Predict Batch"):

            # 🔥 FIX COLUMN NAMES
            df.columns = [col.strip().lower() for col in df.columns]

            df = df.rename(columns={
                "weekend": "is_weekend"
            })

            required = [
                "airline", "origin", "dest", "dep_delay",
                "distance", "crs_dep_time", "month",
                "day_of_week", "is_weekend"
            ]

            # Auto-create weekend if missing
            if "is_weekend" not in df.columns:
                df["is_weekend"] = df["day_of_week"].apply(
                    lambda x: 1 if x in [5, 6] else 0
                )

            df_model = df[required]

            preds = model.predict(df_model)

            if mode == "Delay":
                df["RESULT"] = ["Delayed" if x == 1 else "On Time" for x in preds]
            else:
                df["RESULT"] = ["Cancelled" if x == 1 else "Not Cancelled" for x in preds]

            st.success("✅ Batch Prediction Done")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download Results", csv, "results.csv")

    except Exception as e:
        st.error(f"CSV Error: {e}")
