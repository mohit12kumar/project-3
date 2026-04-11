# ==============================
# BATCH PREDICTION (CSV)
# ==============================
st.header("📂 Batch Prediction (Upload CSV)")

prediction_type = st.selectbox(
    "Select Prediction Type",
    ["Delay Prediction", "Cancellation Prediction"]
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.write("📄 Preview:")
        st.dataframe(df.head())

        if st.button("Predict for All Rows"):

            required_columns = [
                "Airline", "Origin", "Destination", "Departure Delay",
                "Distance", "CRS Dep Time", "Month", "Day of Week", "Weekend"
            ]

            df = df[required_columns]

            # ==============================
            # DELAY PREDICTION
            # ==============================
            if prediction_type == "Delay Prediction":
                model = models_dict[model_choice]
                predictions = model.predict(df)

                df["Prediction"] = predictions
                df["Result"] = df["Prediction"].apply(
                    lambda x: "Delayed" if x == 1 else "On Time"
                )

            # ==============================
            # CANCELLATION PREDICTION
            # ==============================
            else:
                # Simple rule-based logic
                df["Prediction"] = df["Departure Delay"].apply(
                    lambda x: 1 if x > 60 else 0
                )

                df["Result"] = df["Prediction"].apply(
                    lambda x: "Cancelled" if x == 1 else "Not Cancelled"
                )

            st.success("✅ Batch Prediction Done")
            st.dataframe(df)

            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results",
                csv,
                "prediction_results.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"CSV Error: {e}")
