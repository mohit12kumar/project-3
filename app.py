# ==============================
# MULTIPLE PREDICTION (CSV UPLOAD)
# ==============================
st.header("📂 Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    import pandas as pd

    try:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:")
        st.dataframe(df.head())

        if st.button("Predict for All Rows"):
            model = models_dict[model_choice]

            predictions = model.predict(df)

            df["Prediction"] = predictions
            df["Result"] = df["Prediction"].apply(
                lambda x: "Delayed" if x == 1 else "On Time"
            )

            st.success("✅ Predictions completed")

            st.dataframe(df)

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
