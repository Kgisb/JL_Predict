import streamlit as st
import pandas as pd
import joblib

st.title("JetLearn: Dual Enrollment Predictor (Same Month & Next Month)")

# Load the trained model
model = joblib.load("multi_month_enrollment_model.pkl")

# File uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file:
    # Read and check required columns
    input_df = pd.read_csv(uploaded_file)
    required_cols = ['Create Date', 'Country', 'Age', 'JetLearn Deal Source', 'HubSpot Deal Score']

    if not all(col in input_df.columns for col in required_cols):
        st.error(f"Missing required columns. Required columns are: {required_cols}")
    else:
        # Preprocess Create Date
        input_df['Create Date'] = pd.to_datetime(input_df['Create Date'], errors='coerce')
        input_df = input_df.dropna(subset=required_cols)

        # Feature engineering
        input_df['Create_Month'] = input_df['Create Date'].dt.to_period('M').astype(str)
        input_df_proc = input_df.drop(columns=['Create Date'])

        # Predict both targets
        predictions = model.predict(input_df_proc)
        pred_df = pd.DataFrame(predictions, columns=[
            'Predicted Enrollment (Same Month)', 'Predicted Enrollment (Next Month)'
        ])
        pred_df = pred_df.replace({1: 'Yes', 0: 'No'})

        # Combine original and predictions
        result_df = pd.concat([input_df, pred_df], axis=1)

        st.subheader("📊 Prediction Results")
        st.dataframe(result_df)

        # Downloadable CSV
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", data=csv, file_name="dual_enrollment_predictions.csv", mime='text/csv')
