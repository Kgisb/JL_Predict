import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import datetime
import os

st.title("JetLearn ML-based Enrolment Predictor")

# File uploader
uploaded_file = st.file_uploader(
    "Upload Excel or CSV File with Labeled Data (must include 'Enrolled' column)", 
    type=["xlsx", "xls", "csv"]
)

# Use uploaded file OR fallback to default file for local testing
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
elif os.path.exists("prediction_JL_cleaned.csv"):
    df = pd.read_csv("prediction_JL_cleaned.csv")
    st.warning("Using default local file: prediction_JL_cleaned.csv (for testing only)")
else:
    st.stop()

# Required columns
required_cols = ['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score', 'Enrolled']
if not all(col in df.columns for col in required_cols):
    st.error(f"Missing one or more required columns: {required_cols}")
else:
    df = df.dropna(subset=required_cols)

    # Date conversion
    df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', format="%Y-%m-%d %H:%M:%S", exact=False)
    df['Create Month'] = df['Create Date'].dt.month
    df['Create Year'] = df['Create Date'].dt.year

    # Encode categorical columns
    le_country = LabelEncoder()
    le_source = LabelEncoder()
    df['Country_enc'] = le_country.fit_transform(df['Country'])
    df['Source_enc'] = le_source.fit_transform(df['JetLearn Deal Source'])

    features = ['Age', 'HubSpot Deal Score', 'Country_enc', 'Source_enc', 'Create Month', 'Create Year']
    target = 'Enrolled'

    X = df[features]
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train ML model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    st.subheader("Model Performance on Test Data")
    st.text(classification_report(y_test, y_pred))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    # Section for August prediction
    st.markdown("---")
    st.header("📈 Predict Enrolments for August 2025")
    aug_file = st.file_uploader("Upload Excel/CSV for August Prediction", type=["xlsx", "xls", "csv"], key="predict_aug")

    if aug_file:
        if aug_file.name.endswith(".csv"):
            aug_df = pd.read_csv(aug_file)
        else:
            aug_df = pd.read_excel(aug_file)

        aug_required_cols = ['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score']
        if not all(col in aug_df.columns for col in aug_required_cols):
            st.error(f"Missing one or more required columns: {aug_required_cols}")
        else:
            aug_df['Create Date'] = pd.to_datetime(aug_df['Create Date'], errors='coerce', format="%Y-%m-%d %H:%M:%S", exact=False)
            aug_df['Create Month'] = aug_df['Create Date'].dt.month
            aug_df['Create Year'] = aug_df['Create Date'].dt.year

            aug_df['Country_enc'] = le_country.transform(aug_df['Country'])
            aug_df['Source_enc'] = le_source.transform(aug_df['JetLearn Deal Source'])

            X_aug = aug_df[features]
            aug_df['Predicted Enrolment'] = model.predict(X_aug)

            # Classify deal month
            aug_start = pd.Timestamp("2025-08-01")
            aug_end = pd.Timestamp("2025-08-31")

            def classify_month(row):
                if aug_start <= row['Create Date'] <= aug_end:
                    return 'M0 (Aug Deals)'
                elif row['Create Date'].month == 7 and row['Create Date'].year == 2025:
                    return 'M-1 (Jul Deals)'
                elif row['Create Date'] < pd.Timestamp("2025-07-01"):
                    return 'M-n (Before Jul)'
                else:
                    return 'Future/Invalid'

            aug_df['Deal Month Category'] = aug_df.apply(classify_month, axis=1)
            aug_df = aug_df[aug_df['Deal Month Category'] != 'Future/Invalid']

            result = aug_df.groupby('Deal Month Category')['Predicted Enrolment'].sum().reset_index()
            total = result['Predicted Enrolment'].sum()
            result.loc[len(result.index)] = ['Total Predicted Enrolments for August', total]

            st.subheader("Predicted Enrolments Breakdown")
            st.dataframe(result)

            csv = result.to_csv(index=False)
            st.download_button("Download Prediction as CSV", csv, "ml_predicted_enrolments_aug.csv", "text/csv")
