import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import datetime

st.title("JetLearn ML-Based Enrolment Predictor")

# Load historical training data
try:
    df = pd.read_csv("prediction_JL_cleaned.csv")
except FileNotFoundError:
    st.error("Training file 'prediction_JL_cleaned.csv' not found.")
    st.stop()

# Derive Enrolled column from Payment Received Date
if 'Payment Received Date ' not in df.columns:
    st.error("Column 'Payment Received Date ' not found in training data.")
    st.stop()

df['Enrolled'] = df['Payment Received Date '].notna().astype(int)

# Check required columns
required_cols = ['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score', 'Enrolled']
if not all(col in df.columns for col in required_cols):
    st.error("Missing required columns in training data.")
    st.stop()

# Show class distribution
st.subheader("📊 Class Distribution in Training Data")
st.write(df['Enrolled'].value_counts().rename({0: "Not Enrolled (0)", 1: "Enrolled (1)"}))

# Preprocessing
df = df.dropna(subset=required_cols)
df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce')
df['Create Month'] = df['Create Date'].dt.month
df['Create Year'] = df['Create Date'].dt.year

le_country = LabelEncoder()
le_source = LabelEncoder()
df['Country_enc'] = le_country.fit_transform(df['Country'])
df['Source_enc'] = le_source.fit_transform(df['JetLearn Deal Source'])

features = ['Age', 'HubSpot Deal Score', 'Country_enc', 'Source_enc', 'Create Month', 'Create Year']
target = 'Enrolled'

X = df[features]
y = df[target]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X, y)

# Upload new file for prediction
st.markdown("---")
st.header("📁 Upload Deal File to Predict Enrolment")

uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        deals_df = pd.read_csv(uploaded_file)
    else:
        deals_df = pd.read_excel(uploaded_file)

    input_cols = ['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score']
    if not all(col in deals_df.columns for col in input_cols):
        st.error("Missing required columns in uploaded file.")
    else:
        # Preprocess
        deals_df['Create Date'] = pd.to_datetime(deals_df['Create Date'], errors='coerce')
        deals_df['Create Month'] = deals_df['Create Date'].dt.month
        deals_df['Create Year'] = deals_df['Create Date'].dt.year

        # Drop unseen categories
        missing_country = ~deals_df['Country'].isin(le_country.classes_)
        missing_source = ~deals_df['JetLearn Deal Source'].isin(le_source.classes_)

        if missing_country.any() or missing_source.any():
            st.warning("Some rows had unseen countries or sources and were removed.")
            deals_df = deals_df[~missing_country & ~missing_source]

        deals_df['Country_enc'] = le_country.transform(deals_df['Country'])
        deals_df['Source_enc'] = le_source.transform(deals_df['JetLearn Deal Source'])

        X_new = deals_df[features]
        deals_df['Predicted Enrolment'] = model.predict(X_new)

        # Optional: check actual enrolment if Payment Received Date is present
        if 'Payment Received Date ' in deals_df.columns:
            deals_df['Actual Enrolment'] = deals_df['Payment Received Date '].notna().astype(int)
            st.subheader("📊 Prediction vs Actual")
            st.write(deals_df[['Predicted Enrolment', 'Actual Enrolment']].value_counts())

        # Show and download
        st.subheader("🔮 Prediction Summary")
        st.dataframe(deals_df[['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score', 'Predicted Enrolment']])

        csv = deals_df.to_csv(index=False)
        st.download_button("Download Predictions CSV", csv, "predicted_enrolments.csv", "text/csv")
