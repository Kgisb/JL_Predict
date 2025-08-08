import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="JetLearn Enrolment Predictor", layout="wide")
st.title("🎯 JetLearn Enrolment Predictor")

# --- Load and Prepare Training Data ---
@st.cache_data
def load_training_data():
    df = pd.read_csv("prediction_JL_cleaned.csv")
    df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)

    if 'Payment Received Date' in df.columns:
        df['Payment Received Date'] = pd.to_datetime(df['Payment Received Date'], errors='coerce', dayfirst=True)
        df['Enrolled'] = df['Payment Received Date'].notna().astype(int)
    else:
        df['Enrolled'] = 0  # fallback if missing

    return df.dropna(subset=['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score'])

df = load_training_data()

# --- Encode & Prepare Features ---
df['Age'] = df['Age'].astype(int)
df['HubSpot Deal Score'] = pd.to_numeric(df['HubSpot Deal Score'], errors='coerce').fillna(0).astype(int)

le_country = LabelEncoder()
le_source = LabelEncoder()
df['Country_enc'] = le_country.fit_transform(df['Country'])
df['Source_enc'] = le_source.fit_transform(df['JetLearn Deal Source'])

X = df[['Country_enc', 'Age', 'Source_enc', 'HubSpot Deal Score']]
y = df['Enrolled']

# --- Train ML Model ---
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --- Upload File ---
st.header("📤 Upload File for Prediction")
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded_file:
    file_name = uploaded_file.name.lower()

    # Read file based on extension
    if file_name.endswith('.csv'):
        deals_df = pd.read_csv(uploaded_file)
    else:
        deals_df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # --- Preprocess uploaded data ---
    deals_df['Create Date'] = pd.to_datetime(deals_df['Create Date'], errors='coerce', dayfirst=True)
    deals_df['Age'] = deals_df['Age'].astype(int)
    deals_df['HubSpot Deal Score'] = pd.to_numeric(deals_df['HubSpot Deal Score'], errors='coerce').fillna(0).astype(int)

    # Encode using training mappings
    deals_df['Country_enc'] = le_country.transform(
        deals_df['Country'].where(deals_df['Country'].isin(le_country.classes_), le_country.classes_[0])
    )
    deals_df['Source_enc'] = le_source.transform(
        deals_df['JetLearn Deal Source'].where(deals_df['JetLearn Deal Source'].isin(le_source.classes_), le_source.classes_[0])
    )

    # --- Predict Enrolments ---
    features = ['Country_enc', 'Age', 'Source_enc', 'HubSpot Deal Score']
    deals_df['Predicted Enrolment'] = model.predict(deals_df[features])

    # --- Add conversion logic ---
    deals_df['Converted in Current Month'] = 0
    deals_df['Converted in Next Month'] = 0

    if 'Payment Received Date' in deals_df.columns:
        deals_df['Payment Received Date'] = pd.to_datetime(deals_df['Payment Received Date'], errors='coerce')

        def current_month_conv(row):
            if pd.isna(row['Payment Received Date']) or pd.isna(row['Create Date']):
                return 0
            return int(row['Create Date'].to_period("M") == row['Payment Received Date'].to_period("M"))

        def next_month_conv(row):
            if pd.isna(row['Payment Received Date']) or pd.isna(row['Create Date']):
                return 0
            return int(row['Create Date'].to_period("M") + 1 == row['Payment Received Date'].to_period("M"))

        deals_df['Converted in Current Month'] = deals_df.apply(current_month_conv, axis=1)
        deals_df['Converted in Next Month'] = deals_df.apply(next_month_conv, axis=1)

    # --- Display Results ---
    st.subheader("🔮 Prediction Output")
    display_cols = ['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score',
                    'Predicted Enrolment', 'Converted in Current Month', 'Converted in Next Month']
    display_cols = [col for col in display_cols if col in deals_df.columns]

    st.dataframe(deals_df[display_cols])

    # --- Download Output ---
    csv = deals_df.to_csv(index=False)
    st.download_button("📥 Download Results as CSV", csv, "predicted_enrolments_output.csv", "text/csv")
