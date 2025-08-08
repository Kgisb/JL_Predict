import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("JetLearn ML-Based Enrolment Predictor")

# Load training data
try:
    df = pd.read_csv("prediction_JL_cleaned.csv")
except FileNotFoundError:
    st.error("Training file 'prediction_JL_cleaned.csv' not found.")
    st.stop()

# Fix trailing space in column name
df.columns = df.columns.str.strip()

# Derive Enrolled column
if 'Payment Received Date' not in df.columns:
    st.error("Missing 'Payment Received Date' in training data.")
    st.stop()

df['Enrolled'] = df['Payment Received Date'].notna().astype(int)

required_cols = ['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score', 'Enrolled']
if not all(col in df.columns for col in required_cols):
    st.error("Training data missing required columns.")
    st.stop()

# Show class balance
st.subheader("📊 Class Distribution")
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

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X, y)

# Upload for prediction
st.markdown("---")
st.header("📁 Upload Deal File")

uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        deals_df = pd.read_csv(uploaded_file)
    else:
        deals_df = pd.read_excel(uploaded_file)

    # Fix column names
    deals_df.columns = deals_df.columns.str.strip()

    input_cols = ['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score']
    if not all(col in deals_df.columns for col in input_cols):
        st.error("Uploaded file missing required columns.")
    else:
        deals_df['Create Date'] = pd.to_datetime(deals_df['Create Date'], errors='coerce')
        deals_df['Create Month'] = deals_df['Create Date'].dt.month
        deals_df['Create Year'] = deals_df['Create Date'].dt.year

        # Filter unseen values
        missing_country = ~deals_df['Country'].isin(le_country.classes_)
        missing_source = ~deals_df['JetLearn Deal Source'].isin(le_source.classes_)

        if missing_country.any() or missing_source.any():
            st.warning("Some rows removed due to unseen Country or Deal Source.")
            deals_df = deals_df[~missing_country & ~missing_source]

        deals_df['Country_enc'] = le_country.transform(deals_df['Country'])
        deals_df['Source_enc'] = le_source.transform(deals_df['JetLearn Deal Source'])

        X_new = deals_df[features]
        deals_df['Predicted Enrolment'] = model.predict(X_new)

        # Actual Enrolment & Conversion Month logic
        if 'Payment Received Date' in deals_df.columns:
            deals_df['Payment Received Date'] = pd.to_datetime(deals_df['Payment Received Date'], errors='coerce')
            deals_df['Actual Enrolment'] = deals_df['Payment Received Date'].notna().astype(int)

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

            st.subheader("📊 Prediction vs Actual")
            st.write(deals_df[['Predicted Enrolment', 'Actual Enrolment']].value_counts())

        # Show results safely
        display_cols = ['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score',
                        'Predicted Enrolment', 'Converted in Current Month', 'Converted in Next Month']

        display_cols = [col for col in display_cols if col in deals_df.columns]

        if display_cols:
            st.subheader("🔮 Prediction Output")
            st.dataframe(deals_df[display_cols])
        else:
            st.warning("No matching columns found for display.")

        # Download CSV
        csv = deals_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "predicted_enrolments_output.csv", "text/csv")
