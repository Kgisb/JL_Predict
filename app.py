import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

st.set_page_config(page_title="JetLearn Enrolment Predictor", layout="wide")

@st.cache_data
def load_training_data():
    df = pd.read_csv("prediction_JL_cleaned.csv")
    df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)
    df['Payment Received Date'] = pd.to_datetime(df['Payment Received Date'], errors='coerce', dayfirst=True)
    df['Create_YearMonth'] = df['Create Date'].dt.to_period('M')
    df['Payment_YearMonth'] = df['Payment Received Date'].dt.to_period('M')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)
    df['HubSpot Deal Score'] = pd.to_numeric(df['HubSpot Deal Score'], errors='coerce').fillna(0)
    df['Enrolled'] = np.where(df['Payment Received Date'].notnull(), 1, 0)
    return df

def preprocess_input(df, le_country, le_source):
    df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)
    df['Payment Received Date'] = pd.to_datetime(df['Payment Received Date'], errors='coerce', dayfirst=True)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)
    df['HubSpot Deal Score'] = pd.to_numeric(df['HubSpot Deal Score'], errors='coerce').fillna(0)

    df['Country_enc'] = le_country.transform(df['Country'])
    df['Source_enc'] = le_source.transform(df['JetLearn Deal Source'])

    return df

# Load and prepare training data
df = load_training_data()

# Label Encoding
le_country = LabelEncoder()
le_source = LabelEncoder()
df['Country_enc'] = le_country.fit_transform(df['Country'])
df['Source_enc'] = le_source.fit_transform(df['JetLearn Deal Source'])

# ML Model Training
X = df[['Country_enc', 'Age', 'Source_enc', 'HubSpot Deal Score']]
y = df['Enrolled']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# App UI
st.title("📊 JetLearn Monthly Enrolment Predictor")

uploaded_file = st.file_uploader("Upload your prediction file (CSV or Excel)", type=['csv', 'xls', 'xlsx'])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]

    if file_type == 'csv':
        deals_df = pd.read_csv(uploaded_file)
    else:
        deals_df = pd.read_excel(uploaded_file)

    try:
        deals_df = preprocess_input(deals_df, le_country, le_source)

        # Predict enrolments
        X_pred = deals_df[['Country_enc', 'Age', 'Source_enc', 'HubSpot Deal Score']]
        deals_df['Predicted Enrolment'] = model.predict(X_pred)

        # Month markers
        if 'Create Date' in deals_df.columns:
            deals_df['Create_YearMonth'] = deals_df['Create Date'].dt.to_period('M')
        else:
            deals_df['Create_YearMonth'] = pd.NaT

        if 'Payment Received Date' in deals_df.columns:
            deals_df['Payment_YearMonth'] = deals_df['Payment Received Date'].dt.to_period('M')
        else:
            deals_df['Payment_YearMonth'] = pd.NaT

        # Detect current month from earliest Create Date
        if deals_df['Create_YearMonth'].notnull().any():
            current_month = deals_df['Create_YearMonth'].dropna().mode()[0]
            next_month = (current_month.to_timestamp() + pd.DateOffset(months=1)).to_period('M')

            deals_df['Converted in Current Month'] = np.where(
                deals_df['Payment_YearMonth'] == current_month, 1, 0
            )
            deals_df['Converted in Next Month'] = np.where(
                deals_df['Payment_YearMonth'] == next_month, 1, 0
            )
        else:
            deals_df['Converted in Current Month'] = np.nan
            deals_df['Converted in Next Month'] = np.nan

        st.success("✅ Prediction completed!")
        st.dataframe(deals_df[['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score',
                               'Predicted Enrolment', 'Converted in Current Month', 'Converted in Next Month']])

        # Download button
        st.download_button(
            label="📥 Download Prediction Results",
            data=deals_df.to_csv(index=False).encode('utf-8'),
            file_name="predicted_enrolments.csv",
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

