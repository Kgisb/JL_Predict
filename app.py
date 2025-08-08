import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="JetLearn Enrolment Predictor", layout="wide")
st.title("📊 JetLearn Enrolment Predictor")

@st.cache_data
def load_training_data():
    df = pd.read_csv("prediction_JL_cleaned.csv")
    df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)
    df['Payment Received Date '] = pd.to_datetime(df['Payment Received Date '], errors='coerce', dayfirst=True)

    df['Create_YearMonth'] = df['Create Date'].dt.to_period('M')
    df['Payment_YearMonth'] = df['Payment Received Date '].dt.to_period('M')

    df['Enrolled'] = np.where(df['Payment_YearMonth'].notna(), 1, 0)

    df = df[['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score', 'Enrolled']].copy()

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['HubSpot Deal Score'] = df['HubSpot Deal Score'].fillna(0)

    df.dropna(subset=['Country', 'JetLearn Deal Source', 'Create Date'], inplace=True)

    # Encode categorical variables
    le_country = LabelEncoder()
    le_source = LabelEncoder()

    df['Country_enc'] = le_country.fit_transform(df['Country'])
    df['Source_enc'] = le_source.fit_transform(df['JetLearn Deal Source'])

    df['Create_YearMonth'] = df['Create Date'].dt.to_period('M')

    X = df[['Country_enc', 'Age', 'Source_enc', 'HubSpot Deal Score']]
    y = df['Enrolled']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_country, le_source

model, le_country, le_source = load_training_data()

uploaded_file = st.file_uploader("📂 Upload your file (.csv, .xls, .xlsx)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            deals_df = pd.read_csv(uploaded_file)
        else:
            deals_df = pd.read_excel(uploaded_file)

        deals_df['Create Date'] = pd.to_datetime(deals_df['Create Date'], errors='coerce', dayfirst=True)
        if 'Payment Received Date ' in deals_df.columns:
            deals_df['Payment Received Date '] = pd.to_datetime(deals_df['Payment Received Date '], errors='coerce', dayfirst=True)
        else:
            deals_df['Payment Received Date '] = pd.NaT

        deals_df['Age'] = deals_df['Age'].fillna(deals_df['Age'].median())
        deals_df['HubSpot Deal Score'] = deals_df['HubSpot Deal Score'].fillna(0)

        deals_df['Country_enc'] = le_country.transform(deals_df['Country'])
        deals_df['Source_enc'] = le_source.transform(deals_df['JetLearn Deal Source'])

        input_data = deals_df[['Country_enc', 'Age', 'Source_enc', 'HubSpot Deal Score']]
        deals_df['Predicted Enrolment'] = model.predict(input_data)

        deals_df['Create_YearMonth'] = deals_df['Create Date'].dt.to_period('M')
        deals_df['Payment_YearMonth'] = deals_df['Payment Received Date '].dt.to_period('M')

        deals_df['Converted in Current Month'] = np.where(
            (deals_df['Predicted Enrolment'] == 1) & 
            (deals_df['Create_YearMonth'] == deals_df['Payment_YearMonth']), 1, 0)

        deals_df['Converted in Next Month'] = np.where(
            (deals_df['Predicted Enrolment'] == 1) &
            (deals_df['Payment_YearMonth'] == (deals_df['Create_YearMonth'] + 1)), 1, 0)

        st.success("✅ Prediction completed. Here is the output:")

        st.dataframe(deals_df[[
            'Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score',
            'Predicted Enrolment', 'Converted in Current Month', 'Converted in Next Month'
        ]])

        csv = deals_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Result as CSV", data=csv, file_name="prediction_output.csv", mime='text/csv')

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
