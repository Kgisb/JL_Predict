import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(layout="wide")
st.title("JetLearn | Monthly Enrollment Predictor")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xls", "xlsx"])

def load_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

if uploaded_file is not None:
    df = load_file(uploaded_file)

    # Clean column names (remove trailing spaces)
    df.columns = df.columns.str.strip()

    # Handle date parsing
    if 'Create Date' in df.columns:
        df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)
    else:
        st.error("Missing 'Create Date' column.")
        st.stop()

    if 'Payment Received Date' in df.columns:
        df['Payment Received Date'] = pd.to_datetime(df['Payment Received Date'], errors='coerce', dayfirst=True)
    else:
        df['Payment Received Date'] = pd.NaT  # Fill with missing if column not present

    # Clean and handle Age
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Age'] = df['Age'].astype(int)
    else:
        st.error("Missing 'Age' column.")
        st.stop()

    # Drop rows missing key columns
    df.dropna(subset=['Create Date', 'Country', 'JetLearn Deal Source'], inplace=True)

    # Feature engineering
    df['Create_YearMonth'] = df['Create Date'].dt.to_period('M').astype(str)
    df['Payment_YearMonth'] = df['Payment Received Date'].dt.to_period('M').astype(str)

    # Age Group bucketing
    def age_bucket(age):
        if age < 8:
            return '<8'
        elif age <= 11:
            return '8-11'
        else:
            return '12+'
    df['AgeGroup'] = df['Age'].apply(age_bucket)

    # Group for modeling
    group = df.groupby(['Create_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']) \
              .agg(Create_Count=('Create Date', 'count'),
                   Payment_Count=('Payment Received Date', lambda x: x.notna().sum())) \
              .reset_index()

    # Prepare for model
    model_data = group.copy()
    model_data = model_data[model_data['Create_Count'] > 0]
    model_data['Conversion Rate'] = model_data['Payment_Count'] / model_data['Create_Count']
    model_data.dropna(subset=['Conversion Rate'], inplace=True)

    X = model_data[['Create_Count']]
    y = model_data['Payment_Count']
    model = LinearRegression()
    model.fit(X, y)

    # Predict for this and next month
    today = datetime.today()
    this_month_str = today.strftime('%Y-%m')
    next_month_str = (today.replace(day=1) + pd.DateOffset(months=1)).strftime('%Y-%m')

    future_df = df.copy()
    future_df['Create_YearMonth'] = future_df['Create Date'].dt.to_period('M').astype(str)
    future_df['AgeGroup'] = future_df['Age'].apply(age_bucket)

    forecast_group = future_df.groupby(['Create_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']) \
                              .agg(Create_Count=('Create Date', 'count')) \
                              .reset_index()

    forecast_group['Predicted_Payments'] = model.predict(forecast_group[['Create_Count']])

    forecast_group['Month'] = forecast_group['Create_YearMonth'].apply(lambda x:
        "This Month" if x == this_month_str else ("Next Month" if x == next_month_str else "Other"))

    output = forecast_group[forecast_group['Month'].isin(['This Month', 'Next Month'])] \
        .pivot_table(index=['Country', 'JetLearn Deal Source', 'AgeGroup'],
                     columns='Month',
                     values='Predicted_Payments',
                     fill_value=0).reset_index()

    st.subheader("📊 Predicted Payments: This Month vs Next Month")
    st.dataframe(output.style.format({'This Month': '{:.0f}', 'Next Month': '{:.0f}'}))
