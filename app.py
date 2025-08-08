import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import calendar
from datetime import datetime

st.set_page_config(layout="wide")
st.title("JetLearn Monthly Payment Predictor")

# Upload file
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])

# Function to parse uploaded file
def load_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# Main logic
if uploaded_file is not None:
    df = load_file(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove any extra spaces in column names

    # Parse dates
    if 'Create Date' in df.columns:
        df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)
    if 'Payment Received Date' in df.columns:
        df['Payment Received Date'] = pd.to_datetime(df['Payment Received Date'], errors='coerce', dayfirst=True)
    else:
        df['Payment Received Date'] = pd.NaT

    # Handle missing Age
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Age'] = df['Age'].astype(int)
    else:
        st.error("Missing 'Age' column.")
        st.stop()

    # Drop rows with missing values in required fields
    df = df.dropna(subset=['Create Date', 'Country', 'JetLearn Deal Source'])

    # Add Create Month/Year
    df['Create_YearMonth'] = df['Create Date'].dt.to_period('M').astype(str)

    # Add Payment Month/Year if available
    df['Payment_YearMonth'] = df['Payment Received Date'].dt.to_period('M').astype(str)

    # Age Grouping
    def age_group(age):
        if age < 8:
            return '<8'
        elif age <= 11:
            return '8-11'
        else:
            return '12+'

    df['AgeGroup'] = df['Age'].apply(age_group)

    # Grouping and Aggregation
    group = df.groupby(['Create_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']) \
        .agg({'Create Date': 'count', 'Payment Received Date': lambda x: x.notna().sum()}) \
        .reset_index() \
        .rename(columns={'Create Date': 'Create Count', 'Payment Received Date': 'Payment Count'})

    # Train model
    group_model = group.copy()
    group_model['Conversion Rate'] = group_model['Payment Count'] / group_model['Create Count']
    group_model = group_model.replace([np.inf, -np.inf], np.nan).dropna()

    X = group_model[['Create Count']]
    y = group_model['Payment Count']

    model = LinearRegression()
    model.fit(X, y)

    # Forecast for this and next month
    today = datetime.today()
    this_month = today.strftime('%Y-%m')
    next_month = (today.replace(day=1) + pd.DateOffset(months=1)).strftime('%Y-%m')

    forecast_df = df.copy()
    forecast_df['Create_YearMonth'] = forecast_df['Create Date'].dt.to_period('M').astype(str)
    forecast_df['AgeGroup'] = forecast_df['Age'].apply(age_group)

    forecast_group = forecast_df.groupby(['Create_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']) \
        .agg({'Create Date': 'count'}).reset_index().rename(columns={'Create Date': 'Create Count'})

    forecast_group['Predicted Payments'] = model.predict(forecast_group[['Create Count']])

    forecast_group['Month'] = forecast_group['Create_YearMonth'].apply(lambda x: 
        "This Month" if x == this_month else ("Next Month" if x == next_month else "Other"))

    # Final output
    this_next = forecast_group[forecast_group['Month'].isin(['This Month', 'Next Month'])] \
        .pivot_table(index=['Country', 'JetLearn Deal Source', 'AgeGroup'],
                     columns='Month',
                     values='Predicted Payments',
                     fill_value=0).reset_index()

    # Display
    st.subheader("📊 Predicted Payments for This Month and Next Month")
    st.dataframe(this_next.style.format({'This Month': '{:.0f}', 'Next Month': '{:.0f}'}))
