import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📊 JetLearn Monthly Enrolment Predictor")

# File uploader
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # Load file depending on type
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Parse dates
    if 'Create Date' in df.columns:
        df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)
    else:
        st.error("❌ 'Create Date' column is missing.")
        st.stop()

    if 'Payment Received Date' not in df.columns:
        df['Payment Received Date'] = pd.NaT
    else:
        df['Payment Received Date'] = pd.to_datetime(df['Payment Received Date'], errors='coerce', dayfirst=True)

    # Fill missing Age
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)
    else:
        st.error("❌ 'Age' column is missing.")
        st.stop()

    # Prepare features for modeling
    df['Create_YearMonth'] = df['Create Date'].dt.to_period('M').astype(str)
    df['Payment_YearMonth'] = df['Payment Received Date'].dt.to_period('M').astype(str)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 8, 12, 16, 99], labels=['Under 9', '9-12', '13-16', '17+'])

    # Group data
    group = df.groupby(['Create_YearMonth', 'Payment_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']) \
        .size().reset_index(name='Deal Count')

    # Feature engineering
    X = group.groupby(['Create_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup'])['Deal Count'].sum().reset_index()
    y = group.groupby(['Payment_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup'])['Deal Count'].sum().reset_index()

    # Rename for merge
    X.rename(columns={'Create_YearMonth': 'YearMonth'}, inplace=True)
    y.rename(columns={'Payment_YearMonth': 'YearMonth', 'Deal Count': 'Payment Count'}, inplace=True)

    # Merge on features
    data = pd.merge(X, y, on=['YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup'], how='left').fillna(0)

    # Encode categorical features
    X_encoded = pd.get_dummies(data[['YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']])
    y_values = data['Payment Count']

    # Train model
    model = LinearRegression()
    model.fit(X_encoded, y_values)

    # Predict for current and next month
    today = pd.Timestamp.today()
    this_month = today.to_period('M').strftime('%Y-%m')
    next_month = (today + pd.DateOffset(months=1)).to_period('M').strftime('%Y-%m')

    predict_df = data.copy()
    predict_df['YearMonth'] = this_month
    this_month_X = pd.get_dummies(predict_df[['YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']])
    this_month_X = this_month_X.reindex(columns=X_encoded.columns, fill_value=0)
    predict_df['This Month Prediction'] = model.predict(this_month_X)

    predict_df['YearMonth'] = next_month
    next_month_X = pd.get_dummies(predict_df[['YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']])
    next_month_X = next_month_X.reindex(columns=X_encoded.columns, fill_value=0)
    predict_df['Next Month Prediction'] = model.predict(next_month_X)

    result = predict_df[['Country', 'JetLearn Deal Source', 'AgeGroup', 'This Month Prediction', 'Next Month Prediction']]
    result_summary = result.groupby(['Country', 'JetLearn Deal Source', 'AgeGroup']).sum().reset_index()

    st.subheader("📈 Predicted Enrollments by Segment")
    st.dataframe(result_summary)

    # Option to download
    st.download_button("Download Predictions as CSV", result_summary.to_csv(index=False), file_name="predicted_enrollments.csv", mime="text/csv")
