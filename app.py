import pandas as pd
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Title
st.title("📊 JetLearn Enrolment Prediction Tool")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("prediction_JL_cleaned.csv")
        df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce')
        df['Payment Received Date '] = pd.to_datetime(df['Payment Received Date '], errors='coerce')
        df = df.dropna(subset=['Create Date', 'Payment Received Date ', 'Country', 'JetLearn Deal Source', 'Age'])
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# Extract year-month columns
df['Create_YearMonth'] = df['Create Date'].dt.to_period('M')
df['Payment_YearMonth'] = df['Payment Received Date '].dt.to_period('M')

# Age grouping (optional)
df['AgeGroup'] = pd.cut(df['Age'], bins=[4, 8, 12, 16, 20], labels=['5-8', '9-12', '13-16', '17-20'])

# Select Month to Predict
predict_month = st.date_input("Select Month to Predict Enrolments", datetime.today()).strftime('%Y-%m')

# Grouping
group = df.groupby(['Create_YearMonth', 'Payment_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup'], observed=False) \
          .size().reset_index(name='Enrollments')

# Filters for the selected prediction month
group['Create_YearMonth'] = group['Create_YearMonth'].astype(str)
group['Payment_YearMonth'] = group['Payment_YearMonth'].astype(str)
predict_df = group[group['Payment_YearMonth'] == predict_month]

# Breakout
same_month = predict_df[predict_df['Create_YearMonth'] == predict_month]['Enrollments'].sum()
prior_months = predict_df[predict_df['Create_YearMonth'] != predict_month]['Enrollments'].sum()
total = same_month + prior_months

# Output
st.subheader(f"📅 Enrolment Prediction for {predict_month}")
st.write(f"✅ **From Deals Created in Same Month:** {same_month}")
st.write(f"📂 **From Deals Created in Prior Months:** {prior_months}")
st.write(f"🔢 **Total Predicted Enrolments:** {total}")
