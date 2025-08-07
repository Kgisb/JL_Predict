import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")
st.title("📊 JetLearn Monthly Enrolment Predictor")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("prediction_JL_cleaned.csv")
    df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)
    df['Payment Received Date '] = pd.to_datetime(df['Payment Received Date '], errors='coerce', dayfirst=True)
    df.dropna(subset=['Create Date', 'Payment Received Date ', 'Age', 'Country', 'JetLearn Deal Source'], inplace=True)
    df['Create_YearMonth'] = df['Create Date'].dt.to_period('M').astype(str)
    df['Payment_YearMonth'] = df['Payment Received Date '].dt.to_period('M').astype(str)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 5, 8, 11, 14, 17, 100], labels=['0-5', '6-8', '9-11', '12-14', '15-17', '18+'])
    return df

df = load_data()

# Group and count enrollments
group = df.groupby(['Create_YearMonth', 'Payment_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']) \
          .size().reset_index(name='Enrollments')

# Use Payment Month as target
group['Payment_Month'] = pd.to_datetime(group['Payment_YearMonth']).dt.month

# Encode categorical variables
group_encoded = pd.get_dummies(group[['Country', 'JetLearn Deal Source', 'AgeGroup']], drop_first=True)
features = pd.concat([group_encoded, group[['Payment_Month']]], axis=1)
target = group['Enrollments']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = r2_score(y_test, model.predict(X_test))

# User input
st.sidebar.header("🔧 Prediction Inputs")
country_input = st.sidebar.selectbox("Select Country", df['Country'].unique())
source_input = st.sidebar.selectbox("Select JetLearn Deal Source", df['JetLearn Deal Source'].unique())
age_input = st.sidebar.slider("Age", min_value=4, max_value=18, step=1)
target_month = st.sidebar.selectbox("Target Payment Month", range(1, 13), index=datetime.today().month - 1)

# Determine AgeGroup
if age_input <= 5:
    age_group_input = '0-5'
elif age_input <= 8:
    age_group_input = '6-8'
elif age_input <= 11:
    age_group_input = '9-11'
elif age_input <= 14:
    age_group_input = '12-14'
elif age_input <= 17:
    age_group_input = '15-17'
else:
    age_group_input = '18+'

# Prepare input for prediction
input_df = pd.DataFrame(columns=features.columns)
input_df.loc[0] = 0  # Initialize with zeros
input_df.loc[0, 'Payment_Month'] = target_month

if f"Country_{country_input}" in input_df.columns:
    input_df.loc[0, f"Country_{country_input}"] = 1
if f"JetLearn Deal Source_{source_input}" in input_df.columns:
    input_df.loc[0, f"JetLearn Deal Source_{source_input}"] = 1
if f"AgeGroup_{age_group_input}" in input_df.columns:
    input_df.loc[0, f"AgeGroup_{age_group_input}"] = 1

# Make prediction
predicted_enrollments = model.predict(input_df)[0]

# Breakdown by Create Date Timing
same_month_df = group[(group['Create_YearMonth'] == group['Payment_YearMonth'])]
preceding_months_df = group[(group['Create_YearMonth'] < group['Payment_YearMonth'])]

same_month_total = same_month_df.groupby('Payment_YearMonth')['Enrollments'].sum()
preceding_month_total = preceding_months_df.groupby('Payment_YearMonth')['Enrollments'].sum()

# Results
st.subheader("📈 Predicted Enrollments")
st.write(f"**Predicted Enrollments for Month {target_month}:** `{predicted_enrollments:.0f}` (Accuracy: {accuracy:.2%})")

st.markdown("---")
st.subheader("🔍 Historical Enrollment Breakdown")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("From Same Month's Deals", same_month_total.get(f'2025-{target_month:02d}', 0))
with col2:
    st.metric("From Previous Month's Deals", preceding_month_total.get(f'2025-{target_month:02d}', 0))
with col3:
    st.metric("Total Historical Enrollments", df[df['Payment_YearMonth'] == f'2025-{target_month:02d}'].shape[0])
