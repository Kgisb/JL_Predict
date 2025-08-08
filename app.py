import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

st.set_page_config(page_title="JetLearn Enrolment Predictor", layout="wide")

# Function to standardize 'Payment Received Date'
def standardize_payment_column(df):
    for col in df.columns:
        if col.strip().lower() == 'payment received date':
            df.rename(columns={col: 'Payment Received Date'}, inplace=True)
            break
    if 'Payment Received Date' not in df.columns:
        df['Payment Received Date'] = pd.NaT
    df['Payment Received Date'] = pd.to_datetime(df['Payment Received Date'], errors='coerce', dayfirst=True)
    return df

@st.cache_data
def load_training_data():
    df = pd.read_csv("prediction_JL_cleaned.csv")
    df = standardize_payment_column(df)
    df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)
    df['Create_YearMonth'] = df['Create Date'].dt.to_period('M')
    df['Payment_YearMonth'] = df['Payment Received Date'].dt.to_period('M')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 10, 14, 18, 25, 100], labels=['0-10','11-14','15-18','19-25','25+'])

    # Group by to find actual enrolments
    group = df.groupby(['Create_YearMonth', 'Payment_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']) \
              .size().reset_index(name='Enrollments')

    return df, group

df, train_grouped = load_training_data()

# Prepare label encoders
le_country = LabelEncoder()
le_source = LabelEncoder()
le_agegroup = LabelEncoder()

train_grouped['Country_enc'] = le_country.fit_transform(train_grouped['Country'])
train_grouped['Source_enc'] = le_source.fit_transform(train_grouped['JetLearn Deal Source'])
train_grouped['AgeGroup_enc'] = le_agegroup.fit_transform(train_grouped['AgeGroup'].astype(str))

X_train = train_grouped[['Country_enc', 'Source_enc', 'AgeGroup_enc']]
y_train = train_grouped['Enrollments']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# File upload section
st.title("📊 JetLearn Monthly Enrolment Predictor")
uploaded_file = st.file_uploader("Upload data file (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        ext = uploaded_file.name.split('.')[-1]
        if ext == "csv":
            deals_df = pd.read_csv(uploaded_file)
        else:
            deals_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    deals_df = standardize_payment_column(deals_df)
    deals_df['Create Date'] = pd.to_datetime(deals_df['Create Date'], errors='coerce', dayfirst=True)
    deals_df['Payment Received Date'] = pd.to_datetime(deals_df['Payment Received Date'], errors='coerce', dayfirst=True)

    deals_df['Create_YearMonth'] = deals_df['Create Date'].dt.to_period('M')
    deals_df['Payment_YearMonth'] = deals_df['Payment Received Date'].dt.to_period('M')
    deals_df['Age'] = deals_df['Age'].fillna(deals_df['Age'].median())

    deals_df['AgeGroup'] = pd.cut(deals_df['Age'], bins=[0, 10, 14, 18, 25, 100], labels=['0-10','11-14','15-18','19-25','25+'])

    # Fill missing and handle unseen labels
    deals_df['Country'] = deals_df['Country'].fillna("Unknown")
    deals_df['JetLearn Deal Source'] = deals_df['JetLearn Deal Source'].fillna("Unknown")

    for col, encoder in [('Country', le_country), ('JetLearn Deal Source', le_source)]:
        new_labels = set(deals_df[col].unique()) - set(encoder.classes_)
        if new_labels:
            encoder.classes_ = np.concatenate([encoder.classes_, list(new_labels)])

    deals_df['Country_enc'] = le_country.transform(deals_df['Country'])
    deals_df['Source_enc'] = le_source.transform(deals_df['JetLearn Deal Source'])
    deals_df['AgeGroup_enc'] = le_agegroup.transform(deals_df['AgeGroup'].astype(str))

    X_test = deals_df[['Country_enc', 'Source_enc', 'AgeGroup_enc']]
    deals_df['Predicted Enrolment'] = model.predict(X_test).round().astype(int)

    # Compute Converted in Current and Next Month
    deals_df['Converted in Current Month'] = (
        (deals_df['Create_YearMonth'] == deals_df['Payment_YearMonth'])
    ).astype(int)

    deals_df['Converted in Next Month'] = (
        (deals_df['Payment_YearMonth'] == (deals_df['Create_YearMonth'] + 1))
    ).astype(int)

    st.success("✅ Prediction Completed!")
    st.dataframe(deals_df[['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score',
                           'Predicted Enrolment', 'Converted in Current Month', 'Converted in Next Month']])

    csv = deals_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", data=csv, file_name="predicted_enrolments.csv", mime='text/csv')

else:
    st.info("👆 Upload your file to begin prediction.")
