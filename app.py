
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import streamlit as st

@st.cache_data
def load_and_prepare():
    df = pd.read_csv("prediction_JL_cleaned.csv")
    df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)
    df['Payment Received Date '] = pd.to_datetime(df['Payment Received Date '], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['Create Date', 'Payment Received Date ', 'Age', 'Country', 'JetLearn Deal Source'])

    df['Create_YearMonth'] = df['Create Date'].dt.to_period('M').astype(str)
    df['Payment_YearMonth'] = df['Payment Received Date '].dt.to_period('M').astype(str)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 8, 12, 16, 20], labels=['0-8', '9-12', '13-16', '17-20'])

    group = df.groupby(['Create_YearMonth', 'Payment_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']) \
              .size().reset_index(name='Conversion_Count')
    return group

def train_model(df_grouped):
    X = pd.get_dummies(df_grouped[['Create_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']])
    y = df_grouped['Conversion_Count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    accuracy = r2_score(y_test, model.predict(X_test))
    return model, X, y, accuracy

def make_prediction(model, df_grouped, target_month):
    df = df_grouped.copy()
    df = df[df['Payment_YearMonth'] == target_month]

    df['Is_M0'] = df['Create_YearMonth'] == target_month
    df_m0 = df[df['Is_M0']]
    df_mprev = df[~df['Is_M0']]

    X_full = pd.get_dummies(df_grouped[['Create_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']])
    X_m0 = pd.get_dummies(df_m0[['Create_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']])
    X_mprev = pd.get_dummies(df_mprev[['Create_YearMonth', 'Country', 'JetLearn Deal Source', 'AgeGroup']])
    X_m0 = X_m0.reindex(columns=X_full.columns, fill_value=0)
    X_mprev = X_mprev.reindex(columns=X_full.columns, fill_value=0)

    pred_m0 = model.predict(X_m0).sum()
    pred_mprev = model.predict(X_mprev).sum()
    return int(round(pred_m0)), int(round(pred_mprev)), int(round(pred_m0 + pred_mprev))

df_grouped = load_and_prepare()
model, X, y, accuracy = train_model(df_grouped)

st.title("📈 Predict Monthly Enrollments (ML-based)")

all_months = sorted(df_grouped['Payment_YearMonth'].unique())
target_month = st.selectbox("Select Month (YYYY-MM):", all_months, index=len(all_months)-1)

if target_month:
    m0, mprev, total = make_prediction(model, df_grouped, target_month)
    st.subheader(f"📊 Prediction for {target_month}")
    st.write(f"🔹 Enrollments from Same Month (M0): **{m0}**")
    st.write(f"🔹 Enrollments from Previous Months (M-1+): **{mprev}**")
    st.write(f"🔹 Total Predicted Enrollments: **{total}**")
    st.write(f"✅ Model R² Accuracy: **{accuracy:.2%}**")
