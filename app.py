import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Load and clean data ---
@st.cache_data
def load_data():
    df = pd.read_csv("prediction_JL_cleaned.csv")
    df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)
    df['Payment Received Date '] = pd.to_datetime(df['Payment Received Date '], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['Create Date', 'Payment Received Date ', 'Country', 'JetLearn Deal Source', 'Age'])

    df['Create_YearMonth'] = df['Create Date'].dt.to_period('M')
    df['Payment_YearMonth'] = df['Payment Received Date '].dt.to_period('M')
    df['Same_Month_Conversion'] = (df['Create_YearMonth'] == df['Payment_YearMonth']).astype(int)
    return df

df = load_data()

# --- User input ---
st.title("📊 JetLearn Monthly Enrolment Predictor")

target_month = st.selectbox("Select Prediction Month (format: YYYY-MM)", sorted(df['Payment_YearMonth'].unique().astype(str)))
target_month = pd.Period(target_month, freq='M')

# Filter for all deals where Payment is in target month
df_target = df[df['Payment_YearMonth'] == target_month]

# Encode categorical variables
df_model = df.copy()
df_model['Same_Month_Conversion'] = (df_model['Create_YearMonth'] == df_model['Payment_YearMonth']).astype(int)
df_model = df_model[df_model['Payment_YearMonth'].notna()]

X = pd.get_dummies(df_model[['Country', 'JetLearn Deal Source', 'Age', 'Create_YearMonth']], drop_first=True)
y = df_model['Same_Month_Conversion']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on filtered data
df_pred = df_target.copy()
df_pred_input = pd.get_dummies(df_pred[['Country', 'JetLearn Deal Source', 'Age', 'Create_YearMonth']], drop_first=True)
df_pred_input = df_pred_input.reindex(columns=X.columns, fill_value=0)
df_pred['Predicted_SameMonth'] = model.predict(df_pred_input)

# Results
same_month_count = df_pred['Predicted_SameMonth'].sum()
prev_month_count = len(df_pred) - same_month_count
total_count = len(df_pred)

# Accuracy
pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, pred_test)

# --- Output ---
st.header("🔮 Prediction Summary")
st.write(f"📅 Prediction for: **{target_month.strftime('%B %Y')}**")
st.write(f"✅ **Predicted Enrollments from Same-Month Deals:** {int(same_month_count)}")
st.write(f"🔁 **Predicted Enrollments from Previous-Month Deals:** {int(prev_month_count)}")
st.write(f"📊 **Total Predicted Enrollments:** {int(total_count)}")
st.write(f"🎯 **Model Accuracy:** {accuracy:.2%}")
