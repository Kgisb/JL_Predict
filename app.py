# Prediction
X_new = deals_df[features]
deals_df['Predicted Enrolment'] = model.predict(X_new)

# --- Always add these columns ---
deals_df['Converted in Current Month'] = 0
deals_df['Converted in Next Month'] = 0

# Only compute if Payment Received Date exists
if 'Payment Received Date' in deals_df.columns:
    deals_df['Payment Received Date'] = pd.to_datetime(deals_df['Payment Received Date'], errors='coerce')

    def current_month_conv(row):
        if pd.isna(row['Payment Received Date']) or pd.isna(row['Create Date']):
            return 0
        return int(row['Create Date'].to_period("M") == row['Payment Received Date'].to_period("M"))

    def next_month_conv(row):
        if pd.isna(row['Payment Received Date']) or pd.isna(row['Create Date']):
            return 0
        return int(row['Create Date'].to_period("M") + 1 == row['Payment Received Date'].to_period("M"))

    deals_df['Converted in Current Month'] = deals_df.apply(current_month_conv, axis=1)
    deals_df['Converted in Next Month'] = deals_df.apply(next_month_conv, axis=1)

# --- Display Output ---
display_cols = ['Country', 'Age', 'JetLearn Deal Source', 'Create Date', 'HubSpot Deal Score',
                'Predicted Enrolment', 'Converted in Current Month', 'Converted in Next Month']
display_cols = [col for col in display_cols if col in deals_df.columns]

st.subheader("🔮 Prediction Output")
st.dataframe(deals_df[display_cols])

# --- Download Button ---
csv = deals_df.to_csv(index=False)
st.download_button("📥 Download CSV", csv, "predicted_enrolments_output.csv", "text/csv")
