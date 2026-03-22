import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="wide")
st.title("🛡️ FraudShield - My Personal Anti-Fraud Scanner")
st.markdown("**Built by me** — Upload your bank/credit card CSV and get instant fraud flags. Everything runs privately in your browser session.")

# Sidebar - Your unique tips (customize these!)
with st.sidebar:
    st.header("How to Protect Yourself")
    st.markdown("""
    - Never share OTPs or verification codes  
    - Check unrecognized merchants immediately  
    - Enable 2FA everywhere  
    - Review statements weekly  
    """)
    st.caption("Add your own tips here — this section is 100% yours!")

# File uploader
uploaded_file = st.file_uploader("Upload your bank CSV (Date, Amount, Merchant, etc.)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Basic cleaning - you can expand this
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Hour'] = df['Date'].dt.hour
        df['DayOfWeek'] = df['Date'].dt.dayofweek
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    st.write("Preview of your data:")
    st.dataframe(df.head())

    # Feature engineering for ML (customize columns here!)
    numeric_cols = ['Amount']
    if 'Hour' in df.columns:
        numeric_cols.append('Hour')
    if 'DayOfWeek' in df.columns:
        numeric_cols.append('DayOfWeek')
    
    X = df[numeric_cols].dropna()
    if len(X) > 50:  # Need enough data for ML
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Your ML model - Isolation Forest (unsupervised)
        model = IsolationForest(contamination=0.02, random_state=42)
        df['ML_Flag'] = model.fit_predict(X_scaled)
        df['ML_Flag'] = df['ML_Flag'].map({-1: "Suspicious", 1: "Normal"})
        
        # YOUR UNIQUE HYBRID RULES - Edit these to make it truly yours!
        df['Rule_Flag'] = "Normal"
        median_amount = df['Amount'].median()
        df.loc[df['Amount'] > 5 * median_amount, 'Rule_Flag'] = "High Amount"
        if 'Hour' in df.columns:
            df.loc[(df['Hour'] < 6) | (df['Hour'] > 22), 'Rule_Flag'] = "Odd Hours"
        
        # Final score - combine ML + rules (your secret sauce)
        df['Risk_Score'] = 0
        df.loc[df['ML_Flag'] == "Suspicious", 'Risk_Score'] += 70
        df.loc[df['Rule_Flag'] != "Normal", 'Risk_Score'] += 30
        df['Final_Status'] = np.where(df['Risk_Score'] > 50, "🚨 HIGH RISK", "✅ Safe")
        
        # Display results
        st.subheader("Flagged Transactions")
        st.dataframe(df[['Date', 'Amount', 'Merchant', 'Final_Status', 'Risk_Score']].sort_values('Risk_Score', ascending=False))
        
        # Charts (make yours stand out)
        fig = px.histogram(df, x="Amount", color="Final_Status", title="Spending Pattern with Flags")
        st.plotly_chart(fig, use_container_width=True)
        
        # Download button - super helpful for users
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download Full Report (CSV)", csv, "fraud_report.csv", "text/csv")
        
        st.success(f"Scanned {len(df)} transactions — {len(df[df['Final_Status'] == '🚨 HIGH RISK'])} flagged!")
    else:
        st.warning("Need at least 50 transactions for smart detection. Add more data!")

else:
    st.info("👆 Upload a CSV to start. Example columns: Date, Amount, Merchant, Description")
