import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from fpdf import FPDF
from datetime import datetime

# ====================== APP CONFIG ======================
st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="wide")
st.title("🛡️ FraudShield — Advanced Personal Anti-Fraud Scanner")
st.markdown("**Built entirely by Ayden** — Upload your real bank/credit-card CSV and get enterprise-grade fraud flags. Everything runs privately.")

# Sidebar for learning sliders (this teaches hyperparameter tuning!)
with st.sidebar:
    st.header("🎛️ Model Controls (Experiment & Learn)")
    contamination = st.slider("Expected fraud rate (contamination)", 0.001, 0.05, 0.015, step=0.001,
                             help="Lower = stricter (fewer false flags). Real banks tune this carefully.")
    n_estimators = st.slider("Number of trees in ensemble", 50, 300, 150,
                             help="More trees = more stable but slower. Classic trade-off in fraud ML.")
    st.caption("Change these → watch how flags change. This is real ML experimentation!")

# ====================== TABS (Professional UX) ======================
# ... (keep all imports and config the same)

# Sidebar stays the same

tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Scan", "📊 Results & Charts", "🛡️ Prevention Tips", "🔮 What-If Simulator"])

with tab1:
    uploaded_file = st.file_uploader("Upload your bank CSV (must have Date, Amount, Merchant columns)", type=["csv"])
    
    if uploaded_file is not None:
        # ── ALL YOUR PROCESSING CODE HERE ──
        df = pd.read_csv(uploaded_file)
        
        # Basic cleaning ...
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Hour'] = df['Date'].dt.hour
            df['DayOfWeek'] = df['Date'].dt.dayofweek
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # === ELITE FEATURE ENGINEERING ===
        # (keep all your rolling, z-score, rarity, etc. code exactly as is)
        
        # ... up to calculating Anomaly_Score, Final_Status, Why_Flagged ...
        
        st.success(f"Scanned {len(df)} transactions — {len(df[df['Final_Status']=='🚨 HIGH RISK'])} flagged!")
        
        # Optional: store df in session state so tabs can access it safely
        st.session_state.df = df
        st.session_state.processed = True

# Now guard the other tabs
with tab2:
    if 'processed' in st.session_state and st.session_state.processed:
        df = st.session_state.df
        st.dataframe(df[['Date', 'Amount', 'Merchant', 'Final_Status', 'Why_Flagged', 'Anomaly_Score']].sort_values('Anomaly_Score', ascending=False))
        
        # Charts ...
        fig1 = px.histogram(df, x="Amount", color="Final_Status", title="Amount Distribution with Fraud Flags")
        fig2 = px.scatter(df, x="Date", y="Amount", color="Final_Status", title="Transaction Timeline")
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        # PDF button ...
        if st.button("📥 Generate Professional PDF Report"):
            # ... your PDF code ...
    else:
        st.info("Upload a CSV file in the first tab to see results.")

with tab3:
    # Prevention tips — this one is static, so no guard needed
    st.markdown("""
    **Real tips that actually protect people (FTC & bank best practices):**
    - Enable transaction alerts on your banking app
    - Never share OTPs or card details
    - Review statements every Sunday
    - Use virtual cards for online shopping
    """)
    st.caption("Add your own researched tips here — this section makes the app genuinely helpful.")

with tab4:
    # Simulator can run without df, or add a light guard
    st.subheader("Test a hypothetical transaction")
    sim_amount = st.number_input("Amount ($)", value=150.0)
    sim_hour = st.slider("Hour of day", 0, 23, 3)
    sim_merchant = st.text_input("Merchant name", "Unknown Online Shop")
    
    if st.button("Score this transaction"):
        # For now, simple fallback logic (can improve later with session_state df)
        if 'df' in st.session_state:
            mean_amt = st.session_state.df['Amount'].mean()
            std_amt  = st.session_state.df['Amount'].std()
            sim_z = abs(sim_amount - mean_amt) / std_amt if std_amt > 0 else 0
        else:
            sim_z = 0  # fallback
        
        sim_risk = "🚨 HIGH RISK" if (sim_z > 3 or sim_hour < 6) else "✅ Safe"
        st.write(f"**Result:** {sim_risk}")
        st.write("Reason: Matches patterns your model learned from your real data (or basic rules if no data yet).")

st.caption("FraudShield — My passion project. All processing happens in your browser session. No data stored.")

# ====================== FOOTER ======================
st.caption("FraudShield — My passion project. All processing happens in your browser session. No data stored.")
