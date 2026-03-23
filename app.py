import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from fpdf import FPDF
from datetime import datetime
import re

# ====================== CONFIG ======================
st.set_page_config(page_title="FraudShield Ultra", page_icon="🛡️", layout="wide")
st.title("🛡️ FraudShield Ultra — Advanced Personal Anti-Fraud Scanner")
st.markdown("**Built by Ayden** — Deep habit analysis + multi-model ensemble + real Google merchant checks.")

# Fixed good defaults (no sidebar controls)
CONTAMINATION = 0.012
N_ESTIMATORS = 300

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Upload & Deep Analysis",
    "📊 Results Dashboard",
    "📈 Your Habit Profile",
    "🔍 Google Merchant Checks",
    "🧠 Why Flagged",
    "🔮 Simulator & Tips"
])

# ====================== HELPER FUNCTIONS ======================
def clean_and_enrich_data(df):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Hour'] = df['Date'].dt.hour
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def build_habit_baseline(df):
    day_stats = df.groupby('DayOfWeek')['Amount'].agg(['mean', 'median', 'std']).reset_index()
    day_stats.columns = ['DayOfWeek', 'Day_Mean', 'Day_Median', 'Day_Std']
    df = df.merge(day_stats, on='DayOfWeek', how='left')
    df['Dev_From_Day'] = (df['Amount'] - df['Day_Median']) / df['Day_Std'].replace(0, 1)
    
    for w in [3, 5, 7, 14]:
        df[f'Rolling_Mean_{w}'] = df['Amount'].rolling(w, min_periods=1).mean()
        df[f'Tx_Velocity_{w}'] = df['Amount'].rolling(w, min_periods=1).count()
    df['Z_Score'] = (df['Amount'] - df['Rolling_Mean_7']) / df['Rolling_Mean_7'].replace(0, 1)
    
    if 'Merchant' in df.columns:
        df['Merchant'] = df['Merchant'].astype(str).str.strip().str.lower()
        df['Merchant_Freq'] = df['Merchant'].map(df['Merchant'].value_counts())
        df['Rarity'] = 1 / (df['Merchant_Freq'] + 1)
        df['Is_New_Merchant'] = (~df['Merchant'].isin(df['Merchant'].shift(1))).astype(int)
        df['Suspicious_Name'] = df['Merchant'].str.contains(
            r'(free|giftcard|prize|win|refund|support|amaz0n|paypa1|\d{10,}|lllll)', 
            regex=True, na=False
        ).astype(int)
    
    return df

def run_ensemble_models(X_scaled):
    iso = IsolationForest(contamination=CONTAMINATION, n_estimators=N_ESTIMATORS, random_state=42)
    lof = LocalOutlierFactor(contamination=CONTAMINATION, novelty=True)
    ocsvm = OneClassSVM(nu=CONTAMINATION)
    
    iso_score = -iso.fit_predict(X_scaled)
    
    lof.fit(X_scaled)
    lof_score = -lof.score_samples(X_scaled)
    
    ocsvm.fit(X_scaled)
    ocsvm_score = (ocsvm.predict(X_scaled) == -1).astype(int)
    
    ensemble_score = 0.5 * iso_score + 0.3 * lof_score + 0.2 * ocsvm_score
    return ensemble_score

# ====================== TAB 1: UPLOAD & ANALYSIS ======================
with tab1:
    uploaded_file = st.file_uploader("Upload your bank/credit card CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = clean_and_enrich_data(df)
            df = build_habit_baseline(df)
            
            # Column mapper fallback
            if 'Merchant' not in df.columns:
                possible_merchant_cols = [col for col in df.columns if 'merchant' in col.lower() or 'vendor' in col.lower() or 'payee' in col.lower()]
                if possible_merchant_cols:
                    selected = st.selectbox("Which column is the Merchant/Payee?", possible_merchant_cols)
                    df = df.rename(columns={selected: 'Merchant'})
                else:
                    st.warning("No obvious merchant column found. Analysis will be less accurate.")
            
            feature_cols = ['Amount', 'Hour', 'DayOfWeek', 'Dev_From_Day', 'Z_Score', 'Rarity', 
                            'Is_New_Merchant', 'Is_Weekend', 'Suspicious_Name']
            feature_cols += [c for c in df.columns if 'Rolling' in c or 'Velocity' in c]
            
            X = df[feature_cols].dropna()
            
            if len(X) > 20:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                df['Anomaly_Score'] = run_ensemble_models(X_scaled)
                threshold = np.percentile(df['Anomaly_Score'], 95)
                df['Final_Status'] = np.where(df['Anomaly_Score'] > threshold, "🚨 HIGH RISK", "✅ Safe")
                
                df['Why_Flagged'] = ""
                df.loc[df['Dev_From_Day'] > 2.5, 'Why_Flagged'] += "Breaks your typical day-of-week pattern; "
                df.loc[df['Suspicious_Name'] == 1, 'Why_Flagged'] += "Suspicious merchant name pattern; "
                df.loc[df['Rarity'] > 0.3, 'Why_Flagged'] += "Very rare merchant for you; "
                df['Why_Flagged'] = df['Why_Flagged'].str.rstrip("; ")
                
                st.session_state.df = df
                st.session_state.processed = True
                
                st.success(f"Processed {len(df)} transactions — {len(df[df['Final_Status'] == '🚨 HIGH RISK'])} flagged.")
            else:
                st.warning("Need at least 20 valid rows after cleaning for full model analysis.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Upload a CSV file to start analysis.")

# ====================== TAB 2: RESULTS DASHBOARD ======================
with tab2:
    if st.session_state.get("processed", False):
        df = st.session_state.df
        cols_to_show = ['Date', 'Amount', 'Merchant', 'Final_Status', 'Why_Flagged', 'Anomaly_Score']
        st.dataframe(df[cols_to_show].sort_values('Anomaly_Score', ascending=False))
        
        fig1 = px.histogram(df, x="Amount", color="Final_Status", title="Amount Distribution by Risk")
        fig2 = px.scatter(df, x="Date", y="Amount", color="Final_Status", title="Transaction Timeline")
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        if st.button("📥 Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "FraudShield Report — Built by Ayden", ln=1, align='C')
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
            pdf.ln(10)
            high_risk = df[df['Final_Status'] == "🚨 HIGH RISK"]
            for _, row in high_risk.iterrows():
                pdf.cell(0, 10, f"{row['Date'].strftime('%Y-%m-%d %H:%M')} | ${row['Amount']:.2f} | {row.get('Merchant', 'N/A')} | {row['Why_Flagged']}", ln=1)
            pdf.output("FraudShield_Report.pdf")
            with open("FraudShield_Report.pdf", "rb") as f:
                st.download_button("Download PDF Report", f, "FraudShield_Report.pdf", "application/pdf")
    else:
        st.info("Upload and analyze a file first.")

# ====================== TAB 3: HABIT PROFILE ======================
with tab3:
    if st.session_state.get("processed", False):
        df = st.session_state.df
        st.subheader("Your Personal Spending Habits")
        st.metric("Average transaction", f"${df['Amount'].mean():.2f}")
        st.metric("Most common merchant", df['Merchant'].mode()[0] if 'Merchant' in df.columns else "N/A")
        st.plotly_chart(px.box(df, x="DayOfWeek", y="Amount", title="Spending by Day of Week"), use_container_width=True)
    else:
        st.info("Upload data to see your habit profile.")

# ====================== TAB 4: GOOGLE MERCHANT CHECKS ======================
with tab4:
    st.subheader("🔍 Google Merchant Reputation Check")
    st.markdown("For flagged transactions, click to search Google for scam reports / reviews.")
    
    if st.session_state.get("processed", False):
        df = st.session_state.df
        high_risk = df[df['Final_Status'] == "🚨 HIGH RISK"]
        merchants_checked = set()
        
        for _, row in high_risk.iterrows():
            m = row.get('Merchant', '').strip()
            if m and m not in merchants_checked:
                merchants_checked.add(m)
                clean_m = m.replace(" ", "+")
                url = f"https://www.google.com/search?q={clean_m}+scam+reviews+fraud+complaints+BBB+Trustpilot"
                st.markdown(f"**{m}** → [🔍 Search Google]({url})")
    else:
        st.info("Analyze data first to see merchant checks.")

# ====================== TAB 5: WHY FLAGGED EXPLANATIONS ======================
with tab5:
    if st.session_state.get("processed", False):
        df = st.session_state.df
        st.subheader("Explanation of Flags")
        st.dataframe(df[df['Final_Status'] == "🚨 HIGH RISK"][['Date', 'Amount', 'Merchant', 'Why_Flagged', 'Anomaly_Score']])
        st.caption("The model combines personal habits + merchant patterns + time anomalies.")
    else:
        st.info("No data analyzed yet.")

# ====================== TAB 6: SIMULATOR & TIPS ======================
with tab6:
    st.subheader("Quick What-If Simulator")
    sim_amount = st.number_input("Amount ($)", min_value=0.0, value=150.0, step=1.0)
    sim_merchant = st.text_input("Merchant name", "Example Shop")
    
    if st.button("Score this transaction"):
        if st.session_state.get("processed", False):
            df = st.session_state.df
            sim_dev = abs(sim_amount - df['Amount'].median()) / df['Amount'].std() if df['Amount'].std() > 0 else 0
            risk_level = "High" if sim_dev > 3 else "Medium" if sim_dev > 1.5 else "Low"
        else:
            risk_level = "Unknown (no baseline yet)"
        
        st.write(f"**Estimated risk:** {risk_level}")
        
        clean_m = sim_merchant.replace(" ", "+")
        url = f"https://www.google.com/search?q={clean_m}+scam+reviews+fraud+complaints"
        st.markdown(f"**Merchant reputation check:** [Search Google for {sim_merchant}]({url})")
    
    st.markdown("### Prevention Tips")
    st.markdown("""
    • Turn on transaction alerts  
    • Never share OTPs or verification codes  
    • Review statements weekly  
    • Use virtual cards for online shopping  
    • Freeze your card immediately if something looks wrong
    """)

st.caption("FraudShield Ultra — Private local analysis. No data leaves your device.")
