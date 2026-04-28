import streamlit as st
import pandas as pd
import os
import json
from datetime import timedelta
from anthropic import Anthropic
from dotenv import load_dotenv
import numpy as np

st.set_page_config(page_title="Reorder Report", page_icon="📋", layout="wide")

# Load API key from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
api_key = os.getenv('ANTHROPIC_API_KEY')

css_path = os.path.join(os.path.dirname(__file__), '..', 'styles.css')
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="padding: 8px 0 20px 0; border-bottom: 0.5px solid #E8DDD5; margin-bottom: 8px;">
    <div style="font-size: 18px; font-weight: 500; color: #8B5E52;">MedStock AU</div>
    <div style="font-size: 11px; color: #B89080; margin-top: 2px;">Sydney Pharmacy Network</div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    df = pd.read_csv(os.path.join(base, 'pharmacy_demand_with_anomalies.csv'), parse_dates=['date'])
    return df

df = load_data()

st.markdown("""
<div style="padding: 8px 0 20px 0;">
    <h1 style="font-size: 24px;">Reorder Report</h1>
    <p style="color: #B89080; font-size: 13px;">
        LLM-generated replenishment recommendations · Powered by Claude AI
    </p>
</div>
""", unsafe_allow_html=True)

def get_stock_status(location, medication):
    subset = df[
        (df['location'].str.lower().str.contains(location.lower())) &
        (df['medication'].str.lower().str.contains(medication.lower()))
    ].sort_values('date')
    if subset.empty:
        return None
    latest     = subset.iloc[-1]
    avg_demand = subset.tail(7)['demand_units'].mean()
    reorder_pt = latest['reorder_point']
    stock      = max(0, int(reorder_pt - avg_demand * np.random.uniform(0.3, 0.9)))
    status     = 'CRITICAL' if stock < reorder_pt * 0.2 else \
                 'LOW'      if stock < reorder_pt * 0.5 else 'OK'
    return {
        'location'      : latest['location'],
        'medication'    : latest['medication'],
        'stock_units'   : stock,
        'reorder_point' : int(reorder_pt),
        'avg_demand_7d' : round(avg_demand, 1),
        'unit_cost_aud' : float(latest['unit_cost_aud']),
        'status'        : status,
        'days_remaining': round(stock / avg_demand, 1) if avg_demand > 0 else 0
    }

def get_network_summary():
    latest = df['date'].max()
    recent = df[df['date'] >= latest - timedelta(days=7)]
    rows   = []
    for loc in df['location'].unique():
        loc_data = recent[recent['location'] == loc]
        if loc_data.empty:
            continue
        rows.append({
            'location'        : loc,
            'location_type'   : loc_data['location_type'].iloc[0],
            'avg_daily_demand': round(loc_data['demand_units'].mean(), 1),
            'anomaly_count'   : int(loc_data['if_anomaly'].sum()),
            'top_medication'  : loc_data.groupby('medication')['demand_units'].mean().idxmax()
        })
    return rows

# ── Report type selector ─────────────────────────────────────────
st.markdown("#### Generate Report")

report_type = st.radio(
    "Report scope",
    ["Single Location & Medication", "Full Network Summary"],
    horizontal=True
)

if report_type == "Single Location & Medication":
    col1, col2 = st.columns(2)
    with col1:
        selected_loc = st.selectbox("Location", sorted(df['location'].unique().tolist()))
    with col2:
        selected_med = st.selectbox("Medication", sorted(df['medication'].unique().tolist()))

    stock_info = get_stock_status(selected_loc, selected_med)

    if stock_info:
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Current Stock", f"{stock_info['stock_units']} units")
        with m2:
            st.metric("Reorder Point", f"{stock_info['reorder_point']} units")
        with m3:
            st.metric("Avg Daily Demand", f"{stock_info['avg_demand_7d']} units")
        with m4:
            st.metric("Days Remaining", f"{stock_info['days_remaining']} days")

        if stock_info['status'] == 'CRITICAL':
            st.error("CRITICAL — Immediate replenishment required")
        elif stock_info['status'] == 'LOW':
            st.warning("LOW — Replenishment recommended within 24 hours")
        else:
            st.success("OK — Stock level is adequate")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Generate AI Report"):
        with st.spinner("Generating report..."):
            try:
                client = Anthropic(api_key=api_key)
                prompt = f"""You are a pharmaceutical inventory analyst for Sydney hospital pharmacies.

Based on the following inventory data, generate a concise professional replenishment report.

DATA:
{json.dumps(stock_info, indent=2)}

Write a report that includes:
1. Current inventory status summary
2. Key risks (stockouts, cost impact)
3. Specific replenishment recommendation with exact order quantity
4. Priority level (HIGH / MEDIUM / LOW)

Keep it under 250 words. Use clear headings. Be specific with numbers."""

                response = client.messages.create(
                    model      = 'claude-sonnet-4-5',
                    max_tokens = 500,
                    messages   = [{'role': 'user', 'content': prompt}]
                )
                report = response.content[0].text

                st.markdown("""
                <div style="background: #FFFFFF; border: 0.5px solid #EDE5E0;
                            border-radius: 14px; padding: 24px; margin-top: 16px;">
                """, unsafe_allow_html=True)
                st.markdown(report)
                st.markdown("</div>", unsafe_allow_html=True)

                st.download_button(
                    label     = "Download Report",
                    data      = report,
                    file_name = f"reorder_report_{selected_loc}_{selected_med}.txt",
                    mime      = "text/plain"
                )
            except Exception as e:
                st.error(f"Error generating report: {e}")

else:
    st.markdown("<br>", unsafe_allow_html=True)
    summary    = get_network_summary()
    summary_df = pd.DataFrame(summary)

    st.markdown("#### Network Status Overview")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Generate Full Network Report"):
        with st.spinner("Generating network-wide report..."):
            try:
                client = Anthropic(api_key=api_key)
                prompt = f"""You are a pharmaceutical inventory analyst for Sydney hospital pharmacies.

Generate a network-wide replenishment report based on this data:

{json.dumps(summary, indent=2)}

Include:
1. Executive summary of network status
2. Top 3 priority locations needing attention
3. Overall demand trends
4. Recommended actions by priority (HIGH / MEDIUM / LOW)

Keep it under 350 words. Use clear headings."""

                response = client.messages.create(
                    model      = 'claude-sonnet-4-5',
                    max_tokens = 600,
                    messages   = [{'role': 'user', 'content': prompt}]
                )
                report = response.content[0].text

                st.markdown("""
                <div style="background: #FFFFFF; border: 0.5px solid #EDE5E0;
                            border-radius: 14px; padding: 24px; margin-top: 16px;">
                """, unsafe_allow_html=True)
                st.markdown(report)
                st.markdown("</div>", unsafe_allow_html=True)

                st.download_button(
                    label     = "Download Network Report",
                    data      = report,
                    file_name = "reorder_report_network.txt",
                    mime      = "text/plain"
                )
            except Exception as e:
                st.error(f"Error generating report: {e}")