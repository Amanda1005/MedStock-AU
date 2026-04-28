import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import timedelta

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

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
    <div style="font-size: 13px; color: #C4A090; margin-bottom: 4px;">
        AI-Powered Pharmaceutical Demand Forecasting & Inventory Optimisation
    </div>
    <h1 style="font-size: 28px;"> Network Overview</h1>
    <p style="color: #B89080; font-size: 13px;">
        Sydney pharmacy network · 8 locations · 15 medications · Real-time AI monitoring
    </p>
</div>
""", unsafe_allow_html=True)

latest_date = df['date'].max()
recent_7d   = df[df['date'] >= latest_date - timedelta(days=7)]
recent_30d  = df[df['date'] >= latest_date - timedelta(days=30)]

total_locations  = df['location'].nunique()
total_meds       = df['medication'].nunique()
anomalies_7d     = int(recent_7d['if_anomaly'].sum())
avg_demand_today = int(recent_7d['demand_units'].mean())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Locations Monitored", total_locations)
with col2:
    st.metric("Medications Tracked", total_meds)
with col3:
    st.metric("Anomalies (7 days)", anomalies_7d, delta="vs normal range")
with col4:
    st.metric("Avg Daily Demand", f"{avg_demand_today} units")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.markdown("#### Average Daily Demand by Location")
    loc_avg = df.groupby(['location', 'location_type'])['demand_units'].mean().reset_index()
    loc_avg = loc_avg.sort_values('demand_units', ascending=True)

    fig1 = px.bar(
        loc_avg,
        x            = 'demand_units',
        y            = 'location',
        orientation  = 'h',
        color        = 'location_type',
        color_discrete_map = {'hospital': '#C47B70', 'retail': '#D4A099'},
        labels       = {'demand_units': 'Avg Units/Day', 'location': ''},
    )
    fig1.update_layout(
        plot_bgcolor  = '#FFFFFF',
        paper_bgcolor = '#FFFFFF',
        font_family   = 'DM Sans',
        font_color    = '#6B4440',
        legend_title  = 'Type',
        margin        = dict(l=0, r=0, t=10, b=0),
        height        = 300,
    )
    fig1.update_xaxes(showgrid=True, gridcolor='#EDE5E0')
    fig1.update_yaxes(showgrid=False)
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.markdown("#### Demand by Category")
    cat_avg = df.groupby('category')['demand_units'].mean().reset_index()
    cat_avg = cat_avg.sort_values('demand_units', ascending=False)

    fig2 = px.pie(
        cat_avg,
        values = 'demand_units',
        names  = 'category',
        color_discrete_sequence = [
            '#C47B70', '#D4A099', '#E8C4BC', '#F0D8D4',
            '#B89080', '#8B6B60', '#6B4440', '#F5EDE8',
            '#EDE5E0', '#FAF7F4', '#E8DDD5'
        ]
    )
    fig2.update_layout(
        plot_bgcolor  = '#FFFFFF',
        paper_bgcolor = '#FFFFFF',
        font_family   = 'DM Sans',
        font_color    = '#6B4440',
        margin        = dict(l=0, r=0, t=10, b=0),
        height        = 300,
        legend        = dict(font=dict(size=10))
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("#### Monthly Demand Trend — All Locations")

monthly = df.groupby([df['date'].dt.to_period('M'), 'location_type'])['demand_units'].mean().reset_index()
monthly['date'] = monthly['date'].dt.to_timestamp()

fig3 = px.line(
    monthly,
    x     = 'date',
    y     = 'demand_units',
    color = 'location_type',
    color_discrete_map = {'hospital': '#C47B70', 'retail': '#D4A099'},
    labels = {'demand_units': 'Avg Units/Day', 'date': '', 'location_type': 'Type'}
)
fig3.update_layout(
    plot_bgcolor  = '#FFFFFF',
    paper_bgcolor = '#FFFFFF',
    font_family   = 'DM Sans',
    font_color    = '#6B4440',
    margin        = dict(l=0, r=0, t=10, b=0),
    height        = 250,
)
fig3.update_xaxes(showgrid=False)
fig3.update_yaxes(showgrid=True, gridcolor='#EDE5E0')
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("#### Recent Anomaly Summary (Last 30 Days)")

anomaly_summary = recent_30d[recent_30d['if_anomaly'] == 1].groupby(
    ['location', 'medication']
)['if_anomaly'].count().reset_index()
anomaly_summary.columns = ['Location', 'Medication', 'Anomaly Count']
anomaly_summary = anomaly_summary.sort_values('Anomaly Count', ascending=False).head(10)

st.dataframe(
    anomaly_summary,
    use_container_width = True,
    hide_index          = True
)