import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from datetime import timedelta

st.set_page_config(page_title="Medication Search", page_icon="🔍", layout="wide")

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
    <h1 style="font-size: 24px;">Medication Search & Forecast</h1>
    <p style="color: #B89080; font-size: 13px;">Query demand history and forecast for any medication across Sydney locations</p>
</div>
""", unsafe_allow_html=True)

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    selected_med = st.selectbox(
        "Select Medication",
        sorted(df['medication'].unique()),
        index=list(sorted(df['medication'].unique())).index('Paracetamol')
    )
with col_sel2:
    selected_loc = st.selectbox(
        "Select Location",
        ["All Locations"] + sorted(df['location'].unique().tolist())
    )

st.markdown("<br>", unsafe_allow_html=True)

# Filter data
if selected_loc == "All Locations":
    filtered = df[df['medication'] == selected_med]
else:
    filtered = df[
        (df['medication'] == selected_med) &
        (df['location'] == selected_loc)
    ]

latest_date = filtered['date'].max()
recent_7d   = filtered[filtered['date'] >= latest_date - timedelta(days=7)]
recent_30d  = filtered[filtered['date'] >= latest_date - timedelta(days=30)]

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg Daily Demand (7d)",
              f"{recent_7d['demand_units'].mean():.0f} units")
with col2:
    st.metric("Peak Demand (30d)",
              f"{recent_30d['demand_units'].max()} units")
with col3:
    st.metric("Anomalies (30d)",
              int(recent_30d['if_anomaly'].sum()))
with col4:
    stockout_rate = (recent_30d['demand_units'] == 0).mean() * 100
    st.metric("Stockout Rate (30d)", f"{stockout_rate:.1f}%")

st.markdown("---")

# Demand history chart
st.markdown("#### Demand History")

if selected_loc == "All Locations":
    monthly = filtered.groupby(
        [filtered['date'].dt.to_period('M'), 'location']
    )['demand_units'].mean().reset_index()
    monthly['date'] = monthly['date'].dt.to_timestamp()

    fig1 = px.line(
        monthly,
        x      = 'date',
        y      = 'demand_units',
        color  = 'location',
        labels = {'demand_units': 'Avg Units/Day', 'date': '', 'location': 'Location'},
        color_discrete_sequence = [
            '#C47B70', '#D4A099', '#E8C4BC',
            '#B89080', '#8B6B60', '#6B4440',
            '#F0D8D4', '#FAF7F4'
        ]
    )
else:
    daily = filtered[['date', 'demand_units', 'if_anomaly']].copy()
    anomalies = daily[daily['if_anomaly'] == 1]

    fig1 = px.line(
        daily,
        x      = 'date',
        y      = 'demand_units',
        labels = {'demand_units': 'Units/Day', 'date': ''},
        color_discrete_sequence = ['#C47B70']
    )
    fig1.add_scatter(
        x    = anomalies['date'],
        y    = anomalies['demand_units'],
        mode = 'markers',
        marker = dict(color='#E24B4A', size=6),
        name = 'Anomaly'
    )

fig1.update_layout(
    plot_bgcolor  = '#FFFFFF',
    paper_bgcolor = '#FFFFFF',
    font_family   = 'DM Sans',
    font_color    = '#6B4440',
    margin        = dict(l=0, r=0, t=10, b=0),
    height        = 280,
)
fig1.update_xaxes(showgrid=False)
fig1.update_yaxes(showgrid=True, gridcolor='#EDE5E0')
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Demand by Season")
    season_avg = filtered.groupby('season')['demand_units'].mean().reset_index()
    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
    season_avg['season'] = pd.Categorical(season_avg['season'], categories=season_order, ordered=True)
    season_avg = season_avg.sort_values('season')

    fig2 = px.bar(
        season_avg,
        x     = 'season',
        y     = 'demand_units',
        labels = {'demand_units': 'Avg Units/Day', 'season': ''},
        color_discrete_sequence = ['#C47B70']
    )
    fig2.update_layout(
        plot_bgcolor  = '#FFFFFF',
        paper_bgcolor = '#FFFFFF',
        font_family   = 'DM Sans',
        font_color    = '#6B4440',
        margin        = dict(l=0, r=0, t=10, b=0),
        height        = 250,
    )
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=True, gridcolor='#EDE5E0')
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.markdown("#### Weekday vs Weekend")
    weekly = filtered.groupby('is_weekend')['demand_units'].mean().reset_index()
    weekly['day_type'] = weekly['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})

    fig3 = px.bar(
        weekly,
        x     = 'day_type',
        y     = 'demand_units',
        labels = {'demand_units': 'Avg Units/Day', 'day_type': ''},
        color_discrete_sequence = ['#C47B70', '#D4A099']
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