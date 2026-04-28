import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import timedelta

st.set_page_config(page_title="Anomaly Alerts", page_icon="⚠️", layout="wide")

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
    <h1 style="font-size: 24px;">Anomaly Alerts</h1>
    <p style="color: #B89080; font-size: 13px;">
        AI-detected demand anomalies via Isolation Forest · Sydney pharmacy network
    </p>
</div>
""", unsafe_allow_html=True)

# ── Filters ──────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    period = st.selectbox("Time Period", ["Last 7 days", "Last 30 days", "Last 90 days", "All time"])
with col_f2:
    loc_options = ["All locations"] + sorted(df['location'].unique().tolist())
    selected_loc = st.selectbox("Location", loc_options)
with col_f3:
    med_options = ["All medications"] + sorted(df['medication'].unique().tolist())
    selected_med = st.selectbox("Medication", med_options)

period_map = {
    "Last 7 days" : 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "All time"    : 99999
}

latest_date = df['date'].max()
cutoff      = latest_date - timedelta(days=period_map[period])
filtered    = df[df['date'] >= cutoff].copy()

if selected_loc != "All locations":
    filtered = filtered[filtered['location'] == selected_loc]
if selected_med != "All medications":
    filtered = filtered[filtered['medication'] == selected_med]

anomalies = filtered[filtered['if_anomaly'] == 1].copy()

# ── Metric row ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Anomalies", f"{len(anomalies):,}")
with col2:
    anomaly_rate = len(anomalies) / len(filtered) * 100 if len(filtered) > 0 else 0
    st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
with col3:
    top_med = anomalies['medication'].value_counts().index[0] if len(anomalies) > 0 else "—"
    st.metric("Most Affected Med", top_med)
with col4:
    top_loc = anomalies['location'].value_counts().index[0] if len(anomalies) > 0 else "—"
    st.metric("Most Affected Location", top_loc.split()[0])

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ── Charts ───────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Anomalies by Medication")
    if len(anomalies) > 0:
        med_counts = anomalies['medication'].value_counts().reset_index()
        med_counts.columns = ['Medication', 'Count']
        fig1 = px.bar(
            med_counts,
            x                  = 'Count',
            y                  = 'Medication',
            orientation        = 'h',
            color_discrete_sequence = ['#C47B70'],
        )
        fig1.update_layout(
            plot_bgcolor  = '#FFFFFF',
            paper_bgcolor = '#FFFFFF',
            font_family   = 'DM Sans',
            font_color    = '#6B4440',
            margin        = dict(l=0, r=0, t=10, b=0),
            height        = 320,
            showlegend    = False,
        )
        fig1.update_xaxes(showgrid=True, gridcolor='#EDE5E0')
        fig1.update_yaxes(showgrid=False, categoryorder='total ascending')
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No anomalies found for selected filters.")

with col_right:
    st.markdown("#### Anomalies by Location")
    if len(anomalies) > 0:
        loc_counts = anomalies['location'].value_counts().reset_index()
        loc_counts.columns = ['Location', 'Count']
        loc_counts['Short'] = loc_counts['Location'].str.split().str[:2].str.join(' ')
        fig2 = px.bar(
            loc_counts,
            x                  = 'Count',
            y                  = 'Short',
            orientation        = 'h',
            color_discrete_sequence = ['#D4A099'],
        )
        fig2.update_layout(
            plot_bgcolor  = '#FFFFFF',
            paper_bgcolor = '#FFFFFF',
            font_family   = 'DM Sans',
            font_color    = '#6B4440',
            margin        = dict(l=0, r=0, t=10, b=0),
            height        = 320,
            showlegend    = False,
        )
        fig2.update_xaxes(showgrid=True, gridcolor='#EDE5E0')
        fig2.update_yaxes(showgrid=False, categoryorder='total ascending')
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Time series ──────────────────────────────────────────────────
st.markdown("#### Anomaly Timeline")

if len(anomalies) > 0 and selected_med != "All medications":
    plot_df   = filtered[filtered['medication'] == selected_med].copy()
    normal    = plot_df[plot_df['if_anomaly'] == 0]
    anom_pts  = plot_df[plot_df['if_anomaly'] == 1]

    fig3 = px.line(
        plot_df,
        x      = 'date',
        y      = 'demand_units',
        labels = {'demand_units': 'Units/Day', 'date': ''},
        color_discrete_sequence = ['#EDE5E0'],
    )
    fig3.add_scatter(
        x    = anom_pts['date'],
        y    = anom_pts['demand_units'],
        mode = 'markers',
        marker = dict(color='#C47B70', size=8),
        name = 'Anomaly'
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
else:
    st.info("Select a specific medication to view the anomaly timeline.")

st.markdown("---")

# ── Anomaly table ────────────────────────────────────────────────
st.markdown("#### Anomaly Records")

if len(anomalies) > 0:
    display = anomalies[['date', 'location', 'medication', 'category',
                          'demand_units', 'reorder_point', 'anomaly_score']].copy()
    display['date']          = display['date'].dt.strftime('%Y-%m-%d')
    display['anomaly_score'] = display['anomaly_score'].round(3)
    display                  = display.sort_values('date', ascending=False)
    display.columns          = ['Date', 'Location', 'Medication', 'Category',
                                 'Demand Units', 'Reorder Point', 'Anomaly Score']

    def highlight_rows(row):
        if row['Demand Units'] == 0:
            return ['background-color: #FCEBEB'] * len(row)
        elif row['Demand Units'] > row['Reorder Point'] * 2:
            return ['background-color: #FAEEDA'] * len(row)
        return [''] * len(row)

    st.dataframe(
        display.head(50),
        use_container_width = True,
        hide_index          = True
    )
    st.caption(f"Showing top 50 of {len(anomalies):,} anomaly records")
else:
    st.success("No anomalies detected for the selected filters.")