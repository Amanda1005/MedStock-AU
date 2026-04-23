import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import timedelta
from anthropic import Anthropic
from dotenv import load_dotenv

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")

# Load API key from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
api_key = os.getenv('ANTHROPIC_API_KEY')

css_path = os.path.join(os.path.dirname(__file__), '..', 'styles.css')
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="padding: 8px 0 20px 0; border-bottom: 0.5px solid #E8DDD5; margin-bottom: 8px;">
    <div style="font-size: 18px; font-weight: 500; color: #8B5E52;">💊 MedStock AU</div>
    <div style="font-size: 11px; color: #B89080; margin-top: 2px;">Sydney Pharmacy Network</div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    df = pd.read_csv(os.path.join(base, 'pharmacy_demand_with_anomalies.csv'), parse_dates=['date'])
    return df

df = load_data()

def get_current_stock(location, medication):
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

def get_demand_forecast(location, medication, days=7):
    subset = df[
        (df['location'].str.lower().str.contains(location.lower())) &
        (df['medication'].str.lower().str.contains(medication.lower()))
    ].sort_values('date').tail(30)
    if subset.empty:
        return None
    avg   = subset['demand_units'].mean()
    trend = subset['demand_units'].iloc[-7:].mean() - subset['demand_units'].iloc[:7].mean()
    fc    = [max(0, int(avg + trend * 0.1 + np.random.normal(0, avg * 0.05)))
             for _ in range(days)]
    return {
        'location'      : subset.iloc[-1]['location'],
        'medication'    : subset.iloc[-1]['medication'],
        'forecast_days' : days,
        'daily_forecast': fc,
        'avg_forecast'  : round(np.mean(fc), 1),
        'trend'         : 'INCREASING' if trend > 5 else
                          'DECREASING' if trend < -5 else 'STABLE',
    }

def get_anomalies(location=None, days=14):
    latest = df['date'].max()
    cutoff = latest - timedelta(days=days)
    subset = df[df['date'] >= cutoff]
    if location:
        subset = subset[subset['location'].str.lower().str.contains(location.lower())]
    anom = subset[subset['if_anomaly'] == 1][
        ['date', 'location', 'medication', 'demand_units']
    ].sort_values('date', ascending=False)
    return {
        'total_anomalies': len(anom),
        'period_days'    : days,
        'records'        : anom.head(5).to_dict('records')
    }

def get_data_context(message):
    msg = message.lower()
    locations_map = {
        'rpa': 'RPA', 'royal prince alfred': 'RPA',
        'westmead': 'Westmead',
        "st vincent": 'St Vincent',
        'prince of wales': 'Prince of Wales',
        'epping': 'Epping',
        'cbd': 'Sydney CBD', 'sydney cbd': 'Sydney CBD',
        'pitt street': 'Pitt Street', 'priceline': 'Pitt Street',
        'parramatta': 'Parramatta', 'terrywhite': 'Parramatta'
    }
    meds_map = {
        'paracetamol': 'Paracetamol', 'ibuprofen': 'Ibuprofen',
        'amoxicillin': 'Amoxicillin', 'metformin': 'Metformin',
        'atorvastatin': 'Atorvastatin', 'omeprazole': 'Omeprazole',
        'salbutamol': 'Salbutamol', 'pantoprazole': 'Pantoprazole',
        'codeine': 'Codeine', 'ondansetron': 'Ondansetron',
        'enoxaparin': 'Enoxaparin', 'dexamethasone': 'Dexamethasone',
        'sertraline': 'Sertraline', 'rosuvastatin': 'Rosuvastatin',
        'cetirizine': 'Cetirizine'
    }
    detected_loc = next((v for k, v in locations_map.items() if k in msg), None)
    detected_med = next((v for k, v in meds_map.items() if k in msg), None)
    context = ""
    if detected_loc and detected_med:
        stock    = get_current_stock(detected_loc, detected_med)
        forecast = get_demand_forecast(detected_loc, detected_med)
        anomaly  = get_anomalies(detected_loc, days=14)
        context  = json.dumps({
            'stock': stock, 'forecast': forecast, 'anomalies': anomaly
        }, indent=2, default=str)
    elif detected_loc:
        anomaly = get_anomalies(detected_loc, days=14)
        context = json.dumps({'anomalies': anomaly}, indent=2, default=str)
    elif any(w in msg for w in ['anomal', 'alert', 'spike', 'reorder', 'critically']):
        context = json.dumps({'recent_anomalies': get_anomalies(days=7)},
                             indent=2, default=str)
    return context

# ── Page header ──────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 16px 0;">
    <h1 style="font-size: 24px;">💬 MedStock Assistant</h1>
    <p style="color: #B89080; font-size: 13px;">
        Ask me anything about Sydney pharmacy inventory, forecasts, and alerts
    </p>
</div>
""", unsafe_allow_html=True)

# ── Quick chips ──────────────────────────────────────────────────
st.markdown("""
<p style="font-size: 12px; color: #B89080; margin-bottom: 8px;">Quick questions:</p>
""", unsafe_allow_html=True)

quick_questions = [
    ("🚨 RPA Paracetamol stock?", "Does RPA Hospital have Paracetamol in stock?"),
    ("⚠️ Any anomalies today?",   "Are there any anomaly alerts across all Sydney locations today?"),
    ("📦 What to reorder?",       "Which medications are critically low and need urgent reordering?"),
    ("📈 Amoxicillin forecast",   "What is the 7-day demand forecast for Amoxicillin at Westmead Hospital?"),
]

chip_cols = st.columns(4)
for i, (col, (label, question)) in enumerate(zip(chip_cols, quick_questions)):
    with col:
        if st.button(label, key=f"chip_{i}", use_container_width=True):
            st.session_state['quick_input'] = question

st.markdown("<br>", unsafe_allow_html=True)

# ── Chat history ─────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if not st.session_state.messages:
    st.session_state.messages.append({
        'role'   : 'assistant',
        'content': "Hi! I'm MedStock Assistant. I can help you check stock levels, demand forecasts, anomaly alerts, and replenishment recommendations across all 8 Sydney pharmacy locations. What would you like to know?"
    })

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# ── System prompt ────────────────────────────────────────────────
system_prompt = """You are MedStock Assistant, an AI-powered pharmaceutical inventory chatbot strictly for Sydney hospital and retail pharmacies.

You ONLY answer questions related to:
- Current stock levels and status (OK / LOW / CRITICAL)
- 7-day demand forecasts for specific medications
- Anomaly alerts and unusual demand spikes
- Replenishment recommendations with specific quantities
- Inventory insights across the Sydney pharmacy network

You have data for these locations ONLY:
- RPA Hospital Pharmacy (Camperdown)
- Westmead Hospital Pharmacy
- St Vincent's Hospital Pharmacy (Darlinghurst)
- Prince of Wales Hospital Pharmacy (Randwick)
- Chemist Warehouse Epping
- Chemist Warehouse Sydney CBD
- Priceline Pharmacy Pitt Street
- TerryWhite Chemmart Parramatta

Available medications: Paracetamol, Ibuprofen, Amoxicillin, Metformin, Atorvastatin, Omeprazole, Salbutamol, Pantoprazole, Codeine, Ondansetron, Enoxaparin, Dexamethasone, Sertraline, Rosuvastatin, Cetirizine

STRICT RULES:
- If the question is NOT about pharmacy inventory, stock, forecasts, anomalies, or replenishment — respond ONLY with: "I'm sorry, I'm only able to assist with pharmacy inventory-related queries. Please ask me about stock levels, demand forecasts, or replenishment recommendations."
- If asked about a location or medication not in your list — respond: "I'm sorry, that location or medication is not currently monitored in our system."
- Never answer questions about phone numbers, addresses, opening hours, prices, staff, or any non-inventory topics.
- Always be concise and specific with numbers.
- Always mention stock status (OK/LOW/CRITICAL) when discussing inventory."""

# ── Input handling ───────────────────────────────────────────────
if 'quick_input' in st.session_state:
    user_input = st.session_state.pop('quick_input')
else:
    user_input = st.chat_input("Ask about stock, forecasts, anomalies...")

if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    context      = get_data_context(user_input)
    full_message = user_input + (f"\n\nCURRENT DATA:\n{context}" if context else "")

    st.session_state.conversation_history.append({
        'role': 'user', 'content': full_message
    })

    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            try:
                client   = Anthropic(api_key=api_key)
                response = client.messages.create(
                    model      = 'claude-sonnet-4-5',
                    max_tokens = 400,
                    system     = system_prompt,
                    messages   = st.session_state.conversation_history
                )
                reply = response.content[0].text
                st.markdown(reply)
                st.session_state.conversation_history.append({
                    'role': 'assistant', 'content': reply
                })
                st.session_state.messages.append({
                    'role': 'assistant', 'content': reply
                })
            except Exception as e:
                st.error(f"Error: {e}")