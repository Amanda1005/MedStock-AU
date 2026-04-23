# MedStock-AU 💊

**AI-Powered Pharmaceutical Demand Forecasting and Inventory Optimisation System for Australian Hospital Pharmacies**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

MedStock-AU is an end-to-end AI pipeline designed to address a critical gap in Australian hospital pharmacy management — the inability to anticipate demand spikes before they cause stockouts or costly overstock situations.

The system integrates multiple AI paradigms:
- **Time series forecasting** (LSTM) to predict future pharmaceutical demand
- **Unsupervised anomaly detection** (Isolation Forest) to flag unusual demand patterns
- **Reinforcement learning** (PPO) to optimise dynamic replenishment decisions
- **Large language models** (Claude API) to generate human-readable reports and power a conversational chatbot

### Target Users
- Hospital pharmacy managers — RPA, Westmead, St Vincent's, Prince of Wales
- Retail pharmacy chains — Chemist Warehouse, Priceline, TerryWhite Chemmart
- Procurement and inventory teams across the Sydney metropolitan network

---

## System Architecture

![MedStock-AU System Architecture](reports/images/system_architecture.png)

---

## Key Features

| Feature | Technology | Description |
|---------|-----------|-------------|
| Demand forecasting | LSTM + XGBoost + Prophet | 3-model comparison with RMSE, MAE, MAPE evaluation |
| Anomaly detection | Isolation Forest | Per location-medication model, 4.04% anomaly rate |
| Inventory optimisation | PPO (stable-baselines3) | RL agent outperforms rule-based and random policies |
| Report generation | Claude API (claude-sonnet-4-5) | Auto-generated human-readable replenishment reports |
| Conversational interface | FastAPI + Claude API | Natural language stock and forecast queries |
| Dashboard | Streamlit (5 pages) | Korean butter aesthetic, DM Sans, rose-pink palette |

---

## Dataset

MedStock-AU uses a **synthetic dataset** generated to simulate realistic pharmacy demand patterns, in compliance with Australian privacy regulations (Privacy Act 1988, My Health Records Act 2012).

| Parameter | Value |
|-----------|-------|
| Time period | January 2022 – December 2024 |
| Frequency | Daily |
| Locations | 8 (4 hospital + 4 retail pharmacies) |
| Medications | 15 (PBS-listed, commonly dispensed) |
| Total records | 105,216 |
| Region | Sydney, NSW, Australia |

### Locations
**Hospital pharmacies:** RPA Hospital, Westmead Hospital, St Vincent's Hospital, Prince of Wales Hospital

**Retail pharmacies:** Chemist Warehouse Epping, Chemist Warehouse Sydney CBD, Priceline Pharmacy Pitt Street, TerryWhite Chemmart Parramatta

### Medications
Paracetamol, Ibuprofen, Amoxicillin, Metformin, Atorvastatin, Omeprazole, Salbutamol, Pantoprazole, Codeine, Ondansetron, Enoxaparin, Dexamethasone, Sertraline, Rosuvastatin, Cetirizine

---

## Project Structure