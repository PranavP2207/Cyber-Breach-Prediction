"""
Cyber Breach Prediction Across the Enterprise Kill Chain
Enterprise SOC Dashboard & Decision-Support System

Machine learning classification pipeline predicting the probability that an
in-progress enterprise intrusion results in a confirmed breach (Attack_Success = 1)
using pre-breach telemetry, kill-chain progression, and defensive posture.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Cyber Breach Prediction | Enterprise SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# 2. Model Feature Schema & Defaults (16 Features)
# ------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Attack_Stage", "Company_Size",
    "Lateral_Movement", "Privilege_Escalation", "Persistence",
    "Credential_Stolen", "Data_Exfiltration_GB", "Data_Encrypted",
    "Phishing_Click", "Detection_Time_Min", "Response_Time_Min",
    "Firewall", "MFA", "EDR", "CVSS_Score", "Patch_Age_Days",
]

DEFAULTS = {
    "Attack_Stage": "Execution",
    "Company_Size": "Medium",
    "Lateral_Movement": False,
    "Privilege_Escalation": False,
    "Persistence": False,
    "Credential_Stolen": False,
    "Data_Exfiltration_GB": 0.0,
    "Data_Encrypted": False,
    "Phishing_Click": False,
    "Detection_Time_Min": 100,
    "Response_Time_Min": 40,
    "Firewall": True,
    "MFA": True,
    "EDR": True,
    "CVSS_Score": 5.0,
    "Patch_Age_Days": 45,
}

PRESETS = {
    "-- Manual Entry / Custom Telemetry --": None,
    "Case A: Contained Intrusion (Reconnaissance, Hardened Posture)": {
        "Attack_Stage": "Reconnaissance",
        "Company_Size": "Large",
        "Lateral_Movement": False,
        "Privilege_Escalation": False,
        "Persistence": False,
        "Credential_Stolen": False,
        "Data_Exfiltration_GB": 0.0,
        "Data_Encrypted": False,
        "Phishing_Click": False,
        "Detection_Time_Min": 25,
        "Response_Time_Min": 10,
        "Firewall": True,
        "MFA": True,
        "EDR": True,
        "CVSS_Score": 3.0,
        "Patch_Age_Days": 10,
    },
    "Case B: Critical Active Breach (Impact Stage, Weak Defenses)": {
        "Attack_Stage": "Impact",
        "Company_Size": "Small",
        "Lateral_Movement": True,
        "Privilege_Escalation": True,
        "Persistence": True,
        "Credential_Stolen": True,
        "Data_Exfiltration_GB": 85.0,
        "Data_Encrypted": True,
        "Phishing_Click": True,
        "Detection_Time_Min": 210,
        "Response_Time_Min": 150,
        "Firewall": False,
        "MFA": False,
        "EDR": False,
        "CVSS_Score": 9.4,
        "Patch_Age_Days": 160,
    },
    "Case C: Initial Access via Phishing (Execution, Partial Defenses)": {
        "Attack_Stage": "Initial Access",
        "Company_Size": "Medium",
        "Lateral_Movement": False,
        "Privilege_Escalation": False,
        "Persistence": False,
        "Credential_Stolen": True,
        "Data_Exfiltration_GB": 0.5,
        "Data_Encrypted": False,
        "Phishing_Click": True,
        "Detection_Time_Min": 65,
        "Response_Time_Min": 30,
        "Firewall": True,
        "MFA": False,
        "EDR": True,
        "CVSS_Score": 6.8,
        "Patch_Age_Days": 50,
    },
    "Case D: Lateral Movement & Ransomware (Persistence, Weak Controls)": {
        "Attack_Stage": "Persistence",
        "Company_Size": "Medium",
        "Lateral_Movement": True,
        "Privilege_Escalation": True,
        "Persistence": True,
        "Credential_Stolen": True,
        "Data_Exfiltration_GB": 18.0,
        "Data_Encrypted": True,
        "Phishing_Click": True,
        "Detection_Time_Min": 140,
        "Response_Time_Min": 90,
        "Firewall": True,
        "MFA": False,
        "EDR": False,
        "CVSS_Score": 8.2,
        "Patch_Age_Days": 95,
    },
}

CATEGORY_OPTIONS = {
    "Attack_Stage": [
        "Reconnaissance",
        "Initial Access",
        "Execution",
        "Persistence",
        "Impact",
    ],
    "Company_Size": ["Small", "Medium", "Large"],
}

KILL_CHAIN_STAGES = [
    "Reconnaissance",
    "Initial Access",
    "Execution",
    "Persistence",
    "Impact",
]


# ------------------------------------------------------------------
# 3. State Management & Scenario Handler
# ------------------------------------------------------------------
def on_scenario_change():
    """Callback to update widget states when a preset scenario is chosen."""
    scenario = st.session_state.get("scenario_selector")
    if scenario and scenario in PRESETS and PRESETS[scenario] is not None:
        for k, v in PRESETS[scenario].items():
            st.session_state[k] = v


# Initialize session state with defaults
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ------------------------------------------------------------------
# 4. Model Loading with Cache Invalidation
# ------------------------------------------------------------------
@st.cache_resource
def load_model(_model_path: str = "model/cyber_model.joblib"):
    """Load trained pipeline from disk with Streamlit's resource caching."""
    return joblib.load(_model_path)


# ------------------------------------------------------------------
# 5. Risk Assessment Helpers
# ------------------------------------------------------------------
def get_risk_tier(prob: float):
    """Return risk tier metadata (label, hex color, description)."""
    if prob < 0.25:
        return {
            "tier": "LOW",
            "color": "#10b981",
            "bg": "rgba(16, 185, 129, 0.12)",
            "border": "#10b981",
            "glow": "0 0 24px rgba(16, 185, 129, 0.25)",
            "action": "Routine SOC Monitoring",
            "summary": "Current indicators suggest attack containment is highly probable under standard monitoring controls.",
        }
    elif prob < 0.50:
        return {
            "tier": "MEDIUM",
            "color": "#f59e0b",
            "bg": "rgba(245, 158, 11, 0.12)",
            "border": "#f59e0b",
            "glow": "0 0 24px rgba(245, 158, 11, 0.25)",
            "action": "Elevated SOC Vigilance",
            "summary": "Adversary progression warrants increased surveillance and active host telemetry checks.",
        }
    elif prob < 0.75:
        return {
            "tier": "HIGH",
            "color": "#f97316",
            "bg": "rgba(249, 115, 22, 0.12)",
            "border": "#f97316",
            "glow": "0 0 24px rgba(249, 115, 22, 0.25)",
            "action": "Proactive Host Isolation",
            "summary": "Multiple indicators signal elevated breach risk. Proactively isolate compromised endpoints.",
        }
    else:
        return {
            "tier": "CRITICAL",
            "color": "#ef4444",
            "bg": "rgba(239, 68, 68, 0.14)",
            "border": "#ef4444",
            "glow": "0 0 30px rgba(239, 68, 68, 0.35)",
            "action": "Immediate CSIRT Escalation",
            "summary": "Severe intrusion indicators detected. Initiate containment protocols immediately.",
        }


# ------------------------------------------------------------------
# 6. Global Enterprise Dark Theme CSS
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #070b13;
        color: #e2e8f0;
    }

    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Header */
    .soc-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.60));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .soc-header-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .soc-header-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 4px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.73rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #cbd5e1;
    }
    .pill-green {
        color: #10b981;
        border-color: rgba(16, 185, 129, 0.35);
        background: rgba(16, 185, 129, 0.10);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 24px;
        background: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        font-weight: 600;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid #38bdf8;
        color: #38bdf8;
    }

    /* Cards */
    .soc-card {
        background: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 20px;
    }
    .soc-card-title {
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #38bdf8;
        margin-bottom: 16px;
    }

    /* KPI Grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin-bottom: 28px;
    }
    .kpi-tile {
        background: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        padding: 18px 20px;
        transition: all 0.2s ease;
    }
    .kpi-tile:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .kpi-sub {
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 6px;
    }

    /* Kill Chain Timeline */
    .kc-timeline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 18px 14px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.12);
        margin-bottom: 24px;
        overflow-x: auto;
    }
    .kc-stage {
        flex: 1;
        min-width: 130px;
        text-align: center;
        padding: 12px 10px;
        border-radius: 8px;
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.2s ease;
    }
    .kc-stage.active {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid #38bdf8;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.25);
    }
    .kc-stage.passed {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .kc-stage-num {
        font-size: 0.65rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #64748b;
        margin-bottom: 4px;
        text-transform: uppercase;
    }
    .kc-stage.active .kc-stage-num { color: #38bdf8; }
    .kc-stage.passed .kc-stage-num { color: #10b981; }
    .kc-stage-name {
        font-size: 0.84rem;
        font-weight: 700;
        color: #cbd5e1;
    }
    .kc-stage.active .kc-stage-name { color: #ffffff; }
    .kc-stage-status {
        font-size: 0.68rem;
        margin-top: 6px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kc-arrow {
        color: #475569;
        font-size: 1rem;
        font-weight: 800;
    }

    /* Prediction Result */
    .prediction-container {
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .prediction-main-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 20px;
        margin-bottom: 20px;
    }
    .risk-tier-text {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .risk-prob-text {
        font-size: 2.6rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Gauge */
    .gauge-track {
        height: 14px;
        width: 100%;
        background: #1e293b;
        border-radius: 9999px;
        position: relative;
        overflow: hidden;
        display: flex;
        margin: 20px 0;
    }
    .gauge-segment {
        height: 100%;
        opacity: 0.25;
    }
    .gauge-segment.seg-low { width: 25%; background: #10b981; }
    .gauge-segment.seg-med { width: 25%; background: #f59e0b; }
    .gauge-segment.seg-high { width: 25%; background: #f97316; }
    .gauge-segment.seg-crit { width: 25%; background: #ef4444; }
    .gauge-active-bar {
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        border-radius: 9999px;
        transition: width 0.4s ease;
    }
    .gauge-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 8px;
        font-size: 0.70rem;
        font-family: 'JetBrains Mono', monospace;
        color: #64748b;
        font-weight: 600;
    }

    /* Tables */
    .soc-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
    }
    .soc-table th {
        text-align: left;
        padding: 10px 12px;
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    .soc-table td {
        padding: 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.07);
        color: #e2e8f0;
    }

    /* Badges */
    .indicator-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .ind-active {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .ind-inactive {
        background: rgba(148, 163, 184, 0.08);
        color: #64748b;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .ind-good {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    /* Feature importance chart */
    .feature-bar-container {
        margin: 8px 0;
    }
    .feature-bar-label {
        font-size: 0.82rem;
        color: #cbd5e1;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
    }
    .feature-bar-bg {
        height: 8px;
        background: #1e293b;
        border-radius: 4px;
        overflow: hidden;
    }
    .feature-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #38bdf8, #0ea5e9);
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    /* Footer */
    .soc-footer {
        text-align: center;
        padding: 28px 16px;
        color: #64748b;
        font-size: 0.78rem;
        border-top: 1px solid rgba(148, 163, 184, 0.10);
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 7. Top Navigation & Status Header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="soc-header">
        <div>
            <div class="soc-header-title">🛡️ Cyber Breach Prediction</div>
            <div class="soc-header-sub">Enterprise Kill Chain Attack Success Predictor</div>
        </div>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <span class="status-pill pill-green">● MODEL ONLINE</span>
            <span class="status-pill">RF CLASSIFIER</span>
            <span class="status-pill">v1.0 (16F)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 8. Main Tabs Structure
# ------------------------------------------------------------------
tab_predictor, tab_info = st.tabs(["🎯 Predictor", "📊 Model Info"])

# ==================================================================
# TAB 1: PREDICTOR (Interactive Model)
# ==================================================================
with tab_predictor:

    # Scenario Selection
    st.markdown("### ⚡ Scenario Simulation")
    st.markdown("Load predefined threat scenarios or enter custom telemetry below.")

    st.selectbox(
        "Select a scenario to populate telemetry:",
        list(PRESETS.keys()),
        key="scenario_selector",
        on_change=on_scenario_change,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Two-column layout for inputs
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🛡️ Defenses & Vulnerability Posture")

        company_size = st.selectbox(
            "Enterprise Company Size",
            CATEGORY_OPTIONS["Company_Size"],
            key="Company_Size",
            help="Organization scale: Small (<100 staff), Medium (100–1000), Large (>1000)",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            firewall = int(st.toggle("Firewall", key="Firewall", help="Network boundary firewall enabled"))
        with c2:
            mfa = int(st.toggle("MFA", key="MFA", help="Multi-factor authentication enforced"))
        with c3:
            edr = int(st.toggle("EDR", key="EDR", help="Endpoint Detection and Response deployed"))

        cvss_score = float(st.number_input(
            "Max CVSS Score (0.0–10.0)",
            min_value=0.0, max_value=10.0, step=0.1,
            key="CVSS_Score",
            help="Highest CVSS v3 severity of unpatched vulnerabilities",
        ))

        patch_age = int(st.number_input(
            "Patch Age (Days)",
            min_value=0, max_value=365,
            key="Patch_Age_Days",
            help="Days since critical patches were released",
        ))

    with col_right:
        st.markdown("#### 🔥 Kill Chain Telemetry & Response")

        attack_stage = st.selectbox(
            "Furthest Kill Chain Stage Reached",
            CATEGORY_OPTIONS["Attack_Stage"],
            key="Attack_Stage",
            help="Furthest confirmed attacker progression stage",
        )

        # Threat indicators in 2 columns
        tc1, tc2 = st.columns(2)
        with tc1:
            phishing_click = int(st.toggle("Phishing Clicked", key="Phishing_Click"))
            credential_stolen = int(st.toggle("Credentials Stolen", key="Credential_Stolen"))
            privilege_escalation = int(st.toggle("Privilege Escalation", key="Privilege_Escalation"))
        with tc2:
            lateral_movement = int(st.toggle("Lateral Movement", key="Lateral_Movement"))
            persistence = int(st.toggle("Persistence", key="Persistence"))
            data_encrypted = int(st.toggle("Data Encrypted", key="Data_Encrypted"))

        data_exfil = float(st.number_input(
            "Data Exfiltrated (GB)",
            min_value=0.0, max_value=500.0, step=0.1,
            key="Data_Exfiltration_GB",
            help="Confirmed cumulative outbound payload volume",
        ))

        rc1, rc2 = st.columns(2)
        with rc1:
            detection_time = float(st.number_input(
                "Detection Time (Min)",
                min_value=0, max_value=500,
                key="Detection_Time_Min",
                help="Time from intrusion initiation to detection",
            ))
        with rc2:
            response_time = float(st.number_input(
                "Response Time (Min)",
                min_value=0, max_value=500,
                key="Response_Time_Min",
                help="Time from alert to containment",
            ))

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Kill Chain Progression Timeline
    # ------------------------------------------------------------------
    current_stage_idx = (
        KILL_CHAIN_STAGES.index(attack_stage)
        if attack_stage in KILL_CHAIN_STAGES
        else 0
    )

    timeline_nodes_html = ""
    for idx, stage_name in enumerate(KILL_CHAIN_STAGES):
        if idx < current_stage_idx:
            css_class = "kc-stage passed"
            status_text = "✓ COMPLETED"
            status_color = "#10b981"
        elif idx == current_stage_idx:
            css_class = "kc-stage active"
            status_text = "◉ ACTIVE"
            status_color = "#38bdf8"
        else:
            css_class = "kc-stage"
            status_text = "○ PENDING"
            status_color = "#64748b"

        node_html = f'<div class="{css_class}"><div class="kc-stage-num">PHASE 0{idx+1}</div><div class="kc-stage-name">{stage_name}</div><div class="kc-stage-status" style="color: {status_color};">{status_text}</div></div>'
        timeline_nodes_html += node_html
        if idx < len(KILL_CHAIN_STAGES) - 1:
            timeline_nodes_html += '<div class="kc-arrow">→</div>'

    st.markdown(
        f"""
<div style="margin-bottom: 24px;">
    <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 10px;">
        Active Kill Chain Progression Timeline
    </div>
    <div class="kc-timeline">
{timeline_nodes_html}
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Model Prediction Execution
    # ------------------------------------------------------------------
    row_data = {
        "Attack_Stage": attack_stage,
        "Company_Size": company_size,
        "Lateral_Movement": lateral_movement,
        "Privilege_Escalation": privilege_escalation,
        "Persistence": persistence,
        "Credential_Stolen": credential_stolen,
        "Data_Exfiltration_GB": data_exfil,
        "Data_Encrypted": data_encrypted,
        "Phishing_Click": phishing_click,
        "Detection_Time_Min": detection_time,
        "Response_Time_Min": response_time,
        "Firewall": firewall,
        "MFA": mfa,
        "EDR": edr,
        "CVSS_Score": cvss_score,
        "Patch_Age_Days": patch_age,
    }

    input_df = pd.DataFrame([row_data], columns=FEATURE_COLUMNS)

    MODEL_PATH = "model/cyber_model.joblib"
    if not os.path.exists(MODEL_PATH):
        st.error(f"Critical Error: Model file '{MODEL_PATH}' not found.")
        st.stop()

    try:
        model = load_model(MODEL_PATH)
        breach_prob = float(model.predict_proba(input_df)[0, 1])
    except Exception as e:
        st.error(f"Prediction Pipeline Error: {e}")
        st.stop()

    risk_meta = get_risk_tier(breach_prob)
    prob_pct = breach_prob * 100
    gauge_width = min(max(prob_pct, 1.5), 98.5)

    # ------------------------------------------------------------------
    # Prediction Result Display
    # ------------------------------------------------------------------
    st.markdown(
        f"""
<div class="prediction-container" style="background: {risk_meta['bg']}; border: 2px solid {risk_meta['border']}; box-shadow: {risk_meta['glow']};">
    <div class="prediction-main-row">
        <div>
            <div style="font-size: 0.82rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; margin-bottom: 6px;">
                Risk Assessment
            </div>
            <div class="risk-tier-text" style="color: {risk_meta['color']};">
                {risk_meta['tier']} RISK
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.82rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; margin-bottom: 6px;">
                Breach Probability
            </div>
            <div class="risk-prob-text" style="color: {risk_meta['color']};">
                {prob_pct:.1f}%
            </div>
        </div>
    </div>
    <div class="gauge-track">
        <div class="gauge-segment seg-low"></div>
        <div class="gauge-segment seg-med"></div>
        <div class="gauge-segment seg-high"></div>
        <div class="gauge-segment seg-crit"></div>
        <div class="gauge-active-bar" style="width: {gauge_width}%; background: {risk_meta['color']}; box-shadow: 0 0 12px {risk_meta['color']};"></div>
    </div>
    <div class="gauge-labels">
        <span>0% LOW</span>
        <span>25% MEDIUM</span>
        <span>50% HIGH</span>
        <span>75% CRITICAL</span>
        <span>100%</span>
    </div>
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 8px; padding: 16px 20px; margin-top: 16px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: {risk_meta['color']};">
                Recommended Action: {risk_meta['action']}
            </span>
        </div>
        <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;">
            {risk_meta['summary']}
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Key Risk Indicators & Defensive Posture Tables
    # ------------------------------------------------------------------
    col_ind1, col_ind2 = st.columns(2)

    with col_ind1:
        st.markdown(
            f"""
<div class="soc-card">
    <div class="soc-card-title">🔥 Behavioral Threat Indicators</div>
    <table class="soc-table">
        <thead><tr><th>Indicator</th><th>Status</th></tr></thead>
        <tbody>
            <tr><td><strong>Credential Theft</strong></td><td><span class="indicator-badge {'ind-active' if credential_stolen else 'ind-inactive'}">{'DETECTED' if credential_stolen else 'NOT DETECTED'}</span></td></tr>
            <tr><td><strong>Privilege Escalation</strong></td><td><span class="indicator-badge {'ind-active' if privilege_escalation else 'ind-inactive'}">{'OBSERVED' if privilege_escalation else 'NOT OBSERVED'}</span></td></tr>
            <tr><td><strong>Lateral Movement</strong></td><td><span class="indicator-badge {'ind-active' if lateral_movement else 'ind-inactive'}">{'DETECTED' if lateral_movement else 'NOT DETECTED'}</span></td></tr>
            <tr><td><strong>Persistence</strong></td><td><span class="indicator-badge {'ind-active' if persistence else 'ind-inactive'}">{'ESTABLISHED' if persistence else 'ABSENT'}</span></td></tr>
            <tr><td><strong>Ransomware</strong></td><td><span class="indicator-badge {'ind-active' if data_encrypted else 'ind-inactive'}">{'ACTIVE' if data_encrypted else 'INACTIVE'}</span></td></tr>
        </tbody>
    </table>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col_ind2:
        st.markdown(
            f"""
<div class="soc-card">
    <div class="soc-card-title">🛡️ Defensive Posture</div>
    <table class="soc-table">
        <thead><tr><th>Control</th><th>Status</th></tr></thead>
        <tbody>
            <tr><td><strong>Firewall</strong></td><td><span class="indicator-badge {'ind-good' if firewall else 'ind-active'}">{'ENFORCED' if firewall else 'DISABLED'}</span></td></tr>
            <tr><td><strong>MFA</strong></td><td><span class="indicator-badge {'ind-good' if mfa else 'ind-active'}">{'ENFORCED' if mfa else 'DISABLED'}</span></td></tr>
            <tr><td><strong>EDR</strong></td><td><span class="indicator-badge {'ind-good' if edr else 'ind-active'}">{'ACTIVE' if edr else 'UNMONITORED'}</span></td></tr>
            <tr><td><strong>CVSS Exposure</strong></td><td><span class="indicator-badge {'ind-active' if cvss_score >= 7.0 else 'ind-good'}">{'CRITICAL' if cvss_score >= 8.5 else ('HIGH' if cvss_score >= 7.0 else 'MODERATE')}</span></td></tr>
            <tr><td><strong>Patch Hygiene</strong></td><td><span class="indicator-badge {'ind-active' if patch_age > 60 else 'ind-good'}">{'DELINQUENT' if patch_age > 60 else 'ACCEPTABLE'}</span></td></tr>
        </tbody>
    </table>
</div>
            """,
            unsafe_allow_html=True,
        )

# ==================================================================
# TAB 2: MODEL INFO (Static Information)
# ==================================================================
with tab_info:

    # KPI Cards
    st.markdown(
        """
<div class="kpi-grid">
    <div class="kpi-tile">
        <div class="kpi-label">Algorithm</div>
        <div class="kpi-value">Random Forest</div>
        <div class="kpi-sub">150 Trees · Depth 12 · Gini Split</div>
    </div>
    <div class="kpi-tile">
        <div class="kpi-label">Training Records</div>
        <div class="kpi-value">100,000</div>
        <div class="kpi-sub">Enterprise Incident Records</div>
    </div>
    <div class="kpi-tile">
        <div class="kpi-label">Input Features</div>
        <div class="kpi-value">16</div>
        <div class="kpi-sub">Defenses & Kill-Chain Progression</div>
    </div>
    <div class="kpi-tile">
        <div class="kpi-label">ROC-AUC Score</div>
        <div class="kpi-value">0.94</div>
        <div class="kpi-sub">86.2% Accuracy · 0.77 F1 Score</div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Model Architecture
    col_arch1, col_arch2 = st.columns(2)

    with col_arch1:
        st.markdown("#### 🏗️ Model Architecture")
        st.markdown(
            """
<div class="soc-card">
    <div class="soc-card-title">Prediction Pipeline</div>
    <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.7;">
        <p><strong>Preprocessing:</strong> <code>ColumnTransformer</code> with One-Hot Encoding for categorical features (<code>Attack_Stage</code>, <code>Company_Size</code>) and passthrough for 14 numeric/binary indicators.</p>
        <p><strong>Classifier:</strong> <code>RandomForestClassifier</code> with 150 trees, max depth 12, min samples per leaf 5, balanced class weights.</p>
        <p><strong>Output:</strong> Probability estimate via <code>predict_proba()</code> for the breach class.</p>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📊 Data Pipeline")
        st.markdown(
            """
<div class="soc-card">
    <div class="soc-card-title">Processing Steps</div>
    <ul style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.8;">
        <li><strong>01 · INGEST:</strong> 100,500 raw incident records (Jan 2023 – Nov 2025)</li>
        <li><strong>02 · CLEAN:</strong> Drop 500 duplicates, remove leakage columns</li>
        <li><strong>03 · REDUCE:</strong> Feature importance analysis → 16 high-value features</li>
        <li><strong>04 · SPLIT:</strong> 80,000 train / 20,000 test (stratified)</li>
        <li><strong>05 · TRAIN:</strong> Random Forest with cross-validation</li>
        <li><strong>06 · EVALUATE:</strong> ROC-AUC, F1, confusion matrix</li>
    </ul>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col_arch2:
        st.markdown("#### 🎯 Feature Importance")
        st.markdown(
            """
<div class="soc-card">
    <div class="soc-card-title">Top Predictive Features</div>
    <div class="feature-bar-container">
        <div class="feature-bar-label"><span>Attack_Stage</span><span>0.142</span></div>
        <div class="feature-bar-bg"><div class="feature-bar-fill" style="width: 100%;"></div></div>
    </div>
    <div class="feature-bar-container">
        <div class="feature-bar-label"><span>Lateral_Movement</span><span>0.098</span></div>
        <div class="feature-bar-bg"><div class="feature-bar-fill" style="width: 69%;"></div></div>
    </div>
    <div class="feature-bar-container">
        <div class="feature-bar-label"><span>Privilege_Escalation</span><span>0.087</span></div>
        <div class="feature-bar-bg"><div class="feature-bar-fill" style="width: 61%;"></div></div>
    </div>
    <div class="feature-bar-container">
        <div class="feature-bar-label"><span>Persistence</span><span>0.079</span></div>
        <div class="feature-bar-bg"><div class="feature-bar-fill" style="width: 56%;"></div></div>
    </div>
    <div class="feature-bar-container">
        <div class="feature-bar-label"><span>Detection_Time_Min</span><span>0.072</span></div>
        <div class="feature-bar-bg"><div class="feature-bar-fill" style="width: 51%;"></div></div>
    </div>
    <div class="feature-bar-container">
        <div class="feature-bar-label"><span>Credential_Stolen</span><span>0.068</span></div>
        <div class="feature-bar-bg"><div class="feature-bar-fill" style="width: 48%;"></div></div>
    </div>
    <div class="feature-bar-container">
        <div class="feature-bar-label"><span>MFA</span><span>0.063</span></div>
        <div class="feature-bar-bg"><div class="feature-bar-fill" style="width: 44%;"></div></div>
    </div>
    <div class="feature-bar-container">
        <div class="feature-bar-label"><span>CVSS_Score</span><span>0.059</span></div>
        <div class="feature-bar-bg"><div class="feature-bar-fill" style="width: 42%;"></div></div>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ⚠️ Data Leakage Prevention")
        st.markdown(
            """
<div class="soc-card">
    <div class="soc-card-title">Excluded Columns</div>
    <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 12px;">
        The following post-incident columns were <strong>excluded</strong> during training to ensure genuine prospective prediction:
    </p>
    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
        <span class="indicator-badge ind-active">Recovery_Cost_USD</span>
        <span class="indicator-badge ind-active">Records_Compromised</span>
        <span class="indicator-badge ind-active">Downtime_Hours</span>
        <span class="indicator-badge ind-active">Financial_Loss_USD</span>
        <span class="indicator-badge ind-active">Cyber_Risk_Score</span>
        <span class="indicator-badge ind-active">Risk_Level</span>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Tech Stack
    st.markdown("#### 🛠️ Technology Stack")
    st.markdown(
        """
<div class="soc-card">
    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        <span class="indicator-badge ind-good">Python 3.11</span>
        <span class="indicator-badge ind-good">pandas</span>
        <span class="indicator-badge ind-good">NumPy</span>
        <span class="indicator-badge ind-good">scikit-learn</span>
        <span class="indicator-badge ind-good">Streamlit</span>
        <span class="indicator-badge ind-good">joblib</span>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="soc-footer">
        <div style="font-weight: 700; color: #94a3b8; margin-bottom: 6px;">
            Cyber Breach Prediction Across the Enterprise Kill Chain
        </div>
        <div>APSIT · Department of Information Technology · Academic Research Project</div>
    </div>
    """,
    unsafe_allow_html=True,
)
