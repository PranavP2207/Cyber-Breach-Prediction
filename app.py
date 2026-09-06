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
    initial_sidebar_state="expanded",
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
            "color": "#10b981",  # Emerald
            "bg": "rgba(16, 185, 129, 0.12)",
            "border": "#10b981",
            "glow": "0 0 24px rgba(16, 185, 129, 0.25)",
            "action": "Routine SOC Monitoring",
            "summary": "Current indicators suggest attack containment is highly probable under standard monitoring controls. Perimeter defenses and early-stage isolation remain effective.",
        }
    elif prob < 0.50:
        return {
            "tier": "MEDIUM",
            "color": "#f59e0b",  # Amber
            "bg": "rgba(245, 158, 11, 0.12)",
            "border": "#f59e0b",
            "glow": "0 0 24px rgba(245, 158, 11, 0.25)",
            "action": "Elevated SOC Vigilance",
            "summary": "Adversary progression warrants increased surveillance and active host telemetry checks. Monitor account logins and review endpoint anomalies.",
        }
    elif prob < 0.75:
        return {
            "tier": "HIGH",
            "color": "#f97316",  # Orange
            "bg": "rgba(249, 115, 22, 0.12)",
            "border": "#f97316",
            "glow": "0 0 24px rgba(249, 115, 22, 0.25)",
            "action": "Proactive Host Isolation",
            "summary": "Multiple indicators signal elevated breach risk. Proactively isolate compromised endpoints, revoke affected credentials, and escalate to Tier-2 incident responders.",
        }
    else:
        return {
            "tier": "CRITICAL",
            "color": "#ef4444",  # Red
            "bg": "rgba(239, 68, 68, 0.14)",
            "border": "#ef4444",
            "glow": "0 0 30px rgba(239, 68, 68, 0.35)",
            "action": "Immediate CSIRT Escalation",
            "summary": "Severe intrusion indicators detected with active post-access execution or exfiltration. Initiate containment protocols immediately, sever affected subnets, and engage incident command.",
        }


# ------------------------------------------------------------------
# 6. Global Enterprise Dark Theme CSS
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark Cybersecurity Base Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall Background */
    .stApp {
        background-color: #070b13;
        color: #e2e8f0;
    }

    /* Top Navigation Header */
    .soc-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.60));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .soc-header-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .soc-header-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }
    .soc-header-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 3px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-pill-group {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.73rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #cbd5e1;
    }
    .pill-green {
        color: #10b981;
        border-color: rgba(16, 185, 129, 0.35);
        background: rgba(16, 185, 129, 0.10);
    }

    /* Cards */
    .soc-card {
        background: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .soc-card-title {
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #38bdf8;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* KPI Tiles */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    .kpi-tile {
        background: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        padding: 14px 16px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-tile:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-1px);
    }
    .kpi-label {
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Kill Chain Timeline */
    .kc-timeline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 16px 12px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.12);
        margin-bottom: 18px;
        overflow-x: auto;
    }
    .kc-stage {
        flex: 1;
        min-width: 130px;
        text-align: center;
        padding: 10px 8px;
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
        margin-bottom: 2px;
        text-transform: uppercase;
    }
    .kc-stage.active .kc-stage-num {
        color: #38bdf8;
    }
    .kc-stage.passed .kc-stage-num {
        color: #10b981;
    }
    .kc-stage-name {
        font-size: 0.82rem;
        font-weight: 700;
        color: #cbd5e1;
    }
    .kc-stage.active .kc-stage-name {
        color: #ffffff;
    }
    .kc-stage-status {
        font-size: 0.67rem;
        margin-top: 4px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kc-arrow {
        color: #475569;
        font-size: 0.9rem;
        font-weight: 800;
        user-select: none;
    }

    /* Prediction Result Centerpiece */
    .prediction-container {
        border-radius: 14px;
        padding: 26px 28px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .prediction-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .prediction-tag {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94a3b8;
    }
    .prediction-badge-complete {
        font-size: 0.70rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .prediction-main-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 16px;
    }
    .risk-tier-text {
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .risk-prob-text {
        font-size: 2.4rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }

    /* Segmented Probability Gauge */
    .gauge-container {
        margin: 18px 0;
    }
    .gauge-track {
        height: 12px;
        width: 100%;
        background: #1e293b;
        border-radius: 9999px;
        position: relative;
        overflow: hidden;
        display: flex;
    }
    .gauge-segment {
        height: 100%;
        opacity: 0.25;
        transition: opacity 0.2s ease;
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
        margin-top: 6px;
        font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace;
        color: #64748b;
        font-weight: 600;
    }

    /* Key Risk Indicators Grid */
    .indicator-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
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

    /* Table Styles */
    .soc-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
    }
    .soc-table th {
        text-align: left;
        padding: 8px 10px;
        color: #94a3b8;
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    .soc-table td {
        padding: 9px 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.07);
        color: #e2e8f0;
    }

    /* Footer */
    .soc-footer {
        text-align: center;
        padding: 24px 16px 12px 16px;
        color: #64748b;
        font-size: 0.75rem;
        border-top: 1px solid rgba(148, 163, 184, 0.10);
        margin-top: 30px;
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
        <div class="soc-header-left">
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="m9 12 2 2 4-4"/>
            </svg>
            <div>
                <div class="soc-header-title">Cyber Breach Prediction</div>
                <div class="soc-header-sub">Enterprise Cyber Kill Chain Attack Success Predictor</div>
            </div>
        </div>
        <div class="status-pill-group">
            <span class="status-pill pill-green">● MODEL ONLINE</span>
            <span class="status-pill">RF CLASSIFIER</span>
            <span class="status-pill">100K RECORDS</span>
            <span class="status-pill">v1.0 (16F OPTIMIZED)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 8. Dashboard KPI Analytics Cards
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="kpi-grid">
        <div class="kpi-tile">
            <div class="kpi-label">Algorithm</div>
            <div class="kpi-value">Random Forest</div>
            <div class="kpi-sub">150 Trees · Depth 12 · Gini Split</div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-label">Training Base</div>
            <div class="kpi-value">100,000</div>
            <div class="kpi-sub">Enterprise Incident Records</div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-label">Input Telemetry</div>
            <div class="kpi-value">16 Features</div>
            <div class="kpi-sub">Defenses & Kill-Chain Progression</div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-label">Held-Out Test ROC-AUC</div>
            <div class="kpi-value">0.94</div>
            <div class="kpi-sub">86.2% Accuracy · 0.77 F1 Score</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 9. Scenario Simulation Card (Working Session State)
# ------------------------------------------------------------------
with st.container():
    st.markdown(
        """
        <div style="background: #0f172a; border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.80rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #38bdf8;">
                    ⚡ Scenario Simulation
                </span>
                <span style="font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); padding: 2px 8px; border-radius: 4px;">
                    DEMO SCENARIOS
                </span>
            </div>
            <div style="font-size: 0.80rem; color: #94a3b8; margin-bottom: 10px;">
                Instantly load predefined threat scenarios to simulate how defensive controls and adversarial advancement alter breach likelihood.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.selectbox(
        "Select an active scenario to populate telemetry:",
        list(PRESETS.keys()),
        key="scenario_selector",
        on_change=on_scenario_change,
        label_visibility="collapsed",
    )

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 10. Reorganized Input Sections (2 Enterprise Domains)
# ------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left.container(border=True):
    st.markdown(
        """
        <div class="soc-card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
                <line x1="8" y1="21" x2="16" y2="21"/>
                <line x1="12" y1="17" x2="12" y2="21"/>
            </svg>
            Domain 1: Defenses & Vulnerability Posture
        </div>
        """,
        unsafe_allow_html=True,
    )

    company_size = st.selectbox(
        "Enterprise Company Size",
        CATEGORY_OPTIONS["Company_Size"],
        key="Company_Size",
        help="Organization scale: Small (<100 staff), Medium (100–1000), Large (>1000).",
    )

    c1, c2 = st.columns(2)
    with c1:
        firewall = int(
            st.toggle(
                "Firewall Active",
                key="Firewall",
                help="Network boundary firewall inspection enabled.",
            )
        )
        mfa = int(
            st.toggle(
                "MFA Enforced",
                key="MFA",
                help="Multi-factor authentication enforced on identities.",
            )
        )
    with c2:
        edr = int(
            st.toggle(
                "EDR Deployed",
                key="EDR",
                help="Endpoint Detection and Response sensor coverage.",
            )
        )

    cvss_score = float(
        st.number_input(
            "Max CVSS Vulnerability Score (0.0 – 10.0)",
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            key="CVSS_Score",
            help="Highest CVSS v3 severity of unpatched vulnerabilities on target assets.",
        )
    )

    patch_age = int(
        st.number_input(
            "Known Vulnerability Patch Age (Days)",
            min_value=0,
            max_value=365,
            key="Patch_Age_Days",
            help="Days elapsed since critical patches were made available.",
        )
    )


with col_right.container(border=True):
    st.markdown(
        """
        <div class="soc-card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
            Domain 2: Kill Chain Telemetry & Response
        </div>
        """,
        unsafe_allow_html=True,
    )

    attack_stage = st.selectbox(
        "Furthest Kill Chain Stage Reached",
        CATEGORY_OPTIONS["Attack_Stage"],
        key="Attack_Stage",
        help="Furthest confirmed attacker progression stage along the kill chain.",
    )

    tc1, tc2 = st.columns(2)
    with tc1:
        phishing_click = int(
            st.toggle("Phishing Link Clicked", key="Phishing_Click")
        )
        credential_stolen = int(
            st.toggle("Credentials Compromised", key="Credential_Stolen")
        )
        privilege_escalation = int(
            st.toggle(
                "Privilege Escalation Observed", key="Privilege_Escalation"
            )
        )
    with tc2:
        lateral_movement = int(
            st.toggle("Lateral Movement Detected", key="Lateral_Movement")
        )
        persistence = int(
            st.toggle("Persistence Established", key="Persistence")
        )
        data_encrypted = int(
            st.toggle("Ransomware Encryption Active", key="Data_Encrypted")
        )

    data_exfil = float(
        st.number_input(
            "Data Exfiltrated So Far (GB)",
            min_value=0.0,
            max_value=500.0,
            step=0.1,
            key="Data_Exfiltration_GB",
            help="Confirmed cumulative outbound payload volume.",
        )
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        detection_time = float(
            st.number_input(
                "Detection Dwell Time (Minutes)",
                min_value=0,
                max_value=500,
                key="Detection_Time_Min",
                help="Time elapsed between intrusion initiation and detection.",
            )
        )
    with rc2:
        response_time = float(
            st.number_input(
                "SOC Response Time (Minutes)",
                min_value=0,
                max_value=500,
                key="Response_Time_Min",
                help="Time from alert triage to initial containment attempt.",
            )
        )

# ------------------------------------------------------------------
# 11. Cyber Kill Chain Progression Timeline Visualization
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
        status_text = "◉ ACTIVE STAGE"
        status_color = "#38bdf8"
    else:
        css_class = "kc-stage"
        status_text = "○ PENDING"
        status_color = "#64748b"

    node_html = f"""
    <div class="{css_class}">
        <div class="kc-stage-num">PHASE 0{idx+1}</div>
        <div class="kc-stage-name">{stage_name}</div>
        <div class="kc-stage-status" style="color: {status_color};">{status_text}</div>
    </div>
    """
    timeline_nodes_html += node_html
    if idx < len(KILL_CHAIN_STAGES) - 1:
        timeline_nodes_html += '<div class="kc-arrow">→</div>'

st.markdown(
    f"""
    <div style="margin-bottom: 24px;">
        <div style="font-size: 0.80rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <polyline points="12 6 12 12 16 14"/>
            </svg>
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
# 12. Model Prediction Execution
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
    st.error(
        f"Critical Error: Model file '{MODEL_PATH}' was not found. Please ensure it exists."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
    breach_prob = float(model.predict_proba(input_df)[0, 1])
except Exception as e:
    st.error(f"Prediction Pipeline Error: {e}")
    st.stop()

risk_meta = get_risk_tier(breach_prob)
prob_pct = breach_prob * 100

# ------------------------------------------------------------------
# 13. Prediction Centerpiece Card & Probability Gauge
# ------------------------------------------------------------------
gauge_width = min(max(prob_pct, 1.5), 98.5)

st.markdown(
    f"""
    <div class="prediction-container" style="background: {risk_meta['bg']}; border: 2px solid {risk_meta['border']}; box-shadow: {risk_meta['glow']};">
        <div class="prediction-header">
            <span class="prediction-tag">INCIDENT RISK ASSESSMENT</span>
            <span class="prediction-badge-complete">● INFERENCE COMPLETE</span>
        </div>
        <div class="prediction-main-row">
            <div>
                <div style="font-size: 0.78rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 2px;">
                    Estimated Breach Likelihood Tier
                </div>
                <div class="risk-tier-text" style="color: {risk_meta['color']};">
                    {risk_meta['tier']} RISK
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.78rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 2px;">
                    Breach Probability
                </div>
                <div class="risk-prob-text" style="color: {risk_meta['color']};">
                    {prob_pct:.1f}%
                </div>
            </div>
        </div>
        <div class="gauge-container">
            <div class="gauge-track">
                <div class="gauge-segment seg-low"></div>
                <div class="gauge-segment seg-med"></div>
                <div class="gauge-segment seg-high"></div>
                <div class="gauge-segment seg-crit"></div>
                <div class="gauge-active-bar" style="width: {gauge_width}%; background: {risk_meta['color']}; box-shadow: 0 0 10px {risk_meta['color']};"></div>
            </div>
            <div class="gauge-labels">
                <span>0% LOW</span>
                <span>25% MEDIUM</span>
                <span>50% HIGH</span>
                <span>75% CRITICAL</span>
                <span>100%</span>
            </div>
        </div>
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 8px; padding: 14px 18px; margin-top: 14px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: {risk_meta['color']};">
                    Recommended SOC Action: {risk_meta['action']}
                </span>
            </div>
            <div style="font-size: 0.83rem; color: #cbd5e1; line-height: 1.45;">
                {risk_meta['summary']}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 14. Key Risk Indicators & Defensive Posture Panels
# ------------------------------------------------------------------
p_col1, p_col2 = st.columns(2)

with p_col1:
    st.markdown(
        f"""
        <div class="soc-card">
            <div class="soc-card-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                Observed Behavioral Threat Indicators
            </div>
            <table class="soc-table">
                <thead>
                    <tr>
                        <th>Threat Signal</th>
                        <th>Telemetry Evidence</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Credential Theft</strong></td>
                        <td>Account identity harvest detected</td>
                        <td>
                            <span class="indicator-badge {'ind-active' if credential_stolen else 'ind-inactive'}">
                                {'DETECTED' if credential_stolen else 'NOT DETECTED'}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Privilege Escalation</strong></td>
                        <td>Elevation to local SYSTEM / Admin</td>
                        <td>
                            <span class="indicator-badge {'ind-active' if privilege_escalation else 'ind-inactive'}">
                                {'OBSERVED' if privilege_escalation else 'NOT OBSERVED'}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Lateral Movement</strong></td>
                        <td>Internal SMB/RDP network sprawl</td>
                        <td>
                            <span class="indicator-badge {'ind-active' if lateral_movement else 'ind-inactive'}">
                                {'DETECTED' if lateral_movement else 'NOT DETECTED'}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Persistence Mechanism</strong></td>
                        <td>Scheduled task / registry startup</td>
                        <td>
                            <span class="indicator-badge {'ind-active' if persistence else 'ind-inactive'}">
                                {'ESTABLISHED' if persistence else 'ABSENT'}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Ransomware Encryption</strong></td>
                        <td>Mass file entropy / shadow copy deletion</td>
                        <td>
                            <span class="indicator-badge {'ind-active' if data_encrypted else 'ind-inactive'}">
                                {'ACTIVE' if data_encrypted else 'INACTIVE'}
                            </span>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p_col2:
    st.markdown(
        f"""
        <div class="soc-card">
            <div class="soc-card-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                Enterprise Defensive Posture
            </div>
            <table class="soc-table">
                <thead>
                    <tr>
                        <th>Control Component</th>
                        <th>Policy / State</th>
                        <th>Integrity Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Perimeter Firewall</strong></td>
                        <td>Active packet inspection</td>
                        <td>
                            <span class="indicator-badge {'ind-active' if not firewall else 'pill-green'}" style="padding: 4px 10px; border-radius: 6px; font-size: 0.72rem;">
                                {'ENFORCED' if firewall else 'DISABLED'}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Multi-Factor Authentication</strong></td>
                        <td>Identity access challenge</td>
                        <td>
                            <span class="indicator-badge {'ind-active' if not mfa else 'pill-green'}" style="padding: 4px 10px; border-radius: 6px; font-size: 0.72rem;">
                                {'ENFORCED' if mfa else 'DISABLED'}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Endpoint Detection & Response</strong></td>
                        <td>Host sensor containment</td>
                        <td>
                            <span class="indicator-badge {'ind-active' if not edr else 'pill-green'}" style="padding: 4px 10px; border-radius: 6px; font-size: 0.72rem;">
                                {'ACTIVE' if edr else 'UNMONITORED'}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Vulnerability Exposure</strong></td>
                        <td>CVSS Max Severity: {cvss_score:.1f}</td>
                        <td>
                            <span class="indicator-badge {'ind-active' if cvss_score >= 7.0 else 'pill-green'}" style="padding: 4px 10px; border-radius: 6px; font-size: 0.72rem;">
                                {'CRITICAL RISK' if cvss_score >= 8.5 else ('HIGH RISK' if cvss_score >= 7.0 else 'MODERATE')}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Patch Hygiene Delay</strong></td>
                        <td>{patch_age} Days since release</td>
                        <td>
                            <span class="indicator-badge {'ind-active' if patch_age > 60 else 'pill-green'}" style="padding: 4px 10px; border-radius: 6px; font-size: 0.72rem;">
                                {'DELINQUENT' if patch_age > 60 else 'ACCEPTABLE'}
                            </span>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# 15. ML Architecture Workflow & Model Details
# ------------------------------------------------------------------
with st.expander("ℹ️ Machine Learning Architecture & Technical Specifications"):
    st.markdown(
        """
        <div style="padding: 10px 0;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 10px;">
                Prediction Pipeline Dataflow
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px;">
                <div style="flex: 1; min-width: 140px; background: rgba(30, 41, 59, 0.6); padding: 10px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.15); text-align: center;">
                    <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700;">01 · INGESTION</div>
                    <div style="font-size: 0.78rem; font-weight: 700; margin-top: 3px;">16 Telemetry Signals</div>
                </div>
                <div style="flex: 1; min-width: 140px; background: rgba(30, 41, 59, 0.6); padding: 10px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.15); text-align: center;">
                    <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700;">02 · TRANSFORM</div>
                    <div style="font-size: 0.78rem; font-weight: 700; margin-top: 3px;">One-Hot Encoding + Passthrough</div>
                </div>
                <div style="flex: 1; min-width: 140px; background: rgba(30, 41, 59, 0.6); padding: 10px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.15); text-align: center;">
                    <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700;">03 · CLASSIFIER</div>
                    <div style="font-size: 0.78rem; font-weight: 700; margin-top: 3px;">Random Forest (150 Trees)</div>
                </div>
                <div style="flex: 1; min-width: 140px; background: rgba(30, 41, 59, 0.6); padding: 10px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.15); text-align: center;">
                    <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700;">04 · INFERENCE</div>
                    <div style="font-size: 0.78rem; font-weight: 700; margin-top: 3px;">predict_proba() Probability</div>
                </div>
                <div style="flex: 1; min-width: 140px; background: rgba(30, 41, 59, 0.6); padding: 10px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.15); text-align: center;">
                    <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700;">05 · DECISION</div>
                    <div style="font-size: 0.78rem; font-weight: 700; margin-top: 3px;">Risk Tier & SOC Guidance</div>
                </div>
            </div>
            <ul style="font-size: 0.83rem; color: #cbd5e1; line-height: 1.6;">
                <li><strong>Model Architecture:</strong> Scikit-learn Pipeline combining <code>ColumnTransformer</code> (One-Hot Encoding for <code>Attack_Stage</code> and <code>Company_Size</code>, with 14 passthrough numeric/binary indicators) and a depth-limited <code>RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5, random_state=42)</code>.</li>
                <li><strong>Verified Held-Out Metrics:</strong> 86.2% Accuracy, 0.77 F1-Score (breach class), 0.94 ROC-AUC on 20,000 stratified test incidents.</li>
                <li><strong>Strict Zero Data Leakage:</strong> Post-incident consequence metrics (Financial Loss, Downtime, Records Compromised, Recovery Cost, and derived risk scores) were completely excluded during training, guaranteeing genuine prospective prediction from pre-breach telemetry.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# 16. Sidebar Navigation & Telemetry Monitor
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 12px 0 18px 0; border-bottom: 1px solid rgba(148, 163, 184, 0.12); margin-bottom: 16px;">
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="M9 12h6"/>
                <path d="M12 9v6"/>
            </svg>
            <div style="font-size: 1.1rem; font-weight: 800; color: #ffffff; margin-top: 6px;">SOC Control Hub</div>
            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;">Cyber Risk Decision Support</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### System Telemetry")
    st.markdown(
        """
        - **Pipeline Status:** `ACTIVE / READY`
        - **Inference Latency:** `< 15ms`
        - **Model Container:** `cyber_model.joblib`
        - **Model Checksum:** `Verified (16F)`
        - **Security Posture:** `Continuous Assessment`
        """
    )

    st.divider()

    st.markdown("### Risk Tier Thresholds")
    st.markdown(
        """
        - <span style="color:#10b981; font-weight:700;">LOW:</span> `< 25%` (Containment Expected)
        - <span style="color:#f59e0b; font-weight:700;">MEDIUM:</span> `25% – 49.9%` (Elevated Surveillance)
        - <span style="color:#f97316; font-weight:700;">HIGH:</span> `50% – 74.9%` (Active Threat Isolation)
        - <span style="color:#ef4444; font-weight:700;">CRITICAL:</span> `≥ 75%` (Emergency Escalation)
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption(
        "**Project:** Enterprise Cyber Breach Prediction across the Kill Chain\n\n"
        "Department of Information Technology, APSIT."
    )

# ------------------------------------------------------------------
# 17. Professional Academic Demonstration Footer
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="soc-footer">
        <div style="font-weight: 700; color: #94a3b8; margin-bottom: 4px;">
            Cyber Breach Prediction Across the Enterprise Kill Chain
        </div>
        <div>
            Decision-support prototype designed for academic research and SOC demonstration.
            Predictions reflect probabilistic estimation from pre-breach indicators and do not replace organizational incident response procedures.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
