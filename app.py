"""
Cyber Breach Prediction Across the Enterprise Kill Chain
Streamlit SOC-analyst dashboard prototype.

Loads the trained Random Forest pipeline (cyber_model.joblib) and predicts
the probability that an in-progress intrusion will result in a successful
breach (Attack_Success = 1), given the enterprise's defensive posture and
how far the attacker has progressed through the kill chain.

Run locally with:
    streamlit run app.py
(cyber_model.joblib must be in the same folder.)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Cyber Breach Prediction Dashboard",
    layout="wide",
)

# The 16 features the reduced model expects. The ColumnTransformer inside
# the pipeline selects categorical columns by name; we build the input
# frame in this order for clarity.
FEATURE_COLUMNS = [
    "Attack_Stage", "Company_Size",
    "Lateral_Movement", "Privilege_Escalation", "Persistence",
    "Credential_Stolen", "Data_Exfiltration_GB", "Data_Encrypted",
    "Phishing_Click", "Detection_Time_Min", "Response_Time_Min",
    "Firewall", "MFA", "EDR", "CVSS_Score", "Patch_Age_Days",
]

# Default values for the 16 features
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

# Two demo scenarios, used to pre-fill the form for a quick presentation.
PRESETS = {
    "-- Manual entry --": None,
    "Case A: Contained attack (Reconnaissance, hardened defenses)": {
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
    "Case B: Critical breach (Impact stage, weak defenses)": {
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
}

CATEGORY_OPTIONS = {
    "Attack_Stage": ["Reconnaissance", "Initial Access", "Execution",
                      "Persistence", "Impact"],
    "Company_Size": ["Small", "Medium", "Large"],
}


def on_scenario_change():
    """Callback to update widget states when a preset scenario is chosen."""
    scenario = st.session_state.get("scenario_selector")
    if scenario and scenario in PRESETS and PRESETS[scenario] is not None:
        for k, v in PRESETS[scenario].items():
            st.session_state[k] = v


# Initialize session state with defaults on first run
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource
def load_model(model_mtime: float):
    """Load trained pipeline from disk, cache-busting when file timestamp updates."""
    return joblib.load("model/cyber_model.joblib")


def risk_badge(prob: float):
    """Return (label, color) for a breach probability."""
    if prob < 0.25:
        return "LOW", "#2ecc71"
    elif prob < 0.50:
        return "MEDIUM", "#f1c40f"
    elif prob < 0.75:
        return "HIGH", "#e67e22"
    else:
        return "CRITICAL", "#e74c3c"


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("Cyber Breach Prediction Across the Enterprise Kill Chain")
st.caption(
    "Random Forest classifier trained on 100,000 enterprise incident records. "
    "Predicts the probability that an *in-progress* intrusion results in a "
    "successful breach, using only information available before a breach outcome is known."
)

with st.expander("About this model", expanded=False):
    st.markdown(
        """
        - **Model:** Random Forest (150 trees, max depth 12), scikit-learn pipeline
          with one-hot encoding for 2 categorical fields and 14 numeric/binary inputs
          (16 features total).
        - **Test-set performance:** 86.2% accuracy · 0.77 F1-score · 0.94 ROC-AUC
        - **Zero data leakage:** columns only known *after* a breach is confirmed
          (financial loss, downtime, records compromised, recovery cost, and the
          risk/severity scores derived from them) were removed before training,
          so the model genuinely predicts forward from pre-breach signals.
        """
    )

st.selectbox(
    "Quick scenario (optional)",
    list(PRESETS.keys()),
    key="scenario_selector",
    on_change=on_scenario_change,
)
st.divider()

col1, col2 = st.columns(2)

# ------------------------------------------------------------------
# Column 1: Defenses & Vulnerability
# ------------------------------------------------------------------
with col1:
    st.subheader("Defenses & Vulnerability")
    company_size = st.selectbox("Company Size", CATEGORY_OPTIONS["Company_Size"], key="Company_Size")
    firewall = int(st.toggle("Firewall active", key="Firewall"))
    mfa = int(st.toggle("Multi-Factor Authentication enforced", key="MFA"))
    edr = int(st.toggle("EDR deployed", key="EDR"))
    cvss_score = float(st.number_input("Max CVSS Score of open vulns (0-10)", min_value=0.0, max_value=10.0, step=0.1, key="CVSS_Score"))
    patch_age = int(st.number_input("Patch Age (days)", min_value=0, max_value=365, key="Patch_Age_Days"))

# ------------------------------------------------------------------
# Column 2: Kill Chain Telemetry & Response
# ------------------------------------------------------------------
with col2:
    st.subheader("Kill Chain Telemetry & Response")
    attack_stage = st.selectbox("Furthest Kill Chain Stage Reached", CATEGORY_OPTIONS["Attack_Stage"], key="Attack_Stage")
    phishing_click = int(st.toggle("Phishing link clicked", key="Phishing_Click"))
    credential_stolen = int(st.toggle("Credentials stolen", key="Credential_Stolen"))
    privilege_escalation = int(st.toggle("Privilege escalation observed", key="Privilege_Escalation"))
    lateral_movement = int(st.toggle("Lateral movement observed", key="Lateral_Movement"))
    persistence = int(st.toggle("Persistence mechanism established", key="Persistence"))
    data_encrypted = int(st.toggle("Data encrypted by attacker (ransomware)", key="Data_Encrypted"))
    data_exfil = float(st.number_input("Data Exfiltrated so far (GB)", min_value=0.0, max_value=500.0, step=0.1, key="Data_Exfiltration_GB"))
    detection_time = float(st.number_input("Detection Time (minutes)", min_value=0, max_value=500, key="Detection_Time_Min"))
    response_time = float(st.number_input("Response Time (minutes)", min_value=0, max_value=500, key="Response_Time_Min"))

st.divider()

# ------------------------------------------------------------------
# Build the input row and predict
# ------------------------------------------------------------------
row = {
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

input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

MODEL_PATH = "model/cyber_model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"{MODEL_PATH} not found. Ensure the trained model artifact is present."
    )
    st.stop()

try:
    model = load_model(os.path.getmtime(MODEL_PATH))
    breach_prob = float(model.predict_proba(input_df)[0, 1])
except Exception as e:
    st.error(f"Error making prediction: {e}")
    st.stop()

label, color = risk_badge(breach_prob)

st.subheader("Prediction")
res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 1.5rem; border-radius: 12px;
                    background-color:{color}22; border: 2px solid {color};">
            <div style="font-size: 0.9rem; color: #666;">BREACH RISK</div>
            <div style="font-size: 2.2rem; font-weight: 700; color:{color};">{label}</div>
            <div style="font-size: 1.6rem; font-weight: 600; margin-top: 0.3rem;">
                {breach_prob*100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with res_col2:
    st.progress(min(max(breach_prob, 0.0), 1.0))
    st.markdown(
        f"Estimated probability the intrusion described above **results in a "
        f"successful breach** rather than being contained or mitigated: **{breach_prob*100:.1f}%**."
    )
    if label in ("HIGH", "CRITICAL"):
        st.warning(
            "Recommend immediate escalation: isolate affected hosts, force "
            "credential resets, and engage incident response."
        )
    else:
        st.success("Current indicators suggest containment is likely with standard monitoring.")

st.caption(
    "This is a decision-support prototype for academic demonstration and does "
    "not replace analyst judgment or an organization's incident response process."
)
