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

# Two demo scenarios, used to pre-fill the form for a quick presentation.
PRESETS = {
    "-- Manual entry --": None,
    "Case A: Contained attack (Reconnaissance, hardened defenses)": dict(
        Attack_Stage="Reconnaissance", Company_Size="Large",
        Lateral_Movement=0, Privilege_Escalation=0, Persistence=0,
        Credential_Stolen=0, Data_Exfiltration_GB=0.0, Data_Encrypted=0,
        Phishing_Click=0, Detection_Time_Min=25, Response_Time_Min=10,
        Firewall=1, MFA=1, EDR=1, CVSS_Score=3.0, Patch_Age_Days=10,
    ),
    "Case B: Critical breach (Impact stage, weak defenses)": dict(
        Attack_Stage="Impact", Company_Size="Small",
        Lateral_Movement=1, Privilege_Escalation=1, Persistence=1,
        Credential_Stolen=1, Data_Exfiltration_GB=85.0, Data_Encrypted=1,
        Phishing_Click=1, Detection_Time_Min=210, Response_Time_Min=150,
        Firewall=0, MFA=0, EDR=0, CVSS_Score=9.4, Patch_Age_Days=160,
    ),
}

CATEGORY_OPTIONS = {
    "Attack_Stage": ["Reconnaissance", "Initial Access", "Execution",
                      "Persistence", "Impact"],
    "Company_Size": ["Small", "Medium", "Large"],
}


@st.cache_resource
def load_model():
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


def selectbox_from_preset(label, key, preset, default_index=0):
    options = CATEGORY_OPTIONS[key]
    index = options.index(preset[key]) if preset and key in preset else default_index
    return st.selectbox(label, options, index=index, key=key)


def toggle_from_preset(label, key, preset, default=False):
    value = bool(preset[key]) if preset and key in preset else default
    return int(st.toggle(label, value=value, key=key))


def number_from_preset(label, key, preset, default, **kwargs):
    value = preset[key] if preset and key in preset else default
    return st.number_input(label, value=float(value) if isinstance(default, float) else int(value), key=key, **kwargs)


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

scenario = st.selectbox("Quick scenario (optional)", list(PRESETS.keys()))
preset = PRESETS[scenario]
st.divider()

col1, col2 = st.columns(2)

# ------------------------------------------------------------------
# Column 1: Defenses & Vulnerability
# ------------------------------------------------------------------
with col1:
    st.subheader("Defenses & Vulnerability")
    company_size = selectbox_from_preset("Company Size", "Company_Size", preset)
    firewall = toggle_from_preset("Firewall active", "Firewall", preset, True)
    mfa = toggle_from_preset("Multi-Factor Authentication enforced", "MFA", preset, True)
    edr = toggle_from_preset("EDR deployed", "EDR", preset, True)
    cvss_score = number_from_preset("Max CVSS Score of open vulns (0-10)", "CVSS_Score", preset, 5.0, min_value=0.0, max_value=10.0, step=0.1)
    patch_age = number_from_preset("Patch Age (days)", "Patch_Age_Days", preset, 45, min_value=0, max_value=365)

# ------------------------------------------------------------------
# Column 2: Kill Chain Telemetry & Response
# ------------------------------------------------------------------
with col2:
    st.subheader("Kill Chain Telemetry & Response")
    attack_stage = selectbox_from_preset("Furthest Kill Chain Stage Reached", "Attack_Stage", preset)
    phishing_click = toggle_from_preset("Phishing link clicked", "Phishing_Click", preset, False)
    credential_stolen = toggle_from_preset("Credentials stolen", "Credential_Stolen", preset, False)
    privilege_escalation = toggle_from_preset("Privilege escalation observed", "Privilege_Escalation", preset, False)
    lateral_movement = toggle_from_preset("Lateral movement observed", "Lateral_Movement", preset, False)
    persistence = toggle_from_preset("Persistence mechanism established", "Persistence", preset, False)
    data_encrypted = toggle_from_preset("Data encrypted by attacker (ransomware)", "Data_Encrypted", preset, False)
    data_exfil = number_from_preset("Data Exfiltrated so far (GB)", "Data_Exfiltration_GB", preset, 0.0, min_value=0.0, max_value=500.0, step=0.1)
    detection_time = number_from_preset("Detection Time (minutes)", "Detection_Time_Min", preset, 100, min_value=0, max_value=500)
    response_time = number_from_preset("Response Time (minutes)", "Response_Time_Min", preset, 40, min_value=0, max_value=500)

st.divider()

# ------------------------------------------------------------------
# Build the input row and predict
# ------------------------------------------------------------------
row = {
    "Attack_Stage": attack_stage, "Company_Size": company_size,
    "Lateral_Movement": lateral_movement,
    "Privilege_Escalation": privilege_escalation,
    "Persistence": persistence, "Credential_Stolen": credential_stolen,
    "Data_Exfiltration_GB": data_exfil, "Data_Encrypted": data_encrypted,
    "Phishing_Click": phishing_click,
    "Detection_Time_Min": detection_time, "Response_Time_Min": response_time,
    "Firewall": firewall, "MFA": mfa, "EDR": edr,
    "CVSS_Score": cvss_score, "Patch_Age_Days": patch_age,
}

input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

try:
    model = load_model()
    breach_prob = float(model.predict_proba(input_df)[0, 1])
except FileNotFoundError:
    st.error(
        "cyber_model.joblib not found. Place it in the same folder as app.py "
        "(it is produced by the training notebook)."
    )
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
