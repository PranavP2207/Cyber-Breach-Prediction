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
    page_icon="🛡️",
    layout="wide",
)

# Exact column order the model's pipeline was trained on. The
# ColumnTransformer inside the pipeline selects the categorical columns by
# name, but we still build the input frame in this order for clarity and
# to guarantee nothing is missing.
FEATURE_COLUMNS = [
    "Industry", "Country", "Company_Size", "Employee_Count", "Attack_Vector",
    "Threat_Actor", "Firewall", "MFA", "EDR", "IDS", "Security_Training",
    "Password_Policy", "Patch_Age_Days", "Open_Vulnerabilities", "CVSS_Score",
    "Internet_Exposed", "Security_Audit_Score", "Phishing_Click",
    "Credential_Stolen", "Privilege_Escalation", "Lateral_Movement",
    "Persistence", "Data_Encrypted", "Data_Exfiltration_GB", "Attack_Stage",
    "Attack_Complexity", "Detection_Time_Min", "Response_Time_Min", "Hour",
    "DayOfWeek", "Month", "Business_Hours", "Weekend", "Zero_Day",
    "Compliance", "Vendor_Count", "ThirdParty_Risk", "Insider_Risk",
    "Security_Maturity", "SOC_Team_Size",
]

# Two demo scenarios, used to pre-fill the form for a quick presentation.
PRESETS = {
    "-- Manual entry --": None,
    "Case A: Contained attack (Reconnaissance, hardened defenses)": dict(
        Industry="Technology", Country="USA", Company_Size="Large",
        Employee_Count=5000, Attack_Vector="Email", Threat_Actor="Script Kiddie",
        Firewall=1, MFA=1, EDR=1, IDS=1, Security_Training=1,
        Password_Policy="Strong", Patch_Age_Days=10, Open_Vulnerabilities=1,
        CVSS_Score=3.0, Internet_Exposed=0, Security_Audit_Score=95.0,
        Phishing_Click=0, Credential_Stolen=0, Privilege_Escalation=0,
        Lateral_Movement=0, Persistence=0, Data_Encrypted=0,
        Data_Exfiltration_GB=0.0, Attack_Stage="Reconnaissance",
        Attack_Complexity="Basic", Detection_Time_Min=25, Response_Time_Min=10,
        Hour=11, DayOfWeek="Tuesday", Month="March", Business_Hours=1,
        Weekend=0, Zero_Day=0, Compliance="ISO27001", Vendor_Count=8,
        ThirdParty_Risk=15.0, Insider_Risk=5.0, Security_Maturity=95,
        SOC_Team_Size=25,
    ),
    "Case B: Critical breach (Impact stage, weak defenses)": dict(
        Industry="Healthcare", Country="India", Company_Size="Small",
        Employee_Count=120, Attack_Vector="RDP", Threat_Actor="Nation State",
        Firewall=0, MFA=0, EDR=0, IDS=0, Security_Training=0,
        Password_Policy="Weak", Patch_Age_Days=160, Open_Vulnerabilities=15,
        CVSS_Score=9.4, Internet_Exposed=1, Security_Audit_Score=42.0,
        Phishing_Click=1, Credential_Stolen=1, Privilege_Escalation=1,
        Lateral_Movement=1, Persistence=1, Data_Encrypted=1,
        Data_Exfiltration_GB=85.0, Attack_Stage="Impact",
        Attack_Complexity="Advanced", Detection_Time_Min=210, Response_Time_Min=150,
        Hour=3, DayOfWeek="Saturday", Month="December", Business_Hours=0,
        Weekend=1, Zero_Day=1, Compliance="None", Vendor_Count=40,
        ThirdParty_Risk=80.0, Insider_Risk=60.0, Security_Maturity=20,
        SOC_Team_Size=2,
    ),
}

CATEGORY_OPTIONS = {
    "Industry": ["Banking", "Education", "Government", "Healthcare",
                 "Manufacturing", "Retail", "Technology", "Telecom"],
    "Country": ["Australia", "Canada", "Germany", "India", "Japan",
                "Singapore", "UK", "USA"],
    "Company_Size": ["Small", "Medium", "Large"],
    "Attack_Vector": ["API", "Cloud", "Email", "RDP", "USB", "VPN", "Web"],
    "Threat_Actor": ["Cyber Criminal", "Hacktivist", "Insider",
                      "Nation State", "Script Kiddie"],
    "Password_Policy": ["Weak", "Moderate", "Strong"],
    "Attack_Stage": ["Reconnaissance", "Initial Access", "Execution",
                      "Persistence", "Impact"],
    "Attack_Complexity": ["Basic", "Intermediate", "Advanced"],
    "DayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday"],
    "Month": ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"],
    "Compliance": ["None", "HIPAA", "ISO27001", "NIST", "PCI-DSS"],
}


@st.cache_resource
def load_model():
    return joblib.load("cyber_model.joblib")


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
st.title("🛡️ Cyber Breach Prediction Across the Enterprise Kill Chain")
st.caption(
    "Random Forest classifier trained on 100,000 enterprise incident records. "
    "Predicts the probability that an *in-progress* intrusion results in a "
    "successful breach, using only information available before a breach outcome is known."
)

with st.expander("ℹ️ About this model", expanded=False):
    st.markdown(
        """
        - **Model:** Random Forest (150 trees, max depth 12), scikit-learn pipeline
          with one-hot encoding for categorical fields.
        - **Test-set performance:** 86.4% accuracy · 0.77 F1-score · 0.93 ROC-AUC
        - **Zero data leakage:** columns only known *after* a breach is confirmed
          (financial loss, downtime, records compromised, recovery cost, and the
          risk/severity scores derived from them) were removed before training,
          so the model genuinely predicts forward from pre-breach signals.
        """
    )

scenario = st.selectbox("Quick scenario (optional)", list(PRESETS.keys()))
preset = PRESETS[scenario]

st.divider()

col1, col2, col3 = st.columns(3)

# ------------------------------------------------------------------
# Column 1: Enterprise Profile
# ------------------------------------------------------------------
with col1:
    st.subheader("🏢 Enterprise Profile")
    industry = selectbox_from_preset("Industry", "Industry", preset)
    country = selectbox_from_preset("Country", "Country", preset)
    company_size = selectbox_from_preset("Company Size", "Company_Size", preset)
    employee_count = number_from_preset("Employee Count", "Employee_Count", preset, 500, min_value=1, max_value=20000, step=10)
    compliance = selectbox_from_preset("Compliance Framework", "Compliance", preset)
    vendor_count = number_from_preset("Vendor Count", "Vendor_Count", preset, 10, min_value=0, max_value=150)
    thirdparty_risk = number_from_preset("Third-Party Risk Score (0-100)", "ThirdParty_Risk", preset, 30.0, min_value=0.0, max_value=100.0)
    insider_risk = number_from_preset("Insider Risk Score (0-100)", "Insider_Risk", preset, 20.0, min_value=0.0, max_value=100.0)
    security_maturity = number_from_preset("Security Maturity Score (0-100)", "Security_Maturity", preset, 75, min_value=0, max_value=100)
    soc_team_size = number_from_preset("SOC Team Size", "SOC_Team_Size", preset, 8, min_value=0, max_value=100)

    st.markdown("**Incident timing**")
    hour = number_from_preset("Hour of day (0-23)", "Hour", preset, 12, min_value=0, max_value=23)
    day_of_week = selectbox_from_preset("Day of week", "DayOfWeek", preset)
    month = selectbox_from_preset("Month", "Month", preset)
    business_hours = toggle_from_preset("Occurred during business hours", "Business_Hours", preset, True)
    weekend = toggle_from_preset("Occurred on a weekend", "Weekend", preset, False)

# ------------------------------------------------------------------
# Column 2: Defense Posture
# ------------------------------------------------------------------
with col2:
    st.subheader("🛡️ Defense Posture")
    firewall = toggle_from_preset("Firewall active", "Firewall", preset, True)
    mfa = toggle_from_preset("Multi-Factor Authentication enforced", "MFA", preset, True)
    edr = toggle_from_preset("EDR deployed", "EDR", preset, True)
    ids = toggle_from_preset("IDS/IPS deployed", "IDS", preset, True)
    security_training = toggle_from_preset("Security awareness training active", "Security_Training", preset, True)
    password_policy = selectbox_from_preset("Password Policy", "Password_Policy", preset, default_index=1)
    patch_age = number_from_preset("Patch Age (days)", "Patch_Age_Days", preset, 45, min_value=0, max_value=365)
    open_vulns = number_from_preset("Open Vulnerabilities", "Open_Vulnerabilities", preset, 3, min_value=0, max_value=50)
    cvss_score = number_from_preset("Max CVSS Score of open vulns (0-10)", "CVSS_Score", preset, 5.0, min_value=0.0, max_value=10.0, step=0.1)
    internet_exposed = toggle_from_preset("Asset directly internet-exposed", "Internet_Exposed", preset, False)
    audit_score = number_from_preset("Security Audit Score (0-100)", "Security_Audit_Score", preset, 80.0, min_value=0.0, max_value=100.0)

# ------------------------------------------------------------------
# Column 3: Kill Chain Telemetry
# ------------------------------------------------------------------
with col3:
    st.subheader("⚔️ Kill Chain Telemetry")
    attack_vector = selectbox_from_preset("Attack Vector", "Attack_Vector", preset)
    threat_actor = selectbox_from_preset("Threat Actor Type", "Threat_Actor", preset)
    attack_stage = selectbox_from_preset("Furthest Kill Chain Stage Reached", "Attack_Stage", preset)
    attack_complexity = selectbox_from_preset("Attack Complexity", "Attack_Complexity", preset, default_index=1)
    zero_day = toggle_from_preset("Zero-day exploit involved", "Zero_Day", preset, False)
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
    "Industry": industry, "Country": country, "Company_Size": company_size,
    "Employee_Count": employee_count, "Attack_Vector": attack_vector,
    "Threat_Actor": threat_actor, "Firewall": firewall, "MFA": mfa, "EDR": edr,
    "IDS": ids, "Security_Training": security_training,
    "Password_Policy": password_policy, "Patch_Age_Days": patch_age,
    "Open_Vulnerabilities": open_vulns, "CVSS_Score": cvss_score,
    "Internet_Exposed": internet_exposed, "Security_Audit_Score": audit_score,
    "Phishing_Click": phishing_click, "Credential_Stolen": credential_stolen,
    "Privilege_Escalation": privilege_escalation,
    "Lateral_Movement": lateral_movement, "Persistence": persistence,
    "Data_Encrypted": data_encrypted, "Data_Exfiltration_GB": data_exfil,
    "Attack_Stage": attack_stage, "Attack_Complexity": attack_complexity,
    "Detection_Time_Min": detection_time, "Response_Time_Min": response_time,
    "Hour": hour, "DayOfWeek": day_of_week, "Month": month,
    "Business_Hours": business_hours, "Weekend": weekend, "Zero_Day": zero_day,
    "Compliance": compliance, "Vendor_Count": vendor_count,
    "ThirdParty_Risk": thirdparty_risk, "Insider_Risk": insider_risk,
    "Security_Maturity": security_maturity, "SOC_Team_Size": soc_team_size,
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
