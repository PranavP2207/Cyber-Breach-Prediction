"""Shared 16-feature schema, demonstration cases and explanatory metadata."""

FEATURE_COLUMNS = [
    "Attack_Stage", "Company_Size", "Lateral_Movement", "Privilege_Escalation",
    "Persistence", "Credential_Stolen", "Data_Exfiltration_GB", "Data_Encrypted",
    "Phishing_Click", "Detection_Time_Min", "Response_Time_Min", "Firewall",
    "MFA", "EDR", "CVSS_Score", "Patch_Age_Days",
]
CATEGORICAL_FEATURES = ["Attack_Stage", "Company_Size"]
NUMERICAL_FEATURES = [f for f in FEATURE_COLUMNS if f not in CATEGORICAL_FEATURES]
KILL_CHAIN_STAGES = ["Reconnaissance", "Initial Access", "Execution", "Persistence", "Impact"]
CATEGORY_OPTIONS = {"Attack_Stage": KILL_CHAIN_STAGES, "Company_Size": ["Small", "Medium", "Large"]}
DEFAULTS = dict(
    Attack_Stage="Execution", Company_Size="Medium", Lateral_Movement=False,
    Privilege_Escalation=False, Persistence=False, Credential_Stolen=False,
    Data_Exfiltration_GB=0.0, Data_Encrypted=False, Phishing_Click=False,
    Detection_Time_Min=100, Response_Time_Min=40, Firewall=True, MFA=True,
    EDR=True, CVSS_Score=5.0, Patch_Age_Days=45,
)

def scenario(**changes):
    return {**DEFAULTS, **changes}

PRESETS = {
    "Manual entry / custom telemetry": None,
    "A · Contained reconnaissance": scenario(Attack_Stage="Reconnaissance", Company_Size="Large", Detection_Time_Min=25, Response_Time_Min=10, CVSS_Score=3.0, Patch_Age_Days=10),
    "B · Impact / weak defenses": scenario(Attack_Stage="Impact", Company_Size="Small", Lateral_Movement=True, Privilege_Escalation=True, Persistence=True, Credential_Stolen=True, Data_Exfiltration_GB=85.0, Data_Encrypted=True, Phishing_Click=True, Detection_Time_Min=210, Response_Time_Min=150, Firewall=False, MFA=False, EDR=False, CVSS_Score=9.4, Patch_Age_Days=160),
    "C · Initial access / phishing": scenario(Attack_Stage="Initial Access", Credential_Stolen=True, Data_Exfiltration_GB=0.5, Phishing_Click=True, Detection_Time_Min=65, Response_Time_Min=30, MFA=False, CVSS_Score=6.8, Patch_Age_Days=50),
    "D · Persistence / ransomware": scenario(Attack_Stage="Persistence", Lateral_Movement=True, Privilege_Escalation=True, Persistence=True, Credential_Stolen=True, Data_Exfiltration_GB=18.0, Data_Encrypted=True, Phishing_Click=True, Detection_Time_Min=140, Response_Time_Min=90, MFA=False, EDR=False, CVSS_Score=8.2, Patch_Age_Days=95),
    "E · Execution / reference case": scenario(),
    "F · Execution / credentials + escalation": scenario(Credential_Stolen=True, Privilege_Escalation=True),
    "G · Execution / lateral movement added": scenario(Credential_Stolen=True, Privilege_Escalation=True, Lateral_Movement=True),
    "H · Reference / all defenses off": scenario(Firewall=False, MFA=False, EDR=False),
    "I · Reference / EDR off only": scenario(EDR=False),
    "J · Reference / MFA off only": scenario(MFA=False),
    "K · Reference / phishing on only": scenario(Phishing_Click=True),
    "L · Execution / fast detection + response": scenario(Detection_Time_Min=25, Response_Time_Min=10),
    "M · Execution / slow detection + response": scenario(Detection_Time_Min=210, Response_Time_Min=150),
    "N · Persistence / no encryption or exfiltration": scenario(Attack_Stage="Persistence", Credential_Stolen=True, Privilege_Escalation=True, Lateral_Movement=True, Persistence=True),
}
CASE_NOTES = {
    "A": "Original low-risk demo; all six threat flags off and all three defensive controls on.",
    "B": "Original critical demo; all threat flags on, controls off, late-stage impact and long delays.",
    "C": "Original phishing demo. Stage is Initial Access; its risk band is computed, not assigned by its title.",
    "D": "Original ransomware demo with Firewall on and MFA/EDR off.",
    "E": "Reference row for cases F–M: all threat flags off, all controls on, Execution stage.",
    "F": "Compared with E, only Credential_Stolen and Privilege_Escalation turn on.",
    "G": "Compared with F, only Lateral_Movement turns on.",
    "H": "Compared with E, only Firewall, MFA and EDR turn off. This is a model sensitivity experiment, not a causal intervention.",
    "I": "Compared with E, only EDR turns off. The saved forest gives a lower score, illustrating confounding or unusual learned relationships.",
    "J": "Compared with E, only MFA turns off.",
    "K": "Compared with E, only Phishing_Click turns on.",
    "L": "Compared with E, detection/response delays reduce to 25/10 minutes.",
    "M": "Compared with E, detection/response delays rise to 210/150 minutes.",
    "N": "Advanced progression with encryption off and exfiltration at zero; compare with D, which also changes other inputs.",
}

# Meaning and interpretation; these are not signed model weights.
FEATURE_DETAILS = [
    ("Attack_Stage", "Progression", "5 categories", "Observed incident stage", "Summarizes progression from reconnaissance to impact. It has the largest grouped forest importance; the dataset contains stage/telemetry inconsistencies, so the timeline is illustrative."),
    ("Company_Size", "Posture", "3 categories", "Organization size category", "Context for resources, exposure and operating scale. The dataset does not establish exact employee-count boundaries."),
    ("Lateral_Movement", "Telemetry", "0 / 1", "Attacker moved between hosts", "Signals a wider foothold. Its association with success can interact with stage and privileges."),
    ("Privilege_Escalation", "Telemetry", "0 / 1", "Elevated attacker privileges", "Higher privileges can expose sensitive systems; the model learns its association from examples."),
    ("Persistence", "Telemetry", "0 / 1", "Persistent attacker foothold observed", "Indicates mechanisms for continued access. This flag is distinct from the stage category with the same name."),
    ("Credential_Stolen", "Telemetry", "0 / 1", "Credentials compromised", "Captures identity compromise and possible access expansion."),
    ("Data_Exfiltration_GB", "Telemetry", "GB, ≥ 0", "Recorded data transferred out", "Measures incident damage already observed. Positive exfiltration may constitute a breach before prediction, so this is not safely a pre-breach feature."),
    ("Data_Encrypted", "Telemetry", "0 / 1", "Malicious encryption observed", "Captures ransomware-like impact. Often available after harm has occurred; define a prediction cutoff before deployment."),
    ("Phishing_Click", "Telemetry", "0 / 1", "Phishing interaction recorded", "An initial-access signal. A click alone is not a confirmed successful intrusion."),
    ("Detection_Time_Min", "Timing", "Minutes, ≥ 0", "Delay from intrusion to detection", "Describes visibility delay. Missing training values are imputed with a training-set median; exact incident start time may be unknown in practice."),
    ("Response_Time_Min", "Timing", "Minutes, ≥ 0", "Delay from alert to containment", "Describes response speed. Final containment time is future information for an early-warning prediction."),
    ("Firewall", "Posture", "0 / 1", "Firewall enabled", "A defensive-control indicator. Presence alone does not measure configuration quality; turning it off does not necessarily increase this model's score."),
    ("MFA", "Posture", "0 / 1", "Multi-factor authentication enabled", "Represents identity protection. Model importance is predictive association, not measured real-world protection effectiveness."),
    ("EDR", "Posture", "0 / 1", "Endpoint detection and response enabled", "Represents endpoint visibility/control. The reference on/off comparison is counterintuitive and should be discussed as a limitation."),
    ("CVSS_Score", "Posture", "0–10", "Recorded vulnerability severity", "Measures vulnerability severity, not breach probability. CVSS v3.1 bands: None 0, Low 0.1–3.9, Medium 4–6.9, High 7–8.9, Critical 9–10."),
    ("Patch_Age_Days", "Posture", "Days, ≥ 0", "Recorded patch age", "A patch-hygiene proxy. The CSV does not establish whether age starts at release or application; avoid claiming a more precise meaning."),
]
