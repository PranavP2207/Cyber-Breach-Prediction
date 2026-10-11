"""Streamlit entry point for the 16-feature cyber incident classifier comparison."""
from pathlib import Path
from html import escape
import hashlib
import base64
import re
import json

import joblib
import pandas as pd
import streamlit as st
from project_config import (FEATURE_COLUMNS, DEFAULTS, PRESETS, CASE_NOTES,
    CATEGORY_OPTIONS, KILL_CHAIN_STAGES)

ROOT = Path(__file__).resolve().parent
RF_PATH = ROOT / 'model/cyber_model.joblib'
LR_PATH = ROOT / 'model/logistic_model.joblib'
EVAL_PATH = ROOT / 'model/evaluation.json'
MANUAL = next(iter(PRESETS))

# A transparent favicon removes the shield without restoring the default icon.
BLANK_FAVICON = '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"></svg>'
st.set_page_config(page_title='Cyber Breach Prediction', page_icon=BLANK_FAVICON, layout='wide', initial_sidebar_state='collapsed')

def render_markup(content):
    # Streamlit sanitizes inline SVG elements. An SVG data image preserves the
    # same vector gauge while remaining compatible with its HTML sanitizer.
    def image(match):
        svg = match.group(0)
        class_name = re.search(r'class="([^"]+)"', svg)
        if 'xmlns=' not in svg:
            svg = svg.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
        encoded = base64.b64encode(svg.encode()).decode()
        css_class = f' class="{class_name.group(1)}"' if class_name else ''
        return f'<img{css_class} alt="" src="data:image/svg+xml;base64,{encoded}" />'
    return st.html(re.sub(r'<svg\b.*?</svg>', image, content, flags=re.DOTALL))


CSS = r'''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {--bg:#06090f;--accent:#3ed6c4;--muted:#90a3b7;--line:rgba(156,185,206,.14);--ease:cubic-bezier(.2,.8,.2,1)}
html,body,[data-testid="stApp"] {font-family:Inter,sans-serif;color:#e7eef5}
[data-testid="stApp"] {background:radial-gradient(ellipse at 10% 0%,rgba(62,214,196,.08),transparent 45%),radial-gradient(ellipse at 95% 25%,rgba(57,101,171,.09),transparent 40%),#06090f}
[data-testid="stApp"]:before {content:"";position:fixed;inset:0;background-image:radial-gradient(rgba(62,214,196,.13) .8px,transparent .8px);background-size:32px 32px;pointer-events:none;opacity:.45;mask-image:linear-gradient(black,transparent 80%)}
[data-testid="stHeader"] {background:rgba(6,9,15,.8);backdrop-filter:blur(18px)}
[data-testid="stMainBlockContainer"] {max-width:1440px;padding:2.6rem 3rem 2rem}
[data-testid="stSidebar"] {display:none}
[data-testid="stVerticalBlock"] {gap:.85rem}
[data-testid="stMarkdownContainer"] p {color:#a7b7c9;line-height:1.65}
h1,h2,h3,h4 {font-family:Inter,sans-serif!important;letter-spacing:-.035em}
h3 {font-size:1.2rem!important} h4 {font-size:.99rem!important;padding-top:.5rem!important}
.hero {padding:12px 0 24px;animation:enter .55s var(--ease)}
.hero h1 {font-size:clamp(26px,3vw,42px);font-weight:800;line-height:1.2;margin:0;background:linear-gradient(100deg,#fff 20%,#c3eae6 62%,#3ed6c4);background-clip:text;-webkit-background-clip:text;color:transparent;text-wrap:balance}
.stTabs [data-baseweb="tab-list"],.stTabs [role="tablist"] {gap:5px;border-bottom:1px solid var(--line);margin-bottom:22px;padding-bottom:9px;background:transparent}
.stTabs [data-baseweb="tab"],.stTabs [role="tab"] {height:42px;padding:0 18px;border-radius:9px;color:#8ca0b6;font-weight:600;transition:all .25s var(--ease)}
.stTabs [data-baseweb="tab"]:hover,.stTabs [role="tab"]:hover {background:rgba(62,214,196,.04);color:#d7f5f0}
.stTabs [aria-selected="true"] {background:rgba(62,214,196,.1);color:var(--accent)!important}
.stTabs .react-aria-SelectionIndicator,.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] {display:none}
.st-key-posture-card,.st-key-telemetry-card,.st-key-timing-card,[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:var(--line)!important;border-radius:16px!important;background:rgba(17,26,38,.6);backdrop-filter:blur(16px)}
[data-baseweb="select"]>div,[data-testid="stNumberInput"]>div {background:rgba(5,12,20,.74)!important;border-color:rgba(156,185,206,.18)!important;border-radius:8px!important;transition:all .25s var(--ease)}
[data-baseweb="select"]>div:focus-within,[data-testid="stNumberInput"]>div:focus-within {border-color:var(--accent)!important;box-shadow:0 0 0 2px rgba(62,214,196,.09)}
[data-testid="stWidgetLabel"] p {font-size:12px!important;color:#bdcbd9!important}
[data-testid="stCheckbox"] label {font-size:12px}
[data-testid="stCheckbox"] input:checked + div {filter:drop-shadow(0 0 5px rgba(62,214,196,.3))}
[data-testid="stNumberInput"] input {font-family:'JetBrains Mono',monospace!important;font-size:13px!important}
.stButton button,.stDownloadButton button {border:1px solid rgba(62,214,196,.22);border-radius:8px;background:rgba(62,214,196,.06);color:#bcefe7;transition:all .25s var(--ease)}
.stButton button:hover,.stDownloadButton button:hover {border-color:var(--accent);background:rgba(62,214,196,.12);transform:translateY(-2px)}
.section-label {font:10px 'JetBrains Mono',monospace;color:#7e94aa;letter-spacing:.15em;text-transform:uppercase;margin-bottom:11px}
.card {position:relative;background:linear-gradient(135deg,rgba(21,32,47,.75),rgba(12,19,30,.73));backdrop-filter:blur(18px);border:1px solid var(--line);border-radius:16px;padding:22px;overflow:hidden;transition:all .3s var(--ease);animation:enter .55s var(--ease) both}
.card:hover {transform:translateY(-2px);border-color:rgba(62,214,196,.35);box-shadow:0 8px 30px rgba(0,0,0,.18)}
.card:before {content:"";position:absolute;top:0;left:15%;right:15%;height:1px;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:0;transition:opacity .3s}
.card:hover:before {opacity:1}
.kpis {display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 0 18px}
.kpis .card:nth-child(2) {animation-delay:.06s}.kpis .card:nth-child(3) {animation-delay:.12s}.kpis .card:nth-child(4) {animation-delay:.18s}
.kpi-number {font:600 25px 'JetBrains Mono',monospace;margin:9px 0;color:#edf7f6}
.small {font-size:11px;color:#8297ad;line-height:1.65}
.risk-card {border-color:color-mix(in srgb,var(--risk) 40%,transparent);box-shadow:0 0 45px color-mix(in srgb,var(--risk) 8%,transparent)}
.risk-card:hover {border-color:var(--risk)}
.model-results {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;align-items:stretch}
.model-result {min-width:0;padding:18px 14px;display:flex;flex-direction:column}
.model-result .section-label {text-align:center;letter-spacing:.09em;font-size:9px;margin-bottom:6px}
.model-name {min-height:44px;display:flex;align-items:center;justify-content:center;text-align:center;font-size:16px;font-weight:600;line-height:1.35;color:#e7eef5}
.pill {display:inline-block;border:1px solid color-mix(in srgb,var(--risk) 35%,transparent);padding:5px 8px;border-radius:5px;font:10px 'JetBrains Mono',monospace;color:var(--risk);background:color-mix(in srgb,var(--risk) 8%,transparent)}
.gauge {position:relative;margin:16px auto 10px;max-width:230px;width:100%;text-align:center}
.gauge img {display:block;width:100%;overflow:visible}
.gauge-value {position:absolute;inset:auto 0 9px;font:600 clamp(22px,2.5vw,32px) 'JetBrains Mono',monospace;letter-spacing:-.06em;color:var(--risk)}
.gauge-value small {font-size:14px;margin-left:2px}
.gauge-labels {display:flex;justify-content:space-between;font:9px 'JetBrains Mono',monospace;color:#7e94aa}
.model-tier {display:flex;justify-content:center;margin:14px 0 10px}
.model-result .linear {margin:0}
.result-details {padding:0 2px}
.risk-copy {font-size:12px;color:#afbece;margin:15px 0 10px;line-height:1.6}
.linear {height:5px;border-radius:8px;background:#24303b;overflow:hidden;margin:14px 0}
.linear span {display:block;height:100%;background:var(--risk);border-radius:8px;animation:bar .65s var(--ease)}
.legend {display:flex;gap:8px;flex-wrap:wrap;font:9px 'JetBrains Mono',monospace;color:#8499ae;margin-top:12px}
.legend i {display:inline-block;width:5px;height:5px;border-radius:50%;margin-right:4px}
.st-key-prediction-panel {position:sticky;top:75px}
.timeline {display:flex;gap:12px;align-items:center;overflow-x:auto;padding:4px 0 14px;margin:12px 0;scroll-snap-type:x proximity}
.stage {position:relative;box-sizing:border-box;display:flex;flex-direction:column;justify-content:space-between;flex:1;min-width:160px;min-height:112px;padding:18px 16px;border:1px solid var(--line);border-radius:12px;background:rgba(15,25,37,.6);transition:all .25s var(--ease);scroll-snap-align:start}
.stage:hover {transform:translateY(-2px);border-color:var(--stage)}
.stage.current {border-color:var(--stage);background:color-mix(in srgb,var(--stage) 9%,transparent)}
.stage-name {font-size:15px;font-weight:600;line-height:1.35;margin:10px 0}
.stage small {font:10.5px 'JetBrains Mono',monospace;color:#8fa5ba}
.stage.current small {color:var(--stage)}
.stage-track {height:3px;background:#23303d;border-radius:5px;overflow:hidden;margin-top:10px}
.stage-track span {display:block;height:100%;background:var(--stage);animation:bar .7s var(--ease)}
.arrow {flex:0 0 auto;color:#59738b;font-size:26px}
.compare-table {width:100%;border-collapse:collapse;font-size:12px}
.compare-table th {text-align:left;font-size:10px;color:#8195ac;font-weight:500;border-bottom:1px solid var(--line);padding:11px 12px;white-space:nowrap}
.compare-table td {padding:12px;border-bottom:1px solid rgba(156,185,206,.07);color:#bdd0df}
.compare-table td:not(:first-child) {font-family:'JetBrains Mono',monospace;font-size:11px}
.compare-table .best {color:#3ed6c4}
.table-scroll {overflow-x:auto}
.pipeline {border-left:1px solid rgba(62,214,196,.3);padding-left:20px;margin-left:6px}
.pipeline div {position:relative;padding:5px 0 20px;font-size:12px;color:#9cb0c4;transition:transform .25s var(--ease)}
.pipeline div:before {content:'›';position:absolute;left:-28px;top:5px;width:15px;height:15px;border:1px solid rgba(62,214,196,.4);border-radius:50%;background:#0c1720;color:var(--accent);text-align:center;font-size:12px;box-shadow:0 0 8px rgba(62,214,196,.12)}
.pipeline div:hover {transform:translateX(3px)}
.pipeline strong {display:block;color:#e4eef6;margin-bottom:4px}
.importance {margin:12px 0;font-size:11px}.importance header {display:flex;justify-content:space-between;color:#aebfcf;margin-bottom:6px}.importance .linear {height:3px;margin:0}
@keyframes enter {from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes bar {from{transform:scaleX(0);transform-origin:left}to{transform:scaleX(1);transform-origin:left}}
@media(max-width:900px) {[data-testid="stMainBlockContainer"] {padding:2.2rem 1.3rem}.kpis{grid-template-columns:repeat(2,1fr)}.st-key-prediction-panel{position:static}}
@media(max-width:640px) {[data-testid="stMainBlockContainer"]{padding:3.2rem 1rem 1.6rem}.hero{padding-bottom:22px}.hero h1{font-size:26px}.stTabs [data-baseweb="tab"],.stTabs [role="tab"]{padding:0 10px;font-size:12px}.stTabs [role="tab"] p{font-size:12px!important}.card{padding:18px}.kpi-number{font-size:23px}.kpis{gap:9px}.model-results{grid-template-columns:1fr}.gauge-value{font-size:36px}.compare-table td,.compare-table th{padding:10px 8px}}
@media(max-width:360px) {.hero h1{font-size:24px}.stTabs [role="tab"]{padding:0 6px}.stTabs [role="tab"] p{font-size:10.5px!important}}
@media(prefers-reduced-motion:reduce) {*{animation:none!important;transition:none!important}}
</style>
'''
render_markup(CSS)


def on_scenario_change():
    values = PRESETS.get(st.session_state.get('scenario_selector'))
    if values:
        for key, value in values.items():
            st.session_state[key] = value


def on_manual_change():
    st.session_state['scenario_selector'] = MANUAL


def reset_inputs():
    st.session_state.update(DEFAULTS)
    st.session_state['scenario_selector'] = MANUAL


for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


@st.cache_resource
def load_bundle(rf_stamp, lr_stamp, evaluation_stamp):
    """File signatures participate in the cache key; paths are anchored to app.py."""
    metadata = json.loads(EVAL_PATH.read_text(encoding='utf-8'))
    for path in (RF_PATH, LR_PATH):
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if metadata['artifacts'].get(path.name) != digest:
            raise ValueError(f'{path.name} and evaluation.json are out of sync. Run python retrain_model.py.')
    rf, lr = joblib.load(RF_PATH), joblib.load(LR_PATH)
    for model in (rf, lr):
        if list(model.feature_names_in_) != FEATURE_COLUMNS:
            raise ValueError('Model feature schema must match all 16 inputs.')
    return rf, lr, metadata


def risk(prob):
    if prob < .25:
        return 'LOW', '#34d399'
    if prob < .5:
        return 'MEDIUM', '#fbbf24'
    if prob < .75:
        return 'HIGH', '#f97316'
    return 'CRITICAL', '#ef4444'


def probability(model, row):
    return float(model.predict_proba(row)[0, list(model.classes_).index(1)])


def model_card(name, prob, tier, color, element_id):
    """Use identical markup and dimensions for both model outputs."""
    return f'''<div class="card risk-card model-result" style="--risk:{color}">
        <div class="section-label">Breach probability</div><div class="model-name">{escape(name)}</div>
        <div class="gauge" role="img" aria-label="{escape(name)} breach probability {prob:.1%}">
            <svg viewBox="0 0 240 135"><path d="M20 120 A100 100 0 0 1 220 120" fill="none" stroke="#273441" stroke-width="10" stroke-linecap="round"/>
            <path d="M20 120 A100 100 0 0 1 220 120" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round" pathLength="100" stroke-dasharray="{prob*100} 100"/>
            <path d="M31 120 A89 89 0 0 1 209 120" fill="none" stroke="#24313d" stroke-width="1" stroke-dasharray="1 8"/></svg>
            <div class="gauge-value" id="{element_id}">{prob*100:.1f}<small>%</small></div>
        </div><div class="gauge-labels"><span>0%</span><span>100%</span></div>
        <div class="model-tier"><span class="pill">{tier}</span></div>
        <div class="linear"><span style="width:{prob*100}%"></span></div>
    </div>'''


def percent(value, digits=2):
    # Consistent half-up display rounding with the JavaScript guide.
    return f'{value * 100 + 1e-10:.{digits}f}%'


try:
    signatures = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in (RF_PATH, LR_PATH, EVAL_PATH))
    forest, logistic, evaluation = load_bundle(*signatures)
except (OSError, ValueError, KeyError) as error:
    st.error(f'Model setup: {error}')
    st.code('python retrain_model.py')
    st.stop()

render_markup('<div class="hero"><h1>CYBER BREACH PREDICTION USING MACHINE LEARNING</h1></div>')
tab_predictor, tab_comparison = st.tabs(['Predictor', 'Model comparison'])

with tab_predictor:
    inputs, results = st.columns([1.6, 1], gap='large')
    with inputs:
        st.markdown('### Incident simulator')
        st.caption('Load a case, then change any signal. Both models update on every edit.')
        st.selectbox('Demo scenario', list(PRESETS), key='scenario_selector', on_change=on_scenario_change)
        selected = st.session_state.get('scenario_selector', MANUAL)
        if selected != MANUAL:
            st.caption(CASE_NOTES[selected[0]])
        left, right = st.columns(2, gap='medium')
        with left, st.container(border=True, key="posture-card"):
            st.markdown('#### Defensive posture')
            st.selectbox('Company size', CATEGORY_OPTIONS['Company_Size'], key='Company_Size', on_change=on_manual_change, help='Dataset size category; exact staff-count boundaries are not documented.')
            for key, label, help_text in [('Firewall','Firewall','Network boundary firewall enabled.'), ('MFA','MFA','Multi-factor authentication enabled.'), ('EDR','EDR','Endpoint detection and response enabled.')]:
                st.toggle(label, key=key, on_change=on_manual_change, help=help_text)
            st.number_input('CVSS severity · 0–10', min_value=0.0, max_value=10.0, step=.1, key='CVSS_Score', on_change=on_manual_change, help='Vulnerability severity, not breach probability.')
            st.number_input('Patch age · days', min_value=0, max_value=365, key='Patch_Age_Days', on_change=on_manual_change, help='Recorded patch-age proxy; origin of the interval is not specified in the CSV.')
        with right, st.container(border=True, key="telemetry-card"):
            st.markdown('#### Attack telemetry')
            st.selectbox('Observed attack stage', CATEGORY_OPTIONS['Attack_Stage'], key='Attack_Stage', on_change=on_manual_change)
            for key, label in [('Phishing_Click','Phishing clicked'),('Credential_Stolen','Credentials stolen'),('Privilege_Escalation','Privilege escalation'),('Lateral_Movement','Lateral movement'),('Persistence','Persistence observed'),('Data_Encrypted','Data encrypted')]:
                st.toggle(label, key=key, on_change=on_manual_change)
            st.number_input('Data exfiltrated · GB', min_value=0.0, max_value=500.0, step=.1, key='Data_Exfiltration_GB', on_change=on_manual_change)
        with st.container(border=True, key="timing-card"):
            st.markdown('#### Detection & response')
            dc, rc = st.columns(2)
            with dc:
                st.number_input('Detection delay · minutes', min_value=0, max_value=500, key='Detection_Time_Min', on_change=on_manual_change, help='Recorded interval from intrusion to detection.')
            with rc:
                st.number_input('Response delay · minutes', min_value=0, max_value=500, key='Response_Time_Min', on_change=on_manual_change, help='Recorded interval from alert to containment; this may be unavailable at an early prediction cutoff.')
        st.button('Reset to reference inputs', on_click=reset_inputs)

    row_data = {key: int(st.session_state[key]) if isinstance(DEFAULTS[key], bool) else st.session_state[key] for key in FEATURE_COLUMNS}
    frame = pd.DataFrame([row_data], columns=FEATURE_COLUMNS)
    rf_probability, lr_probability = probability(forest, frame), probability(logistic, frame)
    rf_tier, rf_color = risk(rf_probability)
    lr_tier, lr_color = risk(lr_probability)
    with results, st.container(key='prediction-panel'):
        cards = model_card('Random Forest', rf_probability, rf_tier, rf_color, 'rf-probability')
        cards += model_card('Logistic Regression', lr_probability, lr_tier, lr_color, 'lr-probability')
        render_markup('<div class="model-results">' + cards + '</div>' + f'''<div class="result-details">
            <div class="risk-copy">Both scores estimate <strong>Attack_Success = 1</strong> for the same incident inputs.</div>
            <div class="legend"><span><i style="background:#34d399"></i>LOW &lt;25%</span><span><i style="background:#fbbf24"></i>MEDIUM &lt;50%</span><span><i style="background:#f97316"></i>HIGH &lt;75%</span><span><i style="background:#ef4444"></i>CRITICAL ≥75%</span></div>
            <div class="small" style="margin-top:14px">Absolute disagreement: {abs(rf_probability-lr_probability)*100:.1f} percentage points. A higher score does not establish a more accurate prediction.</div>
        </div>''')
        st.caption('Academic incident-classification prototype. These probabilities have not been calibrated or validated for real-world security decisions.')
        if rf_tier != lr_tier:
            st.info('The models assign different risk bands. Compare their validation tradeoffs in the next tab.')
        if row_data['Data_Encrypted'] or row_data['Data_Exfiltration_GB'] > 0:
            st.caption('Encryption or exfiltration already indicates observed harm. Interpret this as incident assessment, not a proven early warning.')
        assessment = {'inputs': row_data, 'outputs': {'Random Forest': {'probability':rf_probability,'risk_tier':rf_tier,'class_at_0_5':int(rf_probability>=.5)}, 'Logistic Regression': {'probability':lr_probability,'risk_tier':lr_tier,'class_at_0_5':int(lr_probability>=.5)}}, 'model_sha256': {p.name:evaluation['artifacts'][p.name] for p in (RF_PATH,LR_PATH)}}
        st.download_button('Download assessment · JSON', json.dumps(assessment,indent=2), 'incident-assessment.json', 'application/json', width='stretch')

    with st.container(key='kill-chain-panel'):
        index = KILL_CHAIN_STAGES.index(st.session_state['Attack_Stage'])
        stage_colors = ['#34d399','#b7df51','#fbbf24','#f97316','#ef4444']
        nodes = []
        for i, stage in enumerate(KILL_CHAIN_STAGES):
            status = 'SELECTED' if i == index else 'EARLIER' if i < index else 'LATER'
            nodes.append(f'<div class="stage {"current" if i == index else ""}" style="--stage:{stage_colors[i]}"><small>0{i+1} / {status}</small><div class="stage-name">{escape(stage)}</div><div class="stage-track"><span style="width:{100 if i <= index else 0}%"></span></div></div>')
        render_markup('<div class="section-label" style="margin-top:16px;font-size:12px">Observed kill-chain stage</div><div class="timeline">' + '<span class="arrow">›</span>'.join(nodes) + '</div>')
        st.caption('The timeline orders stages; it does not infer that every earlier event occurred.')

with tab_comparison:
    rf_metrics, lr_metrics = evaluation['metrics']['Random Forest'], evaluation['metrics']['Logistic Regression']
    render_markup(f'''<div class="kpis"><div class="card"><div class="section-label">Forest accuracy</div><div class="kpi-number">{percent(rf_metrics['accuracy'])}</div><div class="small">150 trees · depth 12</div></div><div class="card"><div class="section-label">Logistic accuracy</div><div class="kpi-number">{percent(lr_metrics['accuracy'])}</div><div class="small">Scaled linear baseline</div></div><div class="card"><div class="section-label">Training rows</div><div class="kpi-number">80,000</div><div class="small">20,000 held-out rows</div></div><div class="card"><div class="section-label">Model inputs</div><div class="kpi-number">16 → 22</div><div class="small">2 categorical · 14 numeric/binary</div></div></div>''')
    st.markdown('### Which model performs better?')
    st.write('The forest leads on accuracy, F1, recall and ROC-AUC in this split. Logistic Regression has slightly higher precision. The differences are small; repeated validation is needed to establish a consistent advantage.')
    rows = ''
    for key, label, percentage in [('accuracy','Accuracy',True),('precision','Breach precision',True),('recall','Breach recall',True),('f1','Breach F1',False),('roc_auc','ROC-AUC',False),('average_precision','Average precision',False),('brier','Brier score ↓',False)]:
        a,b = rf_metrics[key],lr_metrics[key]
        better_rf = a < b if key == 'brier' else a > b
        av,bv = (percent(a),percent(b)) if percentage else (f'{a:.4f}',f'{b:.4f}')
        rows += f'<tr><td>{label}</td><td class="{"best" if better_rf else ""}">{av}</td><td class="{"best" if not better_rf else ""}">{bv}</td></tr>'
    render_markup('<div class="card table-scroll"><table class="compare-table"><thead><tr><th>Same holdout · threshold 0.5</th><th>Random Forest</th><th>Logistic Regression</th></tr></thead><tbody>'+rows+'</tbody></table></div>')
    st.caption(f"Majority-class accuracy baseline: {evaluation['majority_baseline_accuracy']*100+1e-10:.2f}%. Test set: 5,881 breaches and 14,119 non-breaches. Teal indicates the better value; Brier is lower-is-better.")
    a,b = st.columns(2)
    for column, name, metrics in [(a,'Random Forest',rf_metrics),(b,'Logistic Regression',lr_metrics)]:
        tn,fp = metrics['confusion_matrix'][0]; fn,tp = metrics['confusion_matrix'][1]
        with column:
            render_markup(f'<div class="card"><div class="section-label">{name} · confusion matrix</div><table class="compare-table"><thead><tr><th>Actual / predicted</th><th>Contained</th><th>Breach</th></tr></thead><tbody><tr><td>Contained</td><td>{tn:,} TN</td><td>{fp:,} FP</td></tr><tr><td>Breach</td><td>{fn:,} FN</td><td>{tp:,} TP</td></tr></tbody></table></div>')
    st.caption('The forest identifies 152 more true breaches but produces 109 more false alarms than LR on this holdout. Risk-band cutoffs are demonstration choices; the classification metrics use 0.5.')
    with st.expander('Training protocol & model provenance'):
        st.write(evaluation['protocol'])
        st.write(evaluation['forest_provenance'])
        st.write('Forest: 150 trees, max depth 12, min samples per leaf 5, class_weight=None. LR: training-only median imputation, one-hot encoding, numeric standardization, regularized logistic regression; converged in 24 iterations.')
        if evaluation['cross_validation']:
            st.json(evaluation['cross_validation'])
        else:
            st.caption('Cross-validation has not been run for this displayed comparison. Run python retrain_model.py --cv 5 to generate training-fold ROC-AUC results.')
    flow, feature_column = st.columns(2, gap='large')
    with flow:
        st.markdown('#### How the data moves')
        render_markup('''<div class="card"><div class="pipeline"><div><strong>01 / Prepare incidents</strong>100,500 CSV rows → remove 500 duplicates → 100,000 unique rows.</div><div><strong>02 / Select the schema</strong>16 inputs and Attack_Success; outcome-derived fields and identifiers excluded.</div><div><strong>03 / Split before fitting</strong>80,000 train / 20,000 test, stratified, seed 42.</div><div><strong>04 / Learn transformations</strong>5 stage + 3 size dummies, 14 numerical/binary inputs. LR also imputes and scales.</div><div><strong>05 / Train & evaluate</strong>Fit classifiers on train rows; compare predictions on the same test rows.</div><div><strong>06 / Serve the saved pipelines</strong>16 widgets → ordered DataFrame → predict_proba → two scores → risk bands.</div></div></div>''')
    with feature_column:
        st.markdown('#### What the forest uses most')
        bars = ''
        for feature,value in sorted(evaluation['grouped_importance'].items(),key=lambda item:item[1],reverse=True)[:8]:
            bars += f'<div class="importance"><header><span>{escape(feature)}</span><span>{value:.1%}</span></header><div class="linear" style="--risk:#3ed6c4"><span style="width:{value*100}%"></span></div></div>'
        render_markup('<div class="card">'+bars+'<div class="small" style="margin-top:18px">Grouped impurity importance from the actual saved forest. Categorical dummies are summed. This is not a signed effect, probability contribution or causal importance.</div></div>')
    with st.expander('All 14 cases · measured model outputs'):
        case_rows = [{'Case':case['name'],'Random Forest':f"{case['outputs']['Random Forest']['probability']:.1%}",'RF risk':case['outputs']['Random Forest']['tier'],'Logistic Regression':f"{case['outputs']['Logistic Regression']['probability']:.1%}",'LR risk':case['outputs']['Logistic Regression']['tier']} for case in evaluation['cases']]
        st.dataframe(pd.DataFrame(case_rows),hide_index=True,width='stretch')
    st.download_button('Download measured comparison · JSON', EVAL_PATH.read_text(encoding='utf-8'),'model-comparison.json','application/json')
