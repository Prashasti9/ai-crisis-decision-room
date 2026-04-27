import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from langgraph_workflow import run_crisis_graph

st.set_page_config(
    page_title="AI Crisis Decision Room",
    page_icon="🚨",
    layout="wide"
)

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# -----------------------------
# Helpers
# -----------------------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_ai_sections(text: str):
    admin_text = ""
    public_text = ""

    if "Hospital Admin Alert" in text and "Public Advisory" in text:
        parts = text.split("Public Advisory", 1)
        admin_text = parts[0].replace("Hospital Admin Alert", "").strip()
        public_text = parts[1].strip()
    elif "HOSPITAL ADMIN ALERT" in text and "PUBLIC ADVISORY" in text:
        parts = text.split("PUBLIC ADVISORY", 1)
        admin_text = parts[0].replace("HOSPITAL ADMIN ALERT", "").strip()
        public_text = parts[1].strip()
    else:
        admin_text = text.strip()

    return admin_text, public_text


def metric_card(label, value, subtitle=""):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #081229 0%, #0b1736 100%);
            padding: 18px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 24px rgba(0,0,0,0.10);
            min-height: 120px;
        ">
            <div style="font-size:14px; color:#a5b4fc; margin-bottom:10px;">{label}</div>
            <div style="font-size:24px; font-weight:800; color:white; margin-bottom:8px;">{value}</div>
            <div style="font-size:12px; color:#94a3b8;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def status_badge(label, status, color):
    st.markdown(
        f"""
        <div style="
            background:{color};
            color:white;
            padding:10px 14px;
            border-radius:12px;
            font-weight:700;
            margin-bottom:8px;
            display:inline-block;
        ">
            {label}: {status}
        </div>
        """,
        unsafe_allow_html=True
    )


def notify_slack(state: dict, risk_level: str, scenario: str):
    if not SLACK_WEBHOOK_URL:
        return False, "Slack webhook URL not configured"

    actions = state.get("actions", [])
    action_text = "\n".join([f"• {a}" for a in actions]) if actions else "No actions available"

    color = (
        "#991b1b" if risk_level == "SEVERE"
        else "#b91c1c" if risk_level == "HIGH"
        else "#b45309" if risk_level == "MODERATE"
        else "#166534"
    )

    payload = {
        "text": f"🚨 {risk_level} Crisis Detected",
        "attachments": [
            {
                "color": color,
                "title": f"AI Crisis Decision Room — {scenario}",
                "fields": [
                    {"title": "Scenario", "value": scenario, "short": True},
                    {"title": "Risk Level", "value": risk_level, "short": True},
                    {"title": "EMS Calls", "value": str(state.get('ems_calls')), "short": True},
                    {"title": "Predicted Surge", "value": str(state.get('predicted_surge')), "short": True},
                    {"title": "Heat Risk", "value": str(state.get('heat_risk')), "short": True},
                    {"title": "Disaster Flag", "value": str(state.get('disaster_flag')), "short": True},
                    {"title": "Recommended Actions", "value": action_text, "short": False},
                ],
                "footer": "IBM watsonx Crisis System"
            }
        ]
    }

    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        return True, "Slack alert sent"
    except Exception as e:
        return False, str(e)


def scenario_summary_text(scenario: str):
    if scenario == "Heatwave Surge":
        return "Extreme temperature conditions driving elevated emergency demand."
    if scenario == "Flood Disaster":
        return "Flood-related disruption increasing rescue calls, transport delays, and hospital strain."
    if scenario == "Disease Outbreak":
        return "Public health surge increasing hospital load and sustained emergency demand."
    return "Baseline operating conditions with manageable emergency demand."


def get_hospitals_for_scenario(scenario: str):
    if scenario == "Normal Day":
        return [
            {"name": "SF General", "util": 0.68},
            {"name": "Mission Bay", "util": 0.61},
            {"name": "City Hospital", "util": 0.72},
        ]
    if scenario == "Heatwave Surge":
        return [
            {"name": "SF General", "util": 0.88},
            {"name": "Mission Bay", "util": 0.76},
            {"name": "City Hospital", "util": 0.93},
        ]
    if scenario == "Flood Disaster":
        return [
            {"name": "SF General", "util": 0.91},
            {"name": "Mission Bay", "util": 0.83},
            {"name": "City Hospital", "util": 0.95},
        ]
    return [
        {"name": "SF General", "util": 0.94},
        {"name": "Mission Bay", "util": 0.87},
        {"name": "City Hospital", "util": 0.96},
    ]


# -----------------------------
# Load base data
# -----------------------------
situation_path = Path("outputs/situation.json")

if not situation_path.exists():
    st.error("outputs/situation.json not found. Run the notebook first.")
    st.stop()

base_situation = load_json(situation_path)

base_ems_calls = int(base_situation.get("ems_calls", 0))
base_predicted_surge = int(base_situation.get("predicted_surge", 0))
base_heat_risk = bool(base_situation.get("heat_risk", False))
base_disaster_flag = int(base_situation.get("disaster_flag", 0))
base_hospital_utilization = base_situation.get("hospital_utilization", None)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Simulation Controls")

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Heatwave Surge", "Flood Disaster", "Disease Outbreak", "Normal Day"]
)

severity = st.sidebar.slider("Scenario Severity", 1, 5, 3)
show_before_after = st.sidebar.checkbox("Show before/after impact", value=True)
run_langgraph = st.sidebar.button("Run LangGraph + Granite")
enable_slack_alerts = st.sidebar.checkbox("Enable Slack alerts", value=True)
manual_test_alert = st.sidebar.button("Send Test Slack Alert")

# -----------------------------
# Session state
# -----------------------------
if "last_alert" not in st.session_state:
    st.session_state["last_alert"] = None

if "alert_log" not in st.session_state:
    st.session_state["alert_log"] = []

if "graph_result" not in st.session_state:
    st.session_state["graph_result"] = None

# -----------------------------
# Scenario simulation
# -----------------------------
ems_calls = base_ems_calls
predicted_surge = base_predicted_surge
heat_risk = base_heat_risk
disaster_flag = base_disaster_flag
hospital_utilization = base_hospital_utilization

if scenario == "Heatwave Surge":
    ems_calls = base_ems_calls + severity * 40
    predicted_surge = base_predicted_surge + severity * 55
    heat_risk = True
    disaster_flag = 0

elif scenario == "Flood Disaster":
    ems_calls = base_ems_calls + severity * 70
    predicted_surge = base_predicted_surge + severity * 90
    heat_risk = False
    disaster_flag = 1

elif scenario == "Disease Outbreak":
    ems_calls = base_ems_calls + severity * 55
    predicted_surge = base_predicted_surge + severity * 75
    heat_risk = False
    disaster_flag = 1

elif scenario == "Normal Day":
    ems_calls = max(180, base_ems_calls - 180)
    predicted_surge = max(220, base_predicted_surge - 220)
    heat_risk = False
    disaster_flag = 0

scenario_summary = scenario_summary_text(scenario)
hospitals = get_hospitals_for_scenario(scenario)

trend_df = pd.DataFrame({
    "Hour": ["06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
    "EMS Calls": [
        max(100, ems_calls - 260),
        max(150, ems_calls - 210),
        max(200, ems_calls - 150),
        max(250, ems_calls - 100),
        max(300, ems_calls - 40),
        ems_calls
    ]
})

initial_state = {
    "date": base_situation.get("date", "N/A"),
    "scenario": scenario,
    "ems_calls": ems_calls,
    "predicted_surge": predicted_surge,
    "heat_risk": heat_risk,
    "hospital_utilization": hospital_utilization,
    "disaster_flag": disaster_flag,
}

# Run graph on demand
if run_langgraph:
    try:
        with st.spinner("Running LangGraph workflow with Granite..."):
            graph_result = run_crisis_graph(initial_state)
            st.session_state["graph_result"] = graph_result
    except Exception as e:
        st.sidebar.error(f"LangGraph run failed: {e}")

graph_result = st.session_state.get("graph_result")

if graph_result is None:
    graph_result = {
        **initial_state,
        "risk_level": "NOT RUN",
        "risk_note": "Click 'Run LangGraph + Granite' to execute the workflow.",
        "actions": [],
        "resource_recommendations": [],
        "generated_text": "No live AI response yet.",
    }

risk_level = graph_result.get("risk_level", "UNKNOWN")
risk_note = graph_result.get("risk_note", "")
actions = graph_result.get("actions", [])
resource_recommendations = graph_result.get("resource_recommendations", [])
generated_text = graph_result.get("generated_text", "No generated text.")

admin_text, public_text = parse_ai_sections(generated_text)

# -----------------------------
# Slack alert logic
# -----------------------------
current_alert = f"{scenario}-{severity}-{risk_level}-{ems_calls}-{predicted_surge}"

if manual_test_alert:
    success, message = notify_slack(graph_result, risk_level, f"TEST: {scenario}")
    if success:
        st.sidebar.success("Test Slack alert sent")
        st.session_state["alert_log"].append({
            "type": "manual_test",
            "scenario": scenario,
            "risk": risk_level,
            "status": "sent"
        })
    else:
        st.sidebar.error(f"Slack test failed: {message}")

if enable_slack_alerts and risk_level in ["HIGH", "SEVERE"] and st.session_state["last_alert"] != current_alert:
    success, message = notify_slack(graph_result, risk_level, scenario)
    if success:
        st.session_state["last_alert"] = current_alert
        st.session_state["alert_log"].append({
            "type": "auto_alert",
            "scenario": scenario,
            "risk": risk_level,
            "status": "sent"
        })
    else:
        st.session_state["alert_log"].append({
            "type": "auto_alert",
            "scenario": scenario,
            "risk": risk_level,
            "status": f"failed: {message}"
        })

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div style="padding: 8px 0 16px 0;">
        <div style="font-size:18px; font-weight:700; color:#ef4444;">🚨 Emergency Intelligence Dashboard</div>
        <div style="font-size:50px; font-weight:900; line-height:1.05; margin-top:6px;">
            AI Crisis Decision Room
        </div>
        <div style="font-size:18px; color:#6b7280; margin-top:12px;">
            Live crisis simulation using Python + LangGraph + IBM watsonx.ai + watsonx Orchestrate
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

status_col1, status_col2, status_col3 = st.columns([1, 1, 2])

with status_col1:
    if risk_level == "SEVERE":
        status_badge("Risk", risk_level, "#991b1b")
    elif risk_level == "HIGH":
        status_badge("Risk", risk_level, "#b91c1c")
    elif risk_level == "MODERATE":
        status_badge("Risk", risk_level, "#b45309")
    else:
        status_badge("Risk", risk_level, "#166534")

with status_col2:
    status_badge("Scenario", scenario, "#1d4ed8")

with status_col3:
    st.info(scenario_summary)

# -----------------------------
# Metrics
# -----------------------------
row1 = st.columns(4)
with row1[0]:
    metric_card("Date", graph_result.get("date", "N/A"), "Current simulation date")
with row1[1]:
    metric_card("EMS Calls", graph_result.get("ems_calls", "N/A"), "Observed demand")
with row1[2]:
    metric_card("Predicted Surge", graph_result.get("predicted_surge", "N/A"), "Expected near-term pressure")
with row1[3]:
    metric_card("Disaster Flag", graph_result.get("disaster_flag", "N/A"), "1 = active event, 0 = none")

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

row2 = st.columns(3)
with row2[0]:
    metric_card("Heat Risk", str(graph_result.get("heat_risk", "N/A")), "Environmental stress indicator")
with row2[1]:
    metric_card(
        "Hospital Utilization",
        "N/A" if graph_result.get("hospital_utilization") is None else graph_result.get("hospital_utilization"),
        "Current reported system utilization"
    )
with row2[2]:
    metric_card("Active Actions", len(actions), "Response actions currently triggered")

# -----------------------------
# Risk strip
# -----------------------------
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
st.subheader("Risk Level")

if risk_level in ["SEVERE", "HIGH"]:
    st.error(f"{risk_level} RISK - {risk_note}")
elif risk_level == "MODERATE":
    st.warning(f"{risk_level} RISK - {risk_note}")
else:
    st.success(f"{risk_level} RISK - {risk_note}")

# -----------------------------
# Main layout
# -----------------------------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("📈 Demand Trend")
    st.line_chart(trend_df.set_index("Hour"))

    if show_before_after:
        st.subheader("⚖️ Before vs After Response")
        before_col, after_col = st.columns(2)

        with before_col:
            st.markdown("#### Before Response")
            st.markdown(
                f"""
                - EMS calls under pressure: **{graph_result.get('ems_calls')}**
                - Predicted surge: **{graph_result.get('predicted_surge')}**
                - Risk level: **{risk_level}**
                - Likely congestion in emergency intake
                """
            )

        with after_col:
            st.markdown("#### After Recommended Actions")
            reduced_load = max(0, int(graph_result.get("predicted_surge", 0)) - 80)
            st.markdown(
                f"""
                - Effective managed surge: **{reduced_load}**
                - Cooling / triage diversion activated
                - Better coordination across hospitals
                - Faster resource deployment response
                """
            )

    st.subheader("🏥 Hospital Status")
    for h in hospitals:
        util_pct = int(h["util"] * 100)
        if h["util"] >= 0.90:
            st.error(f"{h['name']} - CRITICAL ({util_pct}%)")
        elif h["util"] >= 0.75:
            st.warning(f"{h['name']} - HIGH ({util_pct}%)")
        else:
            st.success(f"{h['name']} - STABLE ({util_pct}%)")

    st.subheader("🚑 Resource Allocation")
    if resource_recommendations:
        for item in resource_recommendations:
            st.markdown(
                f"""
                <div style="
                    background:#f8fafc;
                    border-left:6px solid #2563eb;
                    padding:12px 14px;
                    border-radius:10px;
                    margin-bottom:10px;
                ">
                    {item}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Run the LangGraph workflow to generate resource recommendations.")

with right:
    st.subheader("🧠 Agent Workflow")
    st.markdown(
        """
        1. **Signal Node / Agent** → Receives crisis signals  
        2. **Forecast Node / Agent** → Computes risk and operational note  
        3. **Planner Node / Agent** → Recommends top actions  
        4. **Resource Node / Agent** → Chooses deployment priorities  
        5. **Comms Node / Agent (Granite via watsonx.ai)** → Generates hospital and public communication  
        6. **Coordinator Agent (Orchestrate)** → Demonstrates enterprise multi-agent coordination  
        """
    )

    st.subheader("🎯 Recommended Actions")
    if actions:
        for action in actions:
            st.markdown(
                f"""
                <div style="
                    background:#fff7ed;
                    border-left:6px solid #ea580c;
                    padding:12px 14px;
                    border-radius:10px;
                    margin-bottom:10px;
                ">
                    {action}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Run the LangGraph workflow to generate actions.")

    st.subheader("📢 Live AI Generated Communication")
    st.markdown("#### 🏥 Hospital Admin Alert")
    st.write(admin_text if admin_text else "No hospital admin alert available.")

    st.markdown("#### 📣 Public Advisory")
    st.write(public_text if public_text else "No public advisory available.")

    st.subheader("🔔 Slack Alerts")
    st.write(f"Slack alerts enabled: {'Yes' if enable_slack_alerts else 'No'}")
    if st.session_state["alert_log"]:
        st.dataframe(pd.DataFrame(st.session_state["alert_log"]), use_container_width=True)
    else:
        st.info("No alerts sent yet.")

    if st.button("Acknowledge Last Alert"):
        st.session_state["last_alert"] = None
        st.success("Last alert reset. A new HIGH/SEVERE event can alert again.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
st.caption(
    "Built for IBM SkillsBuild AI Experiential Learning Lab | Python simulation | LangGraph workflow | Granite via watsonx.ai | Multi-agent orchestration with watsonx Orchestrate"
)