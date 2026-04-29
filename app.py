import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

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
NOAA_TOKEN = os.getenv("NOAA_TOKEN")
NOAA_STATION_ID = os.getenv("NOAA_STATION_ID")
HHS_HOSPITAL_API = os.getenv(
    "HHS_HOSPITAL_API",
    "https://healthdata.gov/resource/g62h-syeh.json"
)


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


def clean_generated_text(text: str) -> str:
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = []
    skip_prefixes = [
        "use urgent language",
        "all points must be realistically answerable within context",
        "output:",
        "instructions:",
        "note:",
        "rationalize recommendations",
    ]

    for line in lines:
        lower = line.lower().strip("•- ")
        if any(lower.startswith(prefix) for prefix in skip_prefixes):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


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
# Live data helpers
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_ems_calls(days_back: int = 30):
    try:
        end_dt = datetime.now(timezone.utc).date()
        start_dt = end_dt - timedelta(days=days_back)

        url = "https://data.sfgov.org/resource/nuek-vuh3.json"
        params = {
            "$select": "count(*)",
            "$where": (
                "call_type = 'Medical Incident' "
                f"AND call_date >= '{start_dt.isoformat()}' "
                f"AND call_date <= '{end_dt.isoformat()}'"
            )
        }

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        count_val = int(data[0]["count"]) if data else 0
        daily_avg = max(1, int(count_val / days_back)) if count_val > 0 else 0
        return {"ems_calls": daily_avg, "source": "live"}
    except Exception as e:
        return {"ems_calls": None, "source": f"failed: {e}"}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_fema_flag():
    try:
        url = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
        params = {
            "$filter": "state eq 'CA'",
            "$top": 20,
            "$format": "json"
        }

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        records = data.get("DisasterDeclarationsSummaries", [])
        return {"disaster_flag": 1 if len(records) > 0 else 0, "source": "live"}
    except Exception as e:
        return {"disaster_flag": None, "source": f"failed: {e}"}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_weather():
    try:
        if not NOAA_TOKEN or not NOAA_STATION_ID:
            return {"heat_risk": None, "source": "missing NOAA config"}

        today = datetime.now(timezone.utc).date()
        start_date = (today - timedelta(days=3)).isoformat()
        end_date = today.isoformat()

        url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
        headers = {"token": NOAA_TOKEN}
        params = {
            "datasetid": "GHCND",
            "stationid": NOAA_STATION_ID,
            "startdate": start_date,
            "enddate": end_date,
            "datatypeid": "TMAX",
            "limit": 10,
            "units": "metric",
            "sortfield": "date",
            "sortorder": "desc",
        }

        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        data = response.json().get("results", [])

        if not data:
            return {"heat_risk": None, "source": "no NOAA rows"}

        latest_temp = data[0]["value"]
        return {"heat_risk": latest_temp >= 30, "source": "live"}
    except Exception as e:
        return {"heat_risk": None, "source": f"failed: {e}"}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_hospital_utilization():
    try:
        url = HHS_HOSPITAL_API
        params = {
            "$select": "date",
            "$where": "state = 'CA' AND date IS NOT NULL",
            "$order": "date DESC",
            "$limit": 1,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        rows = response.json()

        if not rows:
            return {"hospital_utilization": None, "source": "no CA hospital rows"}

        latest_date = rows[0].get("date", "")[:10]
        return {
            "hospital_utilization": None,
            "source": f"feed connected ({latest_date})"
        }
    except Exception as e:
        return {"hospital_utilization": None, "source": f"failed: {e}"}


@st.cache_data(ttl=300, show_spinner=False)
def build_live_state(base_situation: dict):
    live_ems = fetch_live_ems_calls(days_back=30)
    live_fema = fetch_live_fema_flag()
    live_weather = fetch_live_weather()
    live_hospital = fetch_live_hospital_utilization()

    ems_calls = live_ems["ems_calls"] if live_ems["ems_calls"] is not None else int(base_situation.get("ems_calls", 0))
    disaster_flag = live_fema["disaster_flag"] if live_fema["disaster_flag"] is not None else int(base_situation.get("disaster_flag", 0))
    heat_risk = live_weather["heat_risk"] if live_weather["heat_risk"] is not None else bool(base_situation.get("heat_risk", False))
    hospital_utilization = (
        live_hospital["hospital_utilization"]
        if live_hospital["hospital_utilization"] is not None
        else base_situation.get("hospital_utilization", None)
    )

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "ems_calls": ems_calls,
        "heat_risk": heat_risk,
        "hospital_utilization": hospital_utilization,
        "disaster_flag": disaster_flag,
        "live_sources": {
            "ems": live_ems["source"],
            "fema": live_fema["source"],
            "weather": live_weather["source"],
            "hospital": live_hospital["source"]
        }
    }


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
# Session state
# -----------------------------
if "last_alert" not in st.session_state:
    st.session_state["last_alert"] = None
if "alert_log" not in st.session_state:
    st.session_state["alert_log"] = []
if "graph_result" not in st.session_state:
    st.session_state["graph_result"] = None
if "live_state" not in st.session_state:
    st.session_state["live_state"] = None


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Simulation Controls")
live_mode = st.sidebar.checkbox("Use Live Public Data", value=False)

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Heatwave Surge", "Flood Disaster", "Disease Outbreak", "Normal Day"]
)

severity = 3 if live_mode else st.sidebar.slider("Scenario Severity", 1, 5, 3)

load_live_data = st.sidebar.button("Load Live Data")
run_langgraph = st.sidebar.button(
    "Run AI Analysis",
    disabled=(live_mode and st.session_state["live_state"] is None)
)

with st.sidebar.expander("Advanced Controls"):
    enable_slack_alerts = st.checkbox("Enable Slack alerts", value=False)
    manual_test_alert = st.button("Send Test Slack Alert")
    refresh_live_data = st.button("Refresh Live Data")

if refresh_live_data:
    fetch_live_ems_calls.clear()
    fetch_live_fema_flag.clear()
    fetch_live_weather.clear()
    fetch_live_hospital_utilization.clear()
    build_live_state.clear()
    st.session_state["live_state"] = None
    st.session_state["graph_result"] = None
    st.sidebar.success("Live data cache cleared")


# -----------------------------
# Scenario / live-state setup
# -----------------------------
if live_mode and load_live_data:
    st.session_state["graph_result"] = None
    with st.spinner("Loading live data..."):
        st.session_state["live_state"] = build_live_state(base_situation)

if live_mode and st.session_state["live_state"] is not None:
    cached_live = st.session_state["live_state"].copy()

    ems_calls = cached_live["ems_calls"]
    disaster_flag = cached_live["disaster_flag"]
    heat_risk = cached_live["heat_risk"]
    hospital_utilization = cached_live["hospital_utilization"]

    surge_multiplier = 1.15
    if scenario == "Heatwave Surge":
        heat_risk = True
        surge_multiplier = 1.25
    elif scenario == "Flood Disaster":
        disaster_flag = 1
        surge_multiplier = 1.30
    elif scenario == "Disease Outbreak":
        disaster_flag = 1
        surge_multiplier = 1.22
    elif scenario == "Normal Day":
        heat_risk = False
        disaster_flag = 0
        surge_multiplier = 1.0

    predicted_surge = int(ems_calls * surge_multiplier)

    initial_state = {
        "date": cached_live["date"],
        "scenario": scenario,
        "ems_calls": ems_calls,
        "predicted_surge": predicted_surge,
        "heat_risk": heat_risk,
        "hospital_utilization": hospital_utilization,
        "disaster_flag": disaster_flag,
        "live_sources": cached_live.get("live_sources", {})
    }
else:
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

    initial_state = {
        "date": base_situation.get("date", "N/A"),
        "scenario": scenario,
        "ems_calls": ems_calls,
        "predicted_surge": predicted_surge,
        "heat_risk": heat_risk,
        "hospital_utilization": hospital_utilization,
        "disaster_flag": disaster_flag,
    }

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


# -----------------------------
# Run AI workflow on demand
# -----------------------------
if run_langgraph:
    try:
        with st.spinner("Running AI analysis..."):
            graph_result = run_crisis_graph(initial_state)
            st.session_state["graph_result"] = graph_result
    except Exception as e:
        st.sidebar.error(f"AI analysis failed: {e}")

graph_result = st.session_state.get("graph_result")
if graph_result is None:
    graph_result = {
        **initial_state,
        "risk_level": "NOT RUN",
        "risk_note": "Click 'Run AI Analysis' to execute the workflow.",
        "actions": [],
        "resource_recommendations": [],
        "generated_text": "No live AI response yet.",
    }

risk_level = graph_result.get("risk_level", "UNKNOWN")
display_status = "Ready" if risk_level == "NOT RUN" else risk_level
risk_note = graph_result.get("risk_note", "")
actions = graph_result.get("actions", [])
resource_recommendations = graph_result.get("resource_recommendations", [])
generated_text = graph_result.get("generated_text", "No generated text.")
admin_text, public_text = parse_ai_sections(clean_generated_text(generated_text))


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
    <div style="padding: 4px 0 14px 0;">
        <div style="font-size:16px; font-weight:700; color:#ef4444; letter-spacing:0.3px;">
            AI-POWERED EMERGENCY INTELLIGENCE
        </div>
        <div style="font-size:44px; font-weight:900; line-height:1.05; margin-top:6px; color:#0f172a;">
            Crisis Decision Room
        </div>
        <div style="font-size:17px; color:#475569; margin-top:10px;">
            Live EMS, FEMA, and NOAA signals with AI-driven crisis response recommendations
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

top1, top2, top3 = st.columns([1, 1, 2])

with top1:
    if risk_level == "SEVERE":
        status_badge("Risk", display_status, "#991b1b")
    elif risk_level == "HIGH":
        status_badge("Risk", display_status, "#b91c1c")
    elif risk_level == "MODERATE":
        status_badge("Risk", display_status, "#b45309")
    elif risk_level == "NOT RUN":
        status_badge("Status", display_status, "#166534")
    else:
        status_badge("Risk", display_status, "#166534")

with top2:
    status_badge("Scenario", scenario, "#1d4ed8")

with top3:
    if live_mode:
        st.markdown(
            """
            <div style="
                background:#f8fafc;
                border:1px solid #e2e8f0;
                border-radius:14px;
                padding:12px 14px;
                font-size:14px;
                color:#334155;
            ">
                <b>Live Sources</b> · EMS · FEMA · Weather · Hospital feed connected
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.session_state["live_state"] is None:
            st.info("Click 'Load Live Data' in the sidebar to fetch current live signals.")
    else:
        st.markdown(
            """
            <div style="
                background:#f8fafc;
                border:1px solid #e2e8f0;
                border-radius:14px;
                padding:12px 14px;
                font-size:14px;
                color:#334155;
            ">
                <b>Mode</b> · Demo snapshot active
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

metric_cols = st.columns(4)

with metric_cols[0]:
    metric_card("EMS Calls", graph_result.get("ems_calls", "N/A"), "Daily estimated emergency demand")

with metric_cols[1]:
    metric_card("Predicted Surge", graph_result.get("predicted_surge", "N/A"), "Expected near-term pressure")

with metric_cols[2]:
    metric_card("Heat Risk", str(graph_result.get("heat_risk", "N/A")), "Environmental stress level")

with metric_cols[3]:
    hospital_val = graph_result.get("hospital_utilization")
    metric_card(
        "Hospital Utilization",
        "Connected" if hospital_val is None else f"{int(hospital_val * 100)}%",
        "Current hospital capacity pressure"
    )

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

if risk_level in ["SEVERE", "HIGH"]:
    st.markdown(
        f"""
        <div style="
            background:#fef2f2;
            border:1px solid #fecaca;
            color:#991b1b;
            padding:14px 16px;
            border-radius:14px;
            font-weight:700;
            font-size:15px;
        ">
            {display_status} RISK · {risk_note}
        </div>
        """,
        unsafe_allow_html=True
    )
elif risk_level == "MODERATE":
    st.markdown(
        f"""
        <div style="
            background:#fffbeb;
            border:1px solid #fde68a;
            color:#92400e;
            padding:14px 16px;
            border-radius:14px;
            font-weight:700;
            font-size:15px;
        ">
            {display_status} RISK · {risk_note}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <div style="
            background:#f0fdf4;
            border:1px solid #bbf7d0;
            color:#166534;
            padding:14px 16px;
            border-radius:14px;
            font-weight:700;
            font-size:15px;
        ">
            {display_status} · {risk_note}
        </div>
        """,
        unsafe_allow_html=True
    )

summary_driver = "Heat + elevated EMS demand"
if scenario == "Flood Disaster":
    summary_driver = "Disaster signal + elevated EMS demand"
elif scenario == "Disease Outbreak":
    summary_driver = "Public health surge + sustained emergency demand"
elif scenario == "Normal Day":
    summary_driver = "Stable operating conditions"

top_action = actions[0] if actions else "Run AI Analysis to generate recommended actions"

st.markdown(
    f"""
    <div style="
        background:#eff6ff;
        border:1px solid #bfdbfe;
        border-radius:14px;
        padding:14px 16px;
        margin-top:12px;
        margin-bottom:8px;
    ">
        <div style="font-size:15px; font-weight:800; color:#1d4ed8; margin-bottom:8px;">
            AI Summary
        </div>
        <div style="font-size:14px; color:#1e3a8a; line-height:1.7;">
            <b>Current status:</b> {display_status}<br>
            <b>Primary driver:</b> {summary_driver}<br>
            <b>Top recommendation:</b> {top_action}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Live Demand Trend")
    st.line_chart(trend_df.set_index("Hour"))

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    st.subheader("Hospital Status")
    if live_mode:
        util = graph_result.get("hospital_utilization")
        if util is None:
            st.info("Hospital feed connected")
        elif util >= 0.90:
            st.error(f"California hospital utilization · CRITICAL ({int(util * 100)}%)")
        elif util >= 0.75:
            st.warning(f"California hospital utilization · HIGH ({int(util * 100)}%)")
        else:
            st.success(f"California hospital utilization · STABLE ({int(util * 100)}%)")
    else:
        for h in hospitals:
            util_pct = int(h["util"] * 100)
            if h["util"] >= 0.90:
                st.error(f"{h['name']} · CRITICAL ({util_pct}%)")
            elif h["util"] >= 0.75:
                st.warning(f"{h['name']} · HIGH ({util_pct}%)")
            else:
                st.success(f"{h['name']} · STABLE ({util_pct}%)")

with right:
    st.subheader("Key Recommended Actions")
    if actions:
        for action in actions[:3]:
            st.markdown(
                f"""
                <div style="
                    background:#fff7ed;
                    border-left:6px solid #ea580c;
                    padding:12px 14px;
                    border-radius:12px;
                    margin-bottom:10px;
                    color:#7c2d12;
                    font-weight:600;
                ">
                    {action}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div style="
                background:#f8fafc;
                border:1px solid #e2e8f0;
                border-radius:12px;
                padding:14px;
                color:#334155;
                font-weight:600;
            ">
                Click <b>Run AI Analysis</b> to generate response actions.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    st.subheader("Executive Alert")
    st.write(admin_text if admin_text else "No executive alert available.")

    st.subheader("Citizen Advisory")
    st.write(public_text if public_text else "No citizen advisory available.")

with st.expander("Technical Workflow"):
    st.markdown(
        """
        1. Signal Agent receives live crisis inputs  
        2. Forecast Agent evaluates risk  
        3. Planner Agent recommends actions  
        4. Resource Agent prioritizes deployment  
        5. Comms Agent generates alerts and advisories  
        """
    )

with st.expander("Resource Allocation Details"):
    if resource_recommendations:
        for item in resource_recommendations:
            st.markdown(f"- {item}")
    else:
        st.info("No resource recommendations yet.")

with st.expander("Alert Log"):
    if st.session_state["alert_log"]:
        st.dataframe(st.session_state["alert_log"], use_container_width=True)
    else:
        st.info("No alerts sent yet.")

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
st.caption("IBM SkillsBuild | Live public signals | Multi-agent crisis response demo")