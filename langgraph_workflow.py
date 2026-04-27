from __future__ import annotations

from typing import TypedDict, List, Optional
import os
import requests

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END


load_dotenv()

IBM_API_KEY = os.getenv("IBM_API_KEY")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID")
IBM_WATSONX_URL = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")


class CrisisState(TypedDict, total=False):
    date: str
    scenario: str
    ems_calls: int
    predicted_surge: int
    heat_risk: bool
    hospital_utilization: Optional[float]
    disaster_flag: int
    risk_level: str
    risk_note: str
    actions: List[str]
    resource_recommendations: List[str]
    generated_text: str


# -----------------------------
# Reuse your existing helper logic
# -----------------------------
def get_risk_level(ems_calls, heat_risk, disaster_flag):
    if disaster_flag == 1 and ems_calls >= 700:
        return "SEVERE", "Multi-system emergency conditions detected"
    if heat_risk and ems_calls >= 500:
        return "HIGH", "Immediate action required"
    if ems_calls >= 300:
        return "MODERATE", "Monitor closely"
    return "LOW", "Normal operations"


def get_ibm_token(api_key: str) -> str:
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}"
    response = requests.post(url, headers=headers, data=data, timeout=30)
    response.raise_for_status()
    return response.json()["access_token"]


def ask_watsonx(prompt: str, token: str, project_id: str, watsonx_url: str) -> str:
    url = f"{watsonx_url}/ml/v1/text/generation?version=2023-05-29"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 260,
            "min_new_tokens": 40,
        },
        "model_id": "ibm/granite-3-8b-instruct",
        "project_id": project_id,
    }
    response = requests.post(url, headers=headers, json=body, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["results"][0]["generated_text"]


def build_live_prompt(state: CrisisState) -> str:
    actions_text = "\n".join([f"- {a}" for a in state.get("actions", [])])
    resources_text = "\n".join([f"- {r}" for r in state.get("resource_recommendations", [])])

    return f"""
You are an emergency response assistant for a city crisis command center.

Scenario: {state['scenario']}

Current crisis situation:
- Date: {state['date']}
- EMS calls: {state['ems_calls']}
- Predicted surge: {state['predicted_surge']}
- Heat risk: {state['heat_risk']}
- Hospital utilization: {state['hospital_utilization']}
- Disaster flag: {state['disaster_flag']}
- Risk level: {state['risk_level']}

Recommended actions:
{actions_text}

Resource deployment priorities:
{resources_text}

Write the output in exactly this format:

HOSPITAL ADMIN ALERT
- bullet 1
- bullet 2
- bullet 3
- bullet 4

PUBLIC ADVISORY
- bullet 1
- bullet 2
- bullet 3
- bullet 4

Rules:
- keep it short, practical, and realistic
- do not invent unsupported statistics
- use only the situation provided
"""


# -----------------------------
# LangGraph nodes
# -----------------------------
def signal_node(state: CrisisState) -> CrisisState:
    # this node just passes through incoming simulation data
    return state


def forecast_node(state: CrisisState) -> CrisisState:
    risk_level, risk_note = get_risk_level(
        state["ems_calls"],
        state["heat_risk"],
        state["disaster_flag"]
    )
    state["risk_level"] = risk_level
    state["risk_note"] = risk_note
    return state


def planner_node(state: CrisisState) -> CrisisState:
    actions: List[str] = []

    if state["scenario"] == "Heatwave Surge":
        actions = [
            "Deploy cooling centers in high-density zones",
            "Increase EMS coverage in hotspot areas",
            "Stock IV fluids and heat-response supplies",
        ]
    elif state["scenario"] == "Flood Disaster":
        actions = [
            "Pre-stage rescue teams and ambulances near flood corridors",
            "Open temporary triage and shelter sites",
            "Coordinate patient transfer routes with public safety agencies",
        ]
    elif state["scenario"] == "Disease Outbreak":
        actions = [
            "Expand isolation and triage capacity",
            "Trigger surge staffing for ED and ICU",
            "Issue public escalation guidance",
        ]
    else:
        actions = [
            "Maintain standard EMS staffing",
            "Continue routine monitoring",
            "Keep surge assets on standby",
        ]

    state["actions"] = actions
    return state


def resource_node(state: CrisisState) -> CrisisState:
    resources: List[str] = []

    if state["scenario"] == "Heatwave Surge":
        resources = [
            "Deploy cooling centers",
            "Increase EMS unit coverage",
            "Pre-position hydration and cooling kits",
        ]
    elif state["scenario"] == "Flood Disaster":
        resources = [
            "Pre-stage rescue boats and ambulances",
            "Open emergency shelters",
            "Protect hospital access routes",
        ]
    elif state["scenario"] == "Disease Outbreak":
        resources = [
            "Expand isolation areas",
            "Stock PPE and testing kits",
            "Increase ICU staffing",
        ]
    else:
        resources = [
            "Keep baseline resources active",
            "Maintain routine coordination",
            "Monitor for changes",
        ]

    state["resource_recommendations"] = resources
    return state


def comms_node(state: CrisisState) -> CrisisState:
    if not IBM_API_KEY or not IBM_PROJECT_ID:
        state["generated_text"] = "IBM credentials missing."
        return state

    prompt = build_live_prompt(state)
    token = get_ibm_token(IBM_API_KEY)
    generated_text = ask_watsonx(prompt, token, IBM_PROJECT_ID, IBM_WATSONX_URL)
    state["generated_text"] = generated_text
    return state


# -----------------------------
# Build graph
# -----------------------------
def build_crisis_graph():
    graph = StateGraph(CrisisState)

    graph.add_node("signal", signal_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("planner", planner_node)
    graph.add_node("resource", resource_node)
    graph.add_node("comms", comms_node)

    graph.set_entry_point("signal")
    graph.add_edge("signal", "forecast")
    graph.add_edge("forecast", "planner")
    graph.add_edge("planner", "resource")
    graph.add_edge("resource", "comms")
    graph.add_edge("comms", END)

    return graph.compile()


def run_crisis_graph(initial_state: CrisisState) -> CrisisState:
    app = build_crisis_graph()
    return app.invoke(initial_state)


if __name__ == "__main__":
    demo_state: CrisisState = {
        "date": "2025-12-31",
        "scenario": "Heatwave Surge",
        "ems_calls": 686,
        "predicted_surge": 823,
        "heat_risk": True,
        "hospital_utilization": None,
        "disaster_flag": 0,
    }

    result = run_crisis_graph(demo_state)
    print(result)