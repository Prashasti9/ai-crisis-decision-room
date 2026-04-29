Absolutely — here’s a polished **README.md** you can paste into GitHub.

````markdown
# AI Crisis Decision Room

AI Crisis Decision Room is an agentic AI prototype for emergency response coordination. It brings together live public signals, multi-agent reasoning, and IBM watsonx-powered communications to help hospitals and public agencies respond faster during crises such as heatwaves, floods, and disease outbreaks. The system is designed to move from fragmented inputs to coordinated action through a shared crisis state and a five-agent workflow. 

## Overview

In emergency situations, critical signals like EMS demand, weather risk, disaster declarations, and hospital pressure often exist in separate systems. This slows decision-making when time matters most. AI Crisis Decision Room solves this by combining those signals into a single operational picture, running a multi-agent pipeline to assess risk and recommend actions, and generating role-specific communications for both decision-makers and the public. 

## Key Features

- Live public data integrations
  - San Francisco EMS
  - NOAA weather
  - FEMA disaster declarations
  - HHS hospital feed connectivity
- Five-agent LangGraph workflow
  - Signal Agent
  - Forecast Agent
  - Planner Agent
  - Resource Agent
  - Comms Agent
- IBM watsonx.ai integration using Granite for executive alerts and citizen advisories
- IBM watsonx Orchestrate as the coordination layer in the architecture
- Streamlit dashboard for live mode and demo mode
- Optional Slack alerting for high-risk and severe-risk scenarios 

## Architecture

The system is structured as a shared crisis-state pipeline.

1. **Signal Agent** ingests crisis context from live public APIs and scenario input.  
2. **Forecast Agent** evaluates risk and predicts surge.  
3. **Planner Agent** recommends operational actions.  
4. **Resource Agent** prioritizes deployment and response resources.  
5. **Comms Agent** uses IBM Granite on watsonx.ai to generate role-specific communication.  

This approach makes the workflow visible, auditable, and more explainable than a single prompt-based chatbot. 

## Tech Stack

- **Python**
- **Streamlit**
- **LangGraph**
- **IBM watsonx.ai**
- **IBM Granite**
- **IBM watsonx Orchestrate**
- **Public APIs**
  - SF EMS
  - NOAA
  - FEMA
  - HHS
- **Slack Webhooks** for optional escalation alerts :contentReference[oaicite:4]{index=4}

## Project Structure

```text
ai-crisis-decision-room/
├── app.py
├── langgraph_workflow.py
├── notebooks/
│   └── CrisisManagement.ipynb
├── data/
├── outputs/
├── requirements.txt
├── .env.example
└── README.md
````

## How It Works

The application creates a shared crisis state using:

* EMS calls
* predicted surge
* heat risk
* hospital utilization / feed status
* disaster flag

That state is passed through the multi-agent workflow. The dashboard then presents:

* core metrics
* risk level
* recommended actions
* AI-generated executive alerts
* citizen advisories
* optional Slack escalation for high-risk conditions

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Prashasti9/ai-crisis-decision-room.git
cd ai-crisis-decision-room
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a template.

Example:

```env
IBM_API_KEY=your_ibm_api_key
IBM_PROJECT_ID=your_ibm_project_id
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
SLACK_WEBHOOK_URL=your_slack_webhook_url
NOAA_TOKEN=your_noaa_token
NOAA_STATION_ID=USW00023272
```

## Running the Notebook

The notebook is used for:

* data exploration
* feature preparation
* situation generation
* watsonx prompt testing
* producing output JSON files used by the dashboard

Open the notebook:

```bash
jupyter notebook notebooks/CrisisManagement.ipynb
```

Make sure it generates:

* `outputs/situation.json`
* `outputs/watsonx_output.json`

## Running the Dashboard

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Demo Modes

### Demo Mode

Uses prepared situation data to simulate scenarios such as:

* Heatwave Surge
* Flood Disaster
* Disease Outbreak
* Normal Day

### Live Mode

Pulls current public data signals from supported APIs and updates the operational view in the dashboard. 

## IBM Technology Usage

This project uses IBM technologies in three important ways:

* **IBM watsonx.ai** powers reasoning and structured communication generation
* **IBM Granite** generates hospital admin alerts and citizen advisories
* **IBM watsonx Orchestrate** is positioned as the enterprise coordination layer in the overall architecture

## Example Use Case

A heatwave surge scenario may show:

* increased EMS calls
* rising predicted surge
* heat risk activation
* recommended interventions such as cooling centers, increased EMS coverage, and supply preparation
* IBM-generated executive and public-facing communication

## Submission Assets

This project was prepared for the **IBM SkillsBuild AI Experiential Learning Lab** and includes:

* a working prototype
* a pitch deck
* a 3-minute demo script
* written problem and solution statement
* written technology statement

## Security Note

Do not commit:

* `.env`
* API keys
* tokens
* local credentials

Use `.env.example` to show required variables without exposing secrets.

## Future Improvements

* richer live hospital utilization metrics
* additional city integrations
* more advanced forecasting models
* stronger Orchestrate integration
* mobile responder interface
* audit logging and policy-based escalation 

## Team

**The Quantum Guild**

* Prashasti Srivastava
* Kartik Patel
* Shikha Sharma
* Pragnaya Priyadarshini
* Rutuja Rajendra Saste 

## License

This project is for educational and hackathon demonstration purposes.

```

A few small edits before you commit it:
- replace the repo URL if needed
- make sure `.env.example` exists
- remove any section that mentions a file you are not actually uploading

If you want, I can also give you a shorter, more recruiter-friendly README version.
```
