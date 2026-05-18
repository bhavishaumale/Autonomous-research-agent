# Autonomous Research Agent

An autonomous AI agent that fetches, summarises, and delivers weekly AI research paper digests.

## What it does
- Fetches latest papers from arXiv across configurable AI topics every week
- Deduplicates using FAISS vector memory — never summarises the same paper twice
- Summarises each paper using Gemini 2.0 Flash (free) into plain English
- Human-in-the-loop approval step before sending
- Delivers a formatted digest to your email every Monday
- Streamlit dashboard to run manually and search past papers

## Tech Stack
Python · LangGraph · FAISS · Google Gemini API · arXiv SDK · Streamlit · APScheduler

## Architecture

```
fetch → filter → summarise → compile → approve → send email
```
Built as a LangGraph state machine with 6 nodes and a conditional edge at the approval step.

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/bhavishaumale/Autonomous-research-agent.git
cd Autonomous-research-agent
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file:
GEMINI_API_KEY=your_gemini_key
EMAIL_FROM=yourgmail@gmail.com
EMAIL_TO=yourgmail@gmail.com
GMAIL_APP_PASSWORD=your_app_password

### 5. Run the agent
```bash
# Run once manually
python main.py

# Launch Streamlit UI
streamlit run app.py

# Run weekly scheduler
python scheduler.py
```

## Project Structure

```
├── agent/
│   ├── graph.py      # LangGraph state machine
│   ├── tools.py      # arXiv fetcher + Gemini summariser
│   ├── memory.py     # FAISS vector memory
│   └── prompts.py    # LLM prompt templates
├── digest/
│   └── emailer.py    # Gmail SMTP email sender
├── app.py            # Streamlit dashboard
├── main.py           # Entry point
└── scheduler.py      # APScheduler weekly cron
```

## Key Design Decisions
- **LangGraph over plain functions** — enables conditional branching and human-in-the-loop
- **FAISS over ChromaDB** — no C++ compiler needed, works on all platforms
- **Gemini over OpenAI** — free tier sufficient for summarisation workload
- **Human approval node** — responsible AI deployment, agent doesn't send without confirmation