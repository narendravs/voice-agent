# CarePulse LiveKit Voice Agent

A real-time, bi-directional conversational AI voice agent built on **LiveKit (WebRTC)**, **OpenAI (`gpt-4o-mini`)**, **AssemblyAI STT**, and **Fish Audio TTS**, integrated with a **Next.js LangGraph** backend API. This service handles live phone/WebRTC medical appointment scheduling, queries real-time doctor availability, and manages structured patient booking workflows.

> [!WARNING]
> **Security Note:** Ensure `.env.local` is added to `.gitignore` and never committed to version control. Rotate any exposed API keys immediately if they were pasted publicly.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Voice & AI Stack](#voice--ai-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Agent Tools & Conversation Flow](#agent-tools--conversation-flow)
- [Running & Testing the Agent](#running--testing-the-agent)
- [Docker Deployment](#docker-deployment)
- [Automated Testing & Linting](#automated-testing--linting)

---

## Architecture Overview

```text
[ Patient (WebRTC / Phone / App) ]
               │
               ▼ (Real-time Audio Stream)
[ LiveKit Cloud / RTC Server ]
               │
               ▼ (STT -> TurnDetector -> LLM -> TTS)
[ Python Voice Agent (`src/agent.py`) ] ──(HTTP POST via aiohttp)──► [ Next.js LangGraph API Backend ]
               │                                                                 │
               ├──► Tool: get_available_doctors()                                ├──► Appwrite Database
               └──► Tool: schedule_appointment()                                 └──► Pinecone Vector Store
```

### Voice & AI Stack

| Layer                                | Technology                                                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| **WebRTC Transport & Orchestration** | LiveKit Agents SDK (`AgentServer`, `AgentSession`)                                                           |
| **Speech-to-Text (STT)**             | AssemblyAI (`assemblyai/universal-3-6-pro`)                                                                  |
| **LLM Reasoning Engine**             | OpenAI (`openai/gpt-4o-mini`) with function calling                                                          |
| **Text-to-Speech (TTS)**             | Fish Audio (`fishaudio/s2.1-pro`)                                                                            |
| **Turn Detection**                   | LiveKit semantic + acoustic `TurnDetector` with adaptive interruption handling                               |
| **Backend Bridge**                   | Asynchronous `aiohttp` client posting room-scoped sessions (`sessionId`) to the CarePulse LangGraph endpoint |

---

## Project Structure

```text
voice-agent/
├── src/
│   ├── __init__.py
│   └── agent.py            # LiveKit voice pipeline, environment router & function tools
├── tests/
│   └── test_agent.py       # Unit and integration tests
├── .env.example            # Template for environment variables
├── .env.local              # Local environment secrets (git-ignored)
├── .dockerignore           # Docker build exclusions
├── .gitignore              # Git exclusions
├── Dockerfile              # Production container image definition
├── livekit.toml            # LiveKit CLI & Cloud deployment config
├── pyproject.toml          # Project metadata & dependencies
├── scenarios.yaml          # LiveKit simulation & evaluation scenarios
├── taskfile.yaml           # Task runner definitions
├── uv.lock                 # Locked dependency tree (Astral uv)
└── README.md               # Project documentation
```

---

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or standard `pip` / `venv`
- **LiveKit Cloud** project (or self-hosted LiveKit server) credentials
- **API Keys:** OpenAI, AssemblyAI, and Fish Audio (or LiveKit Inference credits)
- **CarePulse Backend:** Running local Next.js server (`localhost:3000`) or deployed Vercel instance

---

## Installation & Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/narendravs/voice-agent.git
cd voice-agent
```

### 2. Install Dependencies

Using `uv` (recommended):

```powershell
uv sync
```

Or using standard Python `venv`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

### 3. Configure Environment Variables

Create a `.env.local` file in the project root:

```powershell
copy .env.example .env.local
```

Add your configuration to `.env.local`:

```env
# LiveKit Credentials
LIVEKIT_API_KEY="your_livekit_api_key"
LIVEKIT_API_SECRET="your_livekit_api_secret"
LIVEKIT_URL="wss://your-project.livekit.cloud"

# AI & Inference Provider Keys
OPENAI_API_KEY="your_openai_api_key"
GROQ_API_KEY="your_groq_api_key"
GOOGLE_API_KEY="your_google_api_key"

# LangGraph Backend Endpoints
NEXT_PUBLIC_PROD_LANGGRAPH_AGENT_ENDPOINT="https://your-app.vercel.app/api/aiagents/voiceAgent"
NEXT_PUBLIC_LOCAL_LANGGRAPH_AGENT_ENDPOINT="http://localhost:3000/api/aiagents/voiceAgent"

# Environment Switch: "local" or "production"
NEXT_PUBLIC_LANGGRAPH_ENV="local"

# Vercel Deployment Protection Bypass (if enabled on preview/prod)
VERCEL_PROTECTION_BYPASS_SECRET="your_vercel_bypass_secret"
```

---

## Agent Tools & Conversation Flow

The voice assistant (`Assistant` in `src/agent.py`) enforces strict guardrails to prevent hallucinated doctor names or premature bookings.

### 1. `get_available_doctors()`

- Triggered immediately whenever a caller asks which doctors are available, or mentions a doctor's name before verification.
- Queries the backend database to retrieve real doctor names and specialties.

### 2. `schedule_appointment(patient_name, doctor_name, appointment_time, reason, additional_info)`

- Collects four required details sequentially (one question at a time):
  1. Patient's full name
  2. Verified doctor name
  3. Preferred date and time (converted to `YYYY-MM-DD HH:MM` 24-hour format)
  4. Reason for visit / symptoms
- Submits the booking payload to the LangGraph API and relays confirmation or follow-up requests (such as phone/email) back to the caller.

---

## Running & Testing the Agent

### Local Development Mode

Connects to LiveKit Cloud and listens for incoming room dispatches (`agent_name="voice-agent"`):

```bash
lk agent dev
```

Alternatively, run the agent directly:

```bash
uv run python src/agent.py dev
```

**How to test development sessions:**

1. Start the local development agent using the command above.
2. Open the LiveKit **Development Agent Console**.
3. Click **Start Session** to test live two-way audio with your local worker.

### Production Worker Mode

Create and deploy the worker to LiveKit Cloud using the LiveKit CLI:

```bash
lk agent create
lk agent deploy
```

**How to test production sessions:**

1. Ensure your production deployment is active on LiveKit Cloud.
2. Open the LiveKit **Production Agent Console**.
3. Click **Start Session** to initiate a test call with the deployed production agent.

---

## Docker Deployment

Build and run the containerized worker for custom cloud hosting (Render, AWS ECS, or Google Cloud Run):

```bash
# Build image
docker build -t carepulse-voice-agent .

# Run container with local environment file
docker run --env-file .env.local carepulse-voice-agent
```

---

## Automated Testing & Linting

Run automated tests and code quality checks:

```bash
# Run pytest suite
uv run pytest tests/

# Run Ruff linter
uv run ruff check .
```
