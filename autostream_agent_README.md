# AutoStream AI Agent 🎬

A production-ready Conversational AI Agent for **AutoStream** — a fictional SaaS company offering automated video editing tools for content creators.

Built as part of the **ServiceHive / Inflx Machine Learning Intern Assignment**.

---

## ✨ Features

| Capability | Details |
|---|---|
| Intent Detection | Classifies: Greeting · Inquiry · High-Intent |
| RAG Pipeline | Retrieves answers from a local JSON knowledge base |
| Lead Capture | Collects Name → Email → Platform, then calls `mock_lead_capture()` |
| State Management | Full conversation history across all turns via LangGraph state |
| LLM | Claude 3 Haiku (Anthropic) |

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Or create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 5. Run the agent

```bash
python main.py
```

---

## 💬 Example Conversation

```
Aria: Hi there! 👋 I'm Aria, your AutoStream assistant.

You: Hi, what plans do you offer?

Aria: We have two plans:
• Basic Plan — $29/month: 10 videos/month, 720p, email support.
• Pro Plan — $79/month: Unlimited videos, 4K, AI captions, 24/7 support.

You: That sounds great, I want to try the Pro plan for my YouTube channel.

Aria: That's exciting! 🎉 Could you tell me your full name?

You: Rahul Sharma

Aria: Thanks, Rahul! What's your email address?

You: rahul@example.com

Aria: Which creator platform are you primarily on?

You: YouTube

Aria: 🚀 You're all set, Rahul! I've started your Pro Plan free trial...
```

---

## 🏗️ Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides a **deterministic finite-state machine** model — critical for a lead-capture flow where each step (name → email → platform → capture) must happen in strict sequence. AutoGen's multi-agent conversation style is better suited for open-ended research tasks where agents negotiate; here we need precise, predictable stage transitions.

### How State is Managed

The `AgentState` TypedDict holds:

- **`messages`** — the complete conversation history (all user + assistant turns) passed to the LLM on every invocation, giving the model full context.
- **`stage`** — controls which node runs next (`chat`, `collect_name`, `collect_email`, `collect_platform`, `capture_lead`).
- **`name`, `email`, `platform`** — progressively filled as the user provides them.
- **`lead_captured`** — a flag preventing duplicate tool calls.

Each graph node reads the state, produces a response, and returns an updated state dict. Routing functions (`route_after_*`) inspect `stage` to determine the next node. This architecture scales to 5–10 turn conversations with zero memory loss because the full history is always preserved in `state["messages"]`.

---

## 📱 WhatsApp Integration via Webhooks

To deploy this agent on WhatsApp, the following approach would be used:

### Overview

```
WhatsApp User
     │
     ▼
WhatsApp Business API (Meta Cloud API)
     │  POST /webhook  (message event)
     ▼
FastAPI / Flask Webhook Server
     │  parse sender_id + message text
     ▼
process_user_message(text, session_state)
     │  LangGraph Agent
     ▼
Reply text
     │  POST https://graph.facebook.com/v18.0/{phone_id}/messages
     ▼
WhatsApp User receives reply
```

### Implementation Steps

1. **Register a WhatsApp Business account** on Meta for Developers and obtain a `PHONE_NUMBER_ID` and `ACCESS_TOKEN`.

2. **Create a webhook endpoint** (e.g. with FastAPI):

```python
from fastapi import FastAPI, Request
import httpx, json

app = FastAPI()
sessions = {}   # phone_number → AgentState

@app.post("/webhook")
async def receive_message(req: Request):
    body = await req.json()
    entry = body["entry"][0]["changes"][0]["value"]
    msg = entry["messages"][0]
    phone = msg["from"]
    text = msg["text"]["body"]

    state = sessions.get(phone, copy.deepcopy(INITIAL_STATE))
    reply, new_state = process_user_message(text, state)
    sessions[phone] = new_state

    await send_whatsapp_message(phone, reply)
    return {"status": "ok"}

async def send_whatsapp_message(to: str, text: str):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    payload = {"messaging_product": "whatsapp", "to": to,
                "type": "text", "text": {"body": text}}
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, headers=headers)
```

3. **Persist sessions** — Replace the in-memory `sessions` dict with Redis (e.g. `redis-py`) to survive server restarts and support horizontal scaling.

4. **Deploy** the webhook to a public HTTPS URL (Railway, Render, or AWS Lambda) and register it in the Meta Developer dashboard under **Webhooks → messages**.

5. **Verify webhook** — Meta sends a `GET` request with `hub.challenge`; your server must echo it back to confirm ownership.

---

## 📁 Project Structure

```
autostream-agent/
├── main.py                         # CLI entrypoint
├── requirements.txt
├── README.md
├── knowledge_base/
│   └── autostream_kb.json          # RAG knowledge base
├── agent/
│   ├── __init__.py
│   ├── agent.py                    # LangGraph state machine + nodes
│   └── rag_pipeline.py             # Knowledge base loader & retriever
└── tools/
    ├── __init__.py
    └── tools.py                    # mock_lead_capture()
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (required) |

---

## 📄 License

MIT — free to use and modify.
