# autostream-ai-agent
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
