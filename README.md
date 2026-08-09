
# AutoStream AI Agent 🎬

A production-ready Conversational AI Agent for **AutoStream** — a fictional SaaS company offering automated video editing tools for content creators.

Built as part of the **ServiceHive / Inflx Machine Learning Intern Assignment**.




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
