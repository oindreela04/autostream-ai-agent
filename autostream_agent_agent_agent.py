"""
agent.py
--------
AutoStream Conversational AI Agent built with LangGraph.

State machine:
  CHAT  ──(high intent detected)──►  COLLECT_NAME
                                           │
                                     COLLECT_EMAIL
                                           │
                                    COLLECT_PLATFORM
                                           │
                                    CAPTURE_LEAD  ──► CHAT
"""

import os
import re
from typing import TypedDict, Literal, Optional, List

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.rag_pipeline import FULL_CONTEXT
from tools.tools import mock_lead_capture


# ─────────────────────────────────────────────
# 1.  Agent State
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: List[dict]          # full conversation history
    stage: str                    # current collection stage
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    lead_captured: bool
    last_response: str            # last assistant message (for display)


INITIAL_STATE: AgentState = {
    "messages": [],
    "stage": "chat",
    "name": None,
    "email": None,
    "platform": None,
    "lead_captured": False,
    "last_response": "",
}


# ─────────────────────────────────────────────
# 2.  LLM Setup
# ─────────────────────────────────────────────

def get_llm():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it with:  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=api_key,
        max_tokens=512,
        temperature=0.4,
    )


# ─────────────────────────────────────────────
# 3.  System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are Aria, a friendly and knowledgeable sales assistant for AutoStream — 
an AI-powered automated video editing SaaS platform for content creators.

Your goals:
1. Greet users warmly.
2. Answer product and pricing questions ONLY from the knowledge base below.
3. Detect when a user shows HIGH INTENT (wants to sign up / try / buy / upgrade).
4. When high intent is detected, reply ONLY with the special token:  <<HIGH_INTENT>>

HIGH INTENT signals include phrases like:
- "I want to sign up", "I want to try", "I'd like to buy", "let's do it",
  "sounds good, I'll take it", "I want the Pro plan", "sign me up", "get started"

KNOWLEDGE BASE:
{FULL_CONTEXT}

Rules:
- Always be concise, warm, and helpful.
- Do NOT make up features or prices not in the knowledge base.
- Do NOT ask for personal details yourself; the system will handle that.
- If the user's question is unrelated to AutoStream, politely redirect them.
"""


# ─────────────────────────────────────────────
# 4.  Intent Detection (fast, regex first)
# ─────────────────────────────────────────────

HIGH_INTENT_KEYWORDS = [
    r"\bsign\s*me\s*up\b",
    r"\bi want to (sign up|try|buy|purchase|subscribe|get|start)\b",
    r"\bget started\b",
    r"\bi('ll| will) (take|go with|try|buy)\b",
    r"\bsounds good.{0,20}(i want|i'(ll|d)|let'?s)\b",
    r"\bready to (buy|start|subscribe|try)\b",
    r"\blet'?s do it\b",
    r"\bupgrade\b.*\bpro\b",
    r"\bpro plan\b.*\b(my|for my|want|try)\b",
]

def quick_intent_check(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in HIGH_INTENT_KEYWORDS)


def is_high_intent_llm(state: AgentState, user_text: str) -> bool:
    """Ask LLM to classify intent, as a fallback / confirmation."""
    llm = get_llm()
    classify_prompt = (
        "Classify the intent of the following user message for AutoStream "
        "(video editing SaaS). "
        "Reply with ONLY one word: GREETING, INQUIRY, or HIGH_INTENT.\n\n"
        f"Message: {user_text}"
    )
    result = llm.invoke([HumanMessage(content=classify_prompt)])
    return "HIGH_INTENT" in result.content.upper()


# ─────────────────────────────────────────────
# 5.  Graph Nodes
# ─────────────────────────────────────────────

def node_chat(state: AgentState) -> AgentState:
    """General conversation node — answers questions using RAG context."""
    messages = state["messages"]
    last_user_msg = messages[-1]["content"] if messages else ""

    # Fast path: keyword-based high intent
    if quick_intent_check(last_user_msg):
        return {**state, "stage": "collect_name"}

    # Build LangChain message list
    lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    llm = get_llm()
    response = llm.invoke(lc_messages)
    reply = response.content.strip()

    # LLM-level high intent detection via special token
    if "<<HIGH_INTENT>>" in reply:
        return {**state, "stage": "collect_name"}

    new_messages = messages + [{"role": "assistant", "content": reply}]
    return {**state, "messages": new_messages, "last_response": reply}


def node_collect_name(state: AgentState) -> AgentState:
    messages = state["messages"]

    # If this node was just entered (no name yet), ask for name
    if state["name"] is None:
        # Check if previous user message already contains a name hint
        last_msg = messages[-1]["content"] if messages else ""
        reply = (
            "That's exciting! 🎉 I'd love to get you set up.\n"
            "First, could you tell me your **full name**?"
        )
        new_messages = messages + [{"role": "assistant", "content": reply}]
        return {**state, "messages": new_messages, "stage": "collect_name", "last_response": reply}

    # Name already collected, move on
    return {**state, "stage": "collect_email"}


def node_collect_email(state: AgentState) -> AgentState:
    messages = state["messages"]

    if state["email"] is None:
        reply = f"Thanks, {state['name']}! 😊 What's the best **email address** to reach you at?"
        new_messages = messages + [{"role": "assistant", "content": reply}]
        return {**state, "messages": new_messages, "stage": "collect_email", "last_response": reply}

    return {**state, "stage": "collect_platform"}


def node_collect_platform(state: AgentState) -> AgentState:
    messages = state["messages"]

    if state["platform"] is None:
        reply = (
            "Almost done! Which **creator platform** are you primarily on?\n"
            "(e.g. YouTube, Instagram, TikTok, Twitter/X, etc.)"
        )
        new_messages = messages + [{"role": "assistant", "content": reply}]
        return {**state, "messages": new_messages, "stage": "collect_platform", "last_response": reply}

    return {**state, "stage": "capture_lead"}


def node_capture_lead(state: AgentState) -> AgentState:
    """Call the mock lead capture tool and confirm to the user."""
    result = mock_lead_capture(
        name=state["name"],
        email=state["email"],
        platform=state["platform"],
    )

    reply = (
        f"🚀 You're all set, **{state['name']}**!\n\n"
        f"I've started your **Pro Plan free trial** for your {state['platform']} channel.\n"
        f"A confirmation email is on its way to **{state['email']}**.\n\n"
        f"Your lead ID is `{result['lead_id']}`. Our team will be in touch within 24 hours. "
        f"Welcome to AutoStream! 🎬"
    )

    messages = state["messages"] + [{"role": "assistant", "content": reply}]
    return {
        **state,
        "messages": messages,
        "stage": "chat",
        "lead_captured": True,
        "last_response": reply,
    }


# ─────────────────────────────────────────────
# 6.  Routing Logic
# ─────────────────────────────────────────────

def route_after_chat(state: AgentState) -> str:
    return state["stage"]  # "chat" or "collect_name"


def route_after_collect_name(state: AgentState) -> str:
    return state["stage"]


def route_after_collect_email(state: AgentState) -> str:
    return state["stage"]


def route_after_collect_platform(state: AgentState) -> str:
    return state["stage"]


# ─────────────────────────────────────────────
# 7.  Build the Graph
# ─────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("chat", node_chat)
    g.add_node("collect_name", node_collect_name)
    g.add_node("collect_email", node_collect_email)
    g.add_node("collect_platform", node_collect_platform)
    g.add_node("capture_lead", node_capture_lead)

    g.set_entry_point("chat")

    g.add_conditional_edges(
        "chat",
        route_after_chat,
        {
            "chat": END,
            "collect_name": "collect_name",
        },
    )

    g.add_conditional_edges(
        "collect_name",
        route_after_collect_name,
        {
            "collect_name": END,
            "collect_email": "collect_email",
        },
    )

    g.add_conditional_edges(
        "collect_email",
        route_after_collect_email,
        {
            "collect_email": END,
            "collect_platform": "collect_platform",
        },
    )

    g.add_conditional_edges(
        "collect_platform",
        route_after_collect_platform,
        {
            "collect_platform": END,
            "capture_lead": "capture_lead",
        },
    )

    g.add_edge("capture_lead", END)

    return g.compile()


# ─────────────────────────────────────────────
# 8.  Public API — process_user_message
# ─────────────────────────────────────────────

# Single compiled graph instance (reused across turns)
_graph = build_graph()


def process_user_message(user_input: str, state: AgentState) -> tuple[str, AgentState]:
    """
    Process one user turn.

    Returns:
        (assistant_reply_text, updated_state)
    """
    # Append user message to history
    state = {
        **state,
        "messages": state["messages"] + [{"role": "user", "content": user_input}],
    }

    # If we're in a collection stage, extract the value from user input
    stage = state["stage"]

    if stage == "collect_name" and state["name"] is None:
        state = {**state, "name": user_input.strip()}

    elif stage == "collect_email" and state["email"] is None:
        email = user_input.strip().lower()
        # Basic validation
        if "@" not in email or "." not in email:
            reply = "That doesn't look like a valid email. Could you double-check it?"
            state = {
                **state,
                "messages": state["messages"] + [{"role": "assistant", "content": reply}],
                "last_response": reply,
            }
            return reply, state
        state = {**state, "email": email}

    elif stage == "collect_platform" and state["platform"] is None:
        state = {**state, "platform": user_input.strip()}
        state = {**state, "stage": "capture_lead"}  # trigger capture next

    # Run the graph
    new_state = _graph.invoke(state)

    reply = new_state.get("last_response", "")
    return reply, new_state
