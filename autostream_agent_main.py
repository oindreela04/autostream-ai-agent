"""
main.py
-------
CLI entrypoint for the AutoStream AI Agent.
Run with:  python main.py
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from agent.agent import process_user_message, INITIAL_STATE
import copy


BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AutoStream AI Agent  •  Powered by LangGraph      ║
║        Type  'exit' or 'quit' to end the session         ║
╚══════════════════════════════════════════════════════════╝
"""

WELCOME = (
    "Hi there! 👋 I'm Aria, your AutoStream assistant.\n"
    "I can answer questions about pricing, features, and help you get started.\n"
    "What can I help you with today?"
)


def main():
    print(BANNER)
    print(f"Aria: {WELCOME}\n")

    state = copy.deepcopy(INITIAL_STATE)
    # Seed the welcome message into history
    state["messages"].append({"role": "assistant", "content": WELCOME})
    state["last_response"] = WELCOME

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAria: Thanks for chatting! Goodbye 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
            print("Aria: Thanks for chatting! Goodbye 👋")
            break

        try:
            reply, state = process_user_message(user_input, state)
            print(f"\nAria: {reply}\n")
        except Exception as exc:
            print(f"\n[ERROR] {exc}\n")
            print("Please check your ANTHROPIC_API_KEY and try again.")


if __name__ == "__main__":
    main()
