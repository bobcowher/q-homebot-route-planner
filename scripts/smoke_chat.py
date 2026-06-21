"""Live smoke for the chat session: drives the REAL planner (LLMClient + ollama,
NavigatorTool over run 314) with a scripted reader, to prove cross-turn
persistence -- the second utterance relies on context/state from the first.

Needs the LLM serving:  ollama serve  (qwen2.5:14b-instruct)
Run:  python scripts/smoke_chat.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.navigator_tool import NavigatorTool
from planner.llm_client import LLMClient
from planner.agent_loop import PlannerAgent
from planner.chat import ChatSession

# Two turns that only make sense together: fetch, then deliver what was fetched.
SCRIPT = ["Go to the fridge and grab a drink.",
          "Great, now bring it over to me.",
          "quit"]


def main():
    lines = iter(SCRIPT)

    def read(_prompt=""):
        line = next(lines, None)
        if line is not None:
            print(f"\nyou> {line}")
        return line

    nav = NavigatorTool()
    agent = PlannerAgent(LLMClient(), nav)
    ChatSession(agent, nav, read=read, speak=lambda t: print(f"ROBOT: {t}")).start()
    print(f"\nFINAL STATE: {nav.state()}")


if __name__ == "__main__":
    main()
