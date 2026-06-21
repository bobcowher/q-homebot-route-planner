"""Closed-loop ReAct planner. handle_utterance(text) -> spoken_response over a
persistent message history. Voice is adapters outside this class: ASR feeds text
in, TTS speaks the returned response."""
import json

from planner.llm_client import SYSTEM_PROMPT

MAX_TOOL_CALLS = 12


class PlannerAgent:
    def __init__(self, client, navigator, system_prompt=SYSTEM_PROMPT,
                 max_tool_calls=MAX_TOOL_CALLS, trace=None):
        self.client = client
        self.nav = navigator
        self.max_tool_calls = max_tool_calls
        self.system_prompt = system_prompt
        # trace(name, arguments, result) fires after each tool call, for
        # observability (console printout + chat log). None = silent.
        self.trace = trace
        self.conversation: list[dict] = [
            {"role": "system", "content": system_prompt}]

    def reset(self) -> None:
        """Clear the conversation back to the system prompt (new chat session).
        Pairs with NavigatorTool.reset(), which starts a fresh world episode."""
        self.conversation = [{"role": "system", "content": self.system_prompt}]

    def handle_utterance(self, text: str) -> str:
        self.conversation.append({"role": "user", "content": text})
        for _ in range(self.max_tool_calls):
            resp = self.client.chat(self.conversation)
            if not resp["tool_calls"]:
                self.conversation.append(
                    {"role": "assistant", "content": resp["text"]})
                return resp["text"]
            # Record the assistant's tool-call turn, then each tool result.
            self.conversation.append({
                "role": "assistant",
                "content": resp.get("text"),
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(tc["arguments"])}}
                    for tc in resp["tool_calls"]],
            })
            for tc in resp["tool_calls"]:
                result = self._dispatch(tc)
                if self.trace:
                    self.trace(tc["name"], tc.get("arguments") or {}, result)
                self.conversation.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "content": json.dumps(result)})
        msg = "I couldn't complete that within the step budget."
        self.conversation.append({"role": "assistant", "content": msg})
        return msg

    def _dispatch(self, tc: dict) -> dict:
        """Validate a tool call and run it. A live LLM can emit an unknown tool or
        a go_to with no/invalid destination; return an error result (fed back to
        the model) rather than crashing the run so the loop can recover."""
        if tc["name"] != "go_to":
            return {"error": f"unknown tool {tc['name']!r}; only go_to is available"}
        destination = tc.get("arguments", {}).get("destination")
        if not destination:
            return {"error": "go_to requires a 'destination' argument "
                             "(one of: fridge, human, door, trash)"}
        return self.nav.go_to(destination)
