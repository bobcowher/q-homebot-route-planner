# Routing Brain (Planner Layer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A closed-loop LLM planner that takes a natural-language task and accomplishes it by driving the trained navigator through a `go_to(destination)` tool, reacting to each leg's outcome.

**Architecture:** A local text LLM plans over a symbolic world model (perception stub backed by the sim env) and calls a single `go_to` tool per leg in a ReAct loop. Perception and voice are adapters outside the planner — the planner's I/O is `handle_utterance(text) -> spoken_response` over a persistent message history. Reuses the navigator (run 314 champion) via `chained_eval.run_leg` with the `softmax_rel` readout.

**Tech Stack:** Python 3.12, conda env `sac-homebot`, PyTorch, gymnasium, `homebot` env (`HomeBot2D-V1`), `openai` client pointed at a local OpenAI-compatible endpoint (ollama/vLLM on the RTX 5090).

## Global Constraints

- Tool vocabulary is **`go_to(destination)` only** — no pickup/deliver/interaction tools. Env mechanics are taught in the system prompt.
- Control is **closed-loop (ReAct)**: LLM calls `go_to`, receives `{reached, steps, state}`, decides the next call.
- Planner I/O boundary is **`handle_utterance(text) -> spoken_response` over a persistent message history** (voice = adapters outside the planner).
- Perception is a **sim stub** (`WorldModel` over the env); never read pixels in the planner.
- Destination vocabulary: **`fridge`, `human`, `door`, `trash`** (friendly names), mapped to env goal names.
- Navigator: **run 314 champion** `checkpoints/run314_q_model_best.pt`, `goal_layers=2, head_layers=4, use_motion=True`, readout **`softmax_rel` at temp_rel=0.1**.
- Env config matches `chained_eval`: `HomeBot2D-V1`, `obs_resolution=(96,96)`, `n_trash=2`, `max_steps=20000`, `map_name="default"`, `random_start=True`.
- Reuse, do not duplicate: `chained_eval.run_leg` + `REACH_OVERRIDE`, `evaluate.load_q_model` + `process_observation`, `goal_geometry.distance`/`eval_step_budget`, `task_chain.resolve_goal`, `homebot.goals.GOAL_THRESHOLD`, `motion.MotionState`.
- Per-utterance **tool-call budget = 12**.
- All modules run from the repo root (so `import evaluate`, `import chained_eval`, etc. resolve). `planner/` is a package.
- Run tests with `conda run -n sac-homebot python -m pytest ... -v`. Never use `python3 -c` inline.
- New dependency: `openai` (pip) — add to `requirements.txt`.

---

### Task 1: WorldModel (perception stub) + planner package

**Files:**
- Create: `planner/__init__.py` (empty)
- Create: `planner/world_model.py`
- Test: `tests/test_world_model.py`

**Interfaces:**
- Consumes: `task_chain.resolve_goal(base, env_name)`; the env (`base._robot`, `base._task_manager.get_info(robot)`, `base.goal_to_coordinates`).
- Produces:
  - `DEST_TO_ENV: dict[str, str]` — friendly→env goal name.
  - `class WorldModel(env)` with `list_destinations() -> list[str]`, `state() -> dict`, `resolve(name) -> tuple[float, float]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_world_model.py
import gymnasium as gym
import pytest

import homebot  # noqa: F401  (env registration)
from planner.world_model import WorldModel, DEST_TO_ENV


def _env():
    return gym.make("HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
                    obs_resolution=(96, 96), n_trash=2, max_steps=20000,
                    map_name="default", random_start=True)


def test_list_destinations_includes_fixtures_and_trash_when_present():
    env = _env(); env.reset(seed=0)
    w = WorldModel(env)
    dests = w.list_destinations()
    assert {"fridge", "human", "door"} <= set(dests)
    assert "trash" in dests  # n_trash=2 -> trash present right after reset


def test_resolve_known_returns_float_pair_unknown_raises():
    env = _env(); env.reset(seed=0)
    w = WorldModel(env)
    xy = w.resolve("fridge")
    assert len(xy) == 2 and all(isinstance(v, float) for v in xy)
    with pytest.raises(KeyError):
        w.resolve("bogus")


def test_state_exposes_carry_and_trash():
    env = _env(); env.reset(seed=0)
    w = WorldModel(env)
    s = w.state()
    assert {"carrying", "trash_remaining", "robot_xy"} <= set(s)
    assert s["trash_remaining"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python -m pytest tests/test_world_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'planner'`.

- [ ] **Step 3: Write minimal implementation**

```python
# planner/__init__.py
```
(empty file)

```python
# planner/world_model.py
"""Symbolic world model (perception stub) over the sim env.

The seam where real perception drops in later: today it reads the known map and
TaskManager state; tomorrow a VLM populates the same interface. The planner only
ever sees friendly destination names + task state, never pixels.
"""
from task_chain import resolve_goal

# Friendly destination name -> env goal-registry name.
DEST_TO_ENV = {
    "fridge": "go_to_fridge",
    "human": "go_to_human",
    "door": "go_to_door",
    "trash": "collect_trash",
}


class WorldModel:
    def __init__(self, env):
        self.env = env
        self.base = env.unwrapped

    def list_destinations(self) -> list[str]:
        """Currently reachable named targets. Trash only while some remains."""
        dests = ["fridge", "human", "door"]
        if self.base._task_manager.trash_positions:
            dests.append("trash")
        return dests

    def state(self) -> dict:
        base = self.base
        info = base._task_manager.get_info(base._robot)
        return {
            "carrying": base._robot.carrying,
            "trash_remaining": info["trash_remaining"],
            "drink_delivered": info["drink_delivered"],
            "package_delivered": info["package_delivered"],
            "robot_xy": (float(base._robot.x), float(base._robot.y)),
        }

    def resolve(self, name: str) -> tuple[float, float]:
        """Friendly name -> pixel (x, y). Raises KeyError on an unknown name;
        may raise ValueError if the target is currently unavailable (e.g. trash
        gone) — both are surfaced to the LLM by NavigatorTool as a tool error."""
        if name not in DEST_TO_ENV:
            raise KeyError(f"unknown destination {name!r}; valid: {sorted(DEST_TO_ENV)}")
        return resolve_goal(self.base, DEST_TO_ENV[name])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python -m pytest tests/test_world_model.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add planner/__init__.py planner/world_model.py tests/test_world_model.py
git commit -m "feat(planner): WorldModel perception stub over the sim env"
```

---

### Task 2: NavigatorTool (the `go_to` tool)

**Files:**
- Create: `planner/navigator_tool.py`
- Test: `tests/test_navigator_tool.py`

**Interfaces:**
- Consumes: `WorldModel`, `DEST_TO_ENV` (Task 1); `chained_eval.run_leg`, `chained_eval.REACH_OVERRIDE`; `evaluate.load_q_model`, `evaluate.process_observation`; `goal_geometry.distance`, `goal_geometry.eval_step_budget`; `motion.MotionState`; `homebot.goals.GOAL_THRESHOLD`.
- Produces:
  - `class NavigatorTool(checkpoint=..., readout="softmax_rel", temp=0.1, device=None)` with `reset(seed=None) -> dict` and `go_to(destination: str) -> dict` returning `{"reached": bool, "steps": int, "state": dict}` or `{"reached": False, "error": str, "state": dict}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_navigator_tool.py
from planner.navigator_tool import NavigatorTool


def test_reset_returns_state_and_go_to_returns_outcome_shape():
    nav = NavigatorTool()  # loads the 314 champion
    s = nav.reset(seed=0)
    assert "robot_xy" in s
    out = nav.go_to("human")
    assert {"reached", "steps", "state"} <= set(out)
    assert isinstance(out["reached"], bool) and isinstance(out["steps"], int)


def test_unknown_destination_is_error_not_crash():
    nav = NavigatorTool()
    nav.reset(seed=0)
    out = nav.go_to("bogus")
    assert out["reached"] is False and "error" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python -m pytest tests/test_navigator_tool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'planner.navigator_tool'`.

- [ ] **Step 3: Write minimal implementation**

```python
# planner/navigator_tool.py
"""The single tool the LLM drives: go_to(destination). Holds one live navigator
episode (model, env, obs, motion, pose) so pose persists across calls — exactly
the chained-eval setup, one leg per call."""
import gymnasium as gym
import torch

import homebot  # noqa: F401  (env registration)
from homebot.goals import GOAL_THRESHOLD
from evaluate import load_q_model, process_observation
from goal_geometry import distance, eval_step_budget
from motion import MotionState
from chained_eval import run_leg, REACH_OVERRIDE
from planner.world_model import WorldModel, DEST_TO_ENV


class NavigatorTool:
    def __init__(self, checkpoint="checkpoints/run314_q_model_best.pt",
                 readout="softmax_rel", temp=0.1, device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(
            "HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
            obs_resolution=(96, 96), n_trash=2, max_steps=20000,
            map_name="default", random_start=True)
        self.base = self.env.unwrapped
        self.world = WorldModel(self.env)
        self.model = load_q_model(
            checkpoint, self.env.action_space.n, self.device,
            goal_layers=2, head_layers=4, use_motion=True)
        self.readout, self.temp = readout, temp
        self.obs = None
        self.ms = None

    def reset(self, seed=None) -> dict:
        raw, _ = self.env.reset(seed=seed)
        self.obs = process_observation(raw)
        self.ms = MotionState(self.env.action_space.n)
        return self.world.state()

    def go_to(self, destination: str) -> dict:
        try:
            gx, gy = self.world.resolve(destination)
        except (KeyError, ValueError) as e:
            return {"reached": False, "error": str(e), "state": self.world.state()}
        env_name = DEST_TO_ENV[destination]
        reach = REACH_OVERRIDE.get(env_name, GOAL_THRESHOLD)
        r = self.base._robot
        budget = max(1, int(eval_step_budget(distance(r.x, r.y, gx, gy))))
        reached, steps, self.obs = run_leg(
            self.model, self.env, self.base, self.obs, (gx, gy),
            budget, self.device, self.readout, self.temp, self.ms, reach)
        return {"reached": bool(reached), "steps": int(steps),
                "state": self.world.state()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python -m pytest tests/test_navigator_tool.py -v`
Expected: PASS (2 passed). Note: loads the checkpoint + builds the env, so this test is slower (~10-20s).

- [ ] **Step 5: Commit**

```bash
git add planner/navigator_tool.py tests/test_navigator_tool.py
git commit -m "feat(planner): NavigatorTool go_to over run 314 + softmax_rel"
```

---

### Task 3: LLMClient (system prompt, tool schema, response normalization)

**Files:**
- Create: `planner/llm_client.py`
- Modify: `requirements.txt` (add `openai`)
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Consumes: `openai` package; env vars / args for `base_url`, `model`.
- Produces:
  - `SYSTEM_PROMPT: str`, `GO_TO_TOOL: dict` (OpenAI tool schema).
  - `class LLMClient(base_url=..., model=..., api_key="local")` with `chat(messages: list[dict]) -> dict` returning `{"tool_calls": list[dict], "text": str | None}` where each tool call is `{"id": str, "name": str, "arguments": dict}`.
  - `staticmethod LLMClient._normalize(message) -> dict` — converts an OpenAI chat message object to that shape.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_client.py
import json
from types import SimpleNamespace

from planner.llm_client import LLMClient, SYSTEM_PROMPT, GO_TO_TOOL


def test_go_to_tool_schema_shape():
    assert GO_TO_TOOL["type"] == "function"
    fn = GO_TO_TOOL["function"]
    assert fn["name"] == "go_to"
    assert "destination" in fn["parameters"]["properties"]


def test_system_prompt_teaches_mechanics_and_destinations():
    p = SYSTEM_PROMPT.lower()
    assert "go_to" in p
    for token in ("fridge", "human", "door", "trash"):
        assert token in p
    assert "drink" in p and "deliver" in p  # env mechanics are taught


def test_normalize_tool_call_message():
    msg = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="go_to",
                                     arguments=json.dumps({"destination": "fridge"})))],
    )
    out = LLMClient._normalize(msg)
    assert out["text"] is None
    assert out["tool_calls"] == [
        {"id": "call_1", "name": "go_to", "arguments": {"destination": "fridge"}}]


def test_normalize_final_text_message():
    msg = SimpleNamespace(content="All done.", tool_calls=None)
    out = LLMClient._normalize(msg)
    assert out["tool_calls"] == [] and out["text"] == "All done."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python -m pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'planner.llm_client'`.

- [ ] **Step 3: Write minimal implementation**

First add the dependency:
```
# requirements.txt  (append a line)
openai
```
Then:
```python
# planner/llm_client.py
"""Thin wrapper over a local OpenAI-compatible endpoint (ollama/vLLM on the 5090).
Endpoint-agnostic: base_url + model are configurable. Normalizes responses into a
small shape the agent loop consumes, so the loop can be tested with a mock."""
import json

SYSTEM_PROMPT = """You are the planner for a home robot. You accomplish tasks by \
moving the robot between named destinations with the go_to tool. You cannot do \
anything except call go_to and talk to the user.

Destinations: fridge, human, door, trash.

How the world works:
- go_to("trash") collects the trash there (use it to tidy up).
- go_to("fridge") picks up a drink.
- go_to("human") while carrying a drink delivers it to the person.
- go_to("door") picks up a package waiting at the door; go_to("human") while \
carrying it delivers it.

After each go_to you get back whether the robot reached the destination and the \
current state (what it is carrying, trash remaining, deliveries done). If a \
go_to times out (reached=false), decide whether to retry, try a different route, \
or tell the user you could not complete the task. When the task is finished, or \
cannot be done, reply with a short natural-language message to the user instead \
of calling a tool."""

GO_TO_TOOL = {
    "type": "function",
    "function": {
        "name": "go_to",
        "description": "Drive the robot to a named destination. Returns whether "
                       "it reached the destination and the updated world state.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "enum": ["fridge", "human", "door", "trash"],
                    "description": "Where to send the robot.",
                },
            },
            "required": ["destination"],
        },
    },
}


class LLMClient:
    def __init__(self, base_url="http://localhost:11434/v1",
                 model="qwen2.5:14b-instruct", api_key="local"):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    @staticmethod
    def _normalize(message) -> dict:
        """OpenAI chat message -> {"tool_calls": [...], "text": str|None}."""
        tool_calls = []
        for tc in (message.tool_calls or []):
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments or "{}"),
            })
        return {"tool_calls": tool_calls,
                "text": None if tool_calls else message.content}

    def chat(self, messages: list[dict]) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=[GO_TO_TOOL])
        return self._normalize(resp.choices[0].message)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python -m pytest tests/test_llm_client.py -v`
Expected: PASS (4 passed). (`openai` must be installed: `conda run -n sac-homebot pip install openai`.)

- [ ] **Step 5: Commit**

```bash
git add planner/llm_client.py tests/test_llm_client.py requirements.txt
git commit -m "feat(planner): LLMClient (system prompt, go_to schema, normalization)"
```

---

### Task 4: PlannerAgent (closed-loop ReAct loop)

**Files:**
- Create: `planner/agent_loop.py`
- Test: `tests/test_planner_loop.py`

**Interfaces:**
- Consumes: a `client` with `chat(messages) -> {"tool_calls": [...], "text": str|None}` (Task 3); a `navigator` with `go_to(destination) -> dict` (Task 2); `SYSTEM_PROMPT` (Task 3).
- Produces:
  - `MAX_TOOL_CALLS = 12`.
  - `class PlannerAgent(client, navigator, system_prompt=SYSTEM_PROMPT, max_tool_calls=MAX_TOOL_CALLS)` with `conversation: list[dict]` and `handle_utterance(text: str) -> str` (the spoken response).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_planner_loop.py
from planner.agent_loop import PlannerAgent


class MockLLM:
    """Returns scripted normalized responses, one per chat() call."""
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return self.scripted.pop(0)


class MockNav:
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.visited = []

    def go_to(self, destination):
        self.visited.append(destination)
        return {"reached": self.outcomes.get(destination, True),
                "steps": 1, "state": {"carrying": None}}


def _tool(dest, cid="c1"):
    return {"tool_calls": [{"id": cid, "name": "go_to",
                            "arguments": {"destination": dest}}], "text": None}


def _say(text):
    return {"tool_calls": [], "text": text}


def test_executes_tool_calls_then_returns_spoken_response():
    llm = MockLLM([_tool("trash"), _tool("fridge"), _tool("human"), _say("Done.")])
    nav = MockNav()
    agent = PlannerAgent(llm, nav)
    out = agent.handle_utterance("tidy up and bring me a drink")
    assert nav.visited == ["trash", "fridge", "human"]
    assert out == "Done."


def test_tool_results_are_fed_back_into_conversation():
    llm = MockLLM([_tool("fridge"), _say("ok")])
    nav = MockNav()
    agent = PlannerAgent(llm, nav)
    agent.handle_utterance("get a drink")
    roles = [m["role"] for m in agent.conversation]
    assert "tool" in roles  # the go_to result was appended for the LLM to see


def test_conversation_persists_across_utterances():
    llm = MockLLM([_say("hi"), _say("bye")])
    nav = MockNav()
    agent = PlannerAgent(llm, nav)
    agent.handle_utterance("hello")
    agent.handle_utterance("later")
    user_turns = [m for m in agent.conversation if m["role"] == "user"]
    assert len(user_turns) == 2


def test_tool_call_budget_stops_infinite_loops():
    llm = MockLLM([_tool("fridge")] * 50)  # never says a final message
    nav = MockNav()
    agent = PlannerAgent(llm, nav, max_tool_calls=12)
    out = agent.handle_utterance("loop forever")
    assert len(nav.visited) == 12
    assert "couldn't" in out.lower() or "could not" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python -m pytest tests/test_planner_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'planner.agent_loop'`.

- [ ] **Step 3: Write minimal implementation**

```python
# planner/agent_loop.py
"""Closed-loop ReAct planner. handle_utterance(text) -> spoken_response over a
persistent message history. Voice is adapters outside this class: ASR feeds text
in, TTS speaks the returned response."""
import json

from planner.llm_client import SYSTEM_PROMPT

MAX_TOOL_CALLS = 12


class PlannerAgent:
    def __init__(self, client, navigator, system_prompt=SYSTEM_PROMPT,
                 max_tool_calls=MAX_TOOL_CALLS):
        self.client = client
        self.nav = navigator
        self.max_tool_calls = max_tool_calls
        self.conversation = [{"role": "system", "content": system_prompt}]

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
                result = self.nav.go_to(tc["arguments"]["destination"])
                self.conversation.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "content": json.dumps(result)})
        msg = "I couldn't complete that within the step budget."
        self.conversation.append({"role": "assistant", "content": msg})
        return msg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python -m pytest tests/test_planner_loop.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add planner/agent_loop.py tests/test_planner_loop.py
git commit -m "feat(planner): closed-loop ReAct agent (handle_utterance + budget)"
```

---

### Task 5: CLI runner + NL-task eval suite

**Files:**
- Create: `planner/run_planner.py`
- Create: `planner/eval_planner.py`
- Test: `tests/test_eval_planner.py`

**Interfaces:**
- Consumes: `NavigatorTool` (Task 2), `LLMClient` (Task 3), `PlannerAgent` (Task 4), `WorldModel.state` shape (Task 1).
- Produces:
  - `eval_planner.TASKS: list[tuple[str, callable]]` — `(utterance, check(state)->bool)`.
  - `eval_planner.score_task(agent, nav, utterance, check, seed) -> bool` — reset, run, check final state.
  - `run_planner.main()` — CLI: `--utterance`, `--base-url`, `--model`, `--seed`; prints the spoken response + final state.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_planner.py
from planner.eval_planner import score_task


class _Agent:
    def __init__(self, reply="ok"):
        self.reply = reply
        self.said = []

    def handle_utterance(self, text):
        self.said.append(text)
        return self.reply


class _Nav:
    """Fakes a delivered-drink end state after reset."""
    def __init__(self):
        self._state = {"drink_delivered": False}

    def reset(self, seed=None):
        self._state = {"drink_delivered": True}  # pretend the agent succeeded
        return self._state

    def state(self):
        return self._state


def test_score_task_runs_utterance_and_checks_final_state():
    agent, nav = _Agent(), _Nav()
    ok = score_task(agent, nav, "bring me a drink",
                    check=lambda s: s["drink_delivered"], seed=0)
    assert ok is True
    assert agent.said == ["bring me a drink"]


def test_score_task_fails_when_check_false():
    agent, nav = _Agent(), _Nav()
    ok = score_task(agent, nav, "x", check=lambda s: s.get("nope", False), seed=0)
    assert ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python -m pytest tests/test_eval_planner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'planner.eval_planner'`.

- [ ] **Step 3: Write minimal implementation**

```python
# planner/eval_planner.py
"""NL-task suite for the planner: each task is (utterance, check(state)->bool).
Run the real LLM end-to-end in sim and score on final world state — the
chained_eval philosophy with the LLM generating the chain. score_task is
LLM/Nav-agnostic (duck-typed) so it is unit-testable with fakes."""
import argparse

TASKS = [
    ("Please tidy up — clear the trash.",
     lambda s: s["trash_remaining"] == 0),
    ("Bring me a drink from the fridge.",
     lambda s: s["drink_delivered"]),
    ("Bring me the package from the door.",
     lambda s: s["package_delivered"]),
    ("Clear the trash and then bring me a drink.",
     lambda s: s["trash_remaining"] == 0 and s["drink_delivered"]),
]


def score_task(agent, nav, utterance, check, seed=0) -> bool:
    nav.reset(seed=seed)
    agent.handle_utterance(utterance)
    return bool(check(nav.state()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", default="qwen2.5:14b-instruct")
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()

    from planner.navigator_tool import NavigatorTool
    from planner.llm_client import LLMClient
    from planner.agent_loop import PlannerAgent

    nav = NavigatorTool()
    llm = LLMClient(base_url=args.base_url, model=args.model)
    passed = 0
    total = 0
    for utterance, check in TASKS:
        for i in range(args.episodes):
            agent = PlannerAgent(llm, nav)  # fresh conversation per task run
            ok = score_task(agent, nav, utterance, check, seed=i)
            passed += int(ok); total += 1
        print(f"  {utterance!r}: see running tally")
    print(f"\nPlanner task completion: {passed}/{total} = {100.0 * passed / total:.0f}%")


if __name__ == "__main__":
    main()
```

```python
# planner/run_planner.py
"""Run a single utterance end-to-end in sim against the real local LLM.

    python planner/run_planner.py --utterance "bring me a drink"
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--utterance", required=True)
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", default="qwen2.5:14b-instruct")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from planner.navigator_tool import NavigatorTool
    from planner.llm_client import LLMClient
    from planner.agent_loop import PlannerAgent

    nav = NavigatorTool()
    nav.reset(seed=args.seed)
    agent = PlannerAgent(LLMClient(base_url=args.base_url, model=args.model), nav)
    response = agent.handle_utterance(args.utterance)
    print(f"\nROBOT: {response}")
    print(f"STATE: {nav.world.state()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python -m pytest tests/test_eval_planner.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add planner/run_planner.py planner/eval_planner.py tests/test_eval_planner.py
git commit -m "feat(planner): CLI runner + NL-task eval suite"
```

---

## Manual integration check (after all tasks, requires the local LLM)

Not a unit test — run once the model is serving on the 5090:

1. Serve the model: `ollama pull qwen2.5:14b-instruct && ollama serve` (or vLLM with an OpenAI-compatible endpoint).
2. `conda run -n sac-homebot python planner/run_planner.py --utterance "clear the trash and bring me a drink"`
   Expect: the robot calls `go_to("trash")`, `go_to("fridge")`, `go_to("human")` and reports completion; final STATE shows `trash_remaining: 0`, `drink_delivered: True`.
3. `conda run -n sac-homebot python planner/eval_planner.py --episodes 10`
   Record per-task completion. Bar: completion ≈ the product of the relevant per-leg reach rates (the planner adds no avoidable failures).

## Self-Review

**Spec coverage:** WorldModel (Task 1) = perception stub; NavigatorTool (Task 2) = go_to + softmax_rel + 314; LLMClient (Task 3) = model/prompt/schema; PlannerAgent (Task 4) = closed-loop ReAct + conversational I/O + budget; run/eval (Task 5) = CLI + NL-task suite. Voice provision = `handle_utterance(text)->str` over `conversation` (Task 4). Deferrals (interaction tools, vision, voice, multi-map) honored — none built. All spec sections map to a task.

**Placeholder scan:** none — every code/test step is complete; the only deferred-to-manual item is the real-LLM run, which inherently needs a running model and is labeled a manual check, not a unit test.

**Type consistency:** `go_to(destination) -> {"reached","steps","state"}` is produced by NavigatorTool (Task 2) and consumed by PlannerAgent (Task 4) and the MockNav (Task 4). `client.chat(messages) -> {"tool_calls","text"}` is produced by LLMClient (Task 3) and consumed by PlannerAgent (Task 4) and MockLLM (Task 4). `DEST_TO_ENV` (Task 1) used by NavigatorTool (Task 2). `state()` keys (`carrying, trash_remaining, drink_delivered, package_delivered, robot_xy`) defined in Task 1 and asserted in Task 5's checks. Consistent.
