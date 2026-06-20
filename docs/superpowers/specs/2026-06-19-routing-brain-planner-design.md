# Routing Brain (Planner Layer) — Design

**Status:** approved design, pre-implementation
**Date:** 2026-06-19

## Goal

Build the higher-level "routing brain" that sits on top of the trained navigator:
take a natural-language task and accomplish it by driving the navigator through an
ordered series of destinations, reacting to each leg's outcome. This is the
planner half of the SayCan-style split — the navigator (the trained Q-model) is
the skill; the planner is the thinker that sequences skills.

The navigator is effectively done (run 314 champion + pin penalty + honest door;
deployed with the `softmax_rel` readout). This layer is what makes the system
controllable by a person and, later, by voice.

## Core architectural decisions (resolved in brainstorming)

1. **Planner is a text LLM over a symbolic world model — not a VLM-as-planner.**
   Decomposition (reasoning) and grounding (scene → coordinates) are kept
   separate. The LLM plans over named destinations; a perception module grounds
   names to coordinates. In sim, perception is a stub backed by the known map.
   Vision (a local VLM) drops into the *perception* slot later — the planner never
   changes. This matches how real robot stacks (and SayCan itself) decouple
   planning from perception, and keeps the planner unit-testable without images.

2. **Closed-loop (ReAct), not open-loop.** The LLM calls a `go_to` tool, receives
   the leg outcome (`reached`/`timeout`) plus current state, and decides the next
   call. The navigator is not 100% (legs fail ~15-25%), and a real home robot must
   handle "I couldn't get there" — open-loop blindly executes failed plans.

3. **Tool vocabulary is `go_to(destination)` only.** Interaction (pickup/deliver)
   is NOT exposed as explicit tools in this build. The sim performs pickup/deliver
   automatically on proximity, so the LLM accomplishes tasks like "bring a drink"
   by sequencing `go_to(fridge)` → `go_to(human)`; it is *told the env mechanics in
   its system prompt*. Explicit interaction tools are deferred to a future, richer
   environment where those are trained as goals.

4. **Conversational I/O boundary (voice-ready).** The planner's entry point is
   `handle_utterance(text, conversation) -> (conversation, spoken_response)` over a
   persistent message history — NOT `execute_task(plan) -> done`. This makes the
   natural-language response a first-class output channel and gives multi-turn
   dialogue for free. Voice is then pure adapters *outside* the planner: ASR
   (speech → text) upstream, TTS (`spoken_response` → speech) downstream. No
   planner change when voice is added. The leg-by-leg loop is also the natural
   injection point for barge-in later.

5. **Local LLM on the RTX 5090** (this machine), separate from the 3090 training
   server — no VRAM contention with training. Text-only model for now; vision GPU
   budget is reserved for the future perception VLM.

## Components

Each is independently testable with a well-defined interface.

### 1. WorldModel (perception stub)
The seam where real perception drops in later. Backed now by the sim env
(`HomeBot2D-V1` non-goal env, same as `chained_eval`): map fixtures, TaskManager
state, robot pose.

- `list_destinations() -> list[str]` — currently available named targets given
  state (e.g., `trash` only while trash remains; `fridge`, `human`, `door`).
- `state() -> dict` — task-relevant state (`carrying`, `trash_remaining`, robot
  pose), for the LLM's situational awareness and for success checks.
- `resolve(name) -> tuple[float, float]` — destination name → pixel coordinate.
  Wraps `goal_geometry`/`goal_to_coordinates` + trash selection. Raises on an
  unknown/unavailable name (surfaced to the LLM as a tool error).

Destination vocabulary (sim): `fridge`, `human` (recliner), `door`, `trash`.
`human` is the alias for the recliner-occupant target used throughout eval.

### 2. NavigatorTool
The single tool exposed to the LLM. Holds the live episode (model, env, current
obs, persistent pose).

- `go_to(destination: str) -> dict` →
  `{"reached": bool, "steps": int, "state": <WorldModel.state()>}`.
  Resolves the name via WorldModel, runs ONE navigator leg, returns the outcome.
- Reuses the navigator execution from `chained_eval` (`run_leg`) with the
  **`softmax_rel` readout at temp_rel=0.1** (the deployment readout) and the 314
  champion checkpoint. Pose persists across `go_to` calls (one live episode), as in
  the chained eval.
- Unknown/unavailable destination → returns an error payload the LLM can read and
  recover from (not an exception that kills the loop).

### 3. Planner / LLMClient
Thin wrapper over a local OpenAI-compatible endpoint (ollama or vLLM on
localhost). Endpoint-agnostic: configurable `base_url` + `model`.

- Registers the `go_to` tool schema (function calling).
- System prompt carries: the map/room layout at a high level, the destination
  list, the **env mechanics** ("going to the fridge picks up a drink; going to the
  human while carrying it delivers it; going to trash collects it"), the `go_to`
  contract, and the behavior contract (report progress/outcome in natural
  language; state clearly when the task is done or cannot be completed).

### 4. Agent loop (ReAct executor)
- `handle_utterance(text, conversation) -> (conversation, spoken_response)`:
  append the user turn; loop — call the LLM; if it returns a `go_to` tool call,
  execute via NavigatorTool and append the tool result; if it returns a final
  natural-language message, that is `spoken_response` and the turn ends. Enforce a
  per-utterance tool-call budget to kill infinite loops.
- `conversation` (message history: system + user + tool + assistant turns)
  persists across utterances within a session → multi-turn dialogue.
- **Session = one env episode.** Robot pose persists across `go_to` calls and
  across utterances in a session; a new session resets the env.

## Data flow

```
"tidy up and bring me a drink"
  -> ASR (future)            -> NL text
  -> Agent loop -> LLM
       -> go_to("trash")     -> NavigatorTool -> {reached:true, state:{trash_remaining:0}}
       -> go_to("fridge")    -> NavigatorTool -> {reached:true, state:{carrying:"drink"}}
       -> go_to("human")     -> NavigatorTool -> {reached:true, state:{drink_delivered:true}}
       -> final message: "Done — trash cleared and your drink is delivered."
  -> spoken_response -> TTS (future) -> speaker
```

## Model

Start with **Qwen2.5-14B-Instruct** — strong, reliable tool-calling; fits the
5090's 32GB in fp16 with headroom. Bump to a 32B (AWQ/GPTQ) if reasoning needs it.
Served via ollama (simplest, OpenAI-compatible at `localhost:11434`) or vLLM (for
throughput/latency). The LLMClient is endpoint-agnostic so the model can change
without touching the loop.

## Error handling

- `go_to` timeout → returned to the LLM, which decides retry / different route /
  report failure.
- Per-utterance **tool-call budget** (e.g. 12) → prevents infinite loops; on
  exhaustion the loop returns a "couldn't complete" response.
- Unknown/unavailable destination → structured error payload back to the LLM.
- LLM/endpoint unreachable → surfaced as a clear error (not a silent hang).

## Testing

The separation pays off here — every layer is testable in isolation:

- **Agent loop with a MOCK LLM** (scripted tool calls / responses): exercises the
  ReAct loop, tool-call budget, error recovery, and the `(actions,
  spoken_response)` contract with no model running.
- **NavigatorTool** against known coordinates via `run_leg`: reuses the
  already-validated navigator; asserts `reached` for reachable destinations.
- **WorldModel**: `list_destinations`/`state`/`resolve` against a reset env.
- **Integration**: a small NL-task suite — `(utterance, success_check)` pairs run
  in fresh sessions with the real LLM, scored on task completion (e.g.
  "bring me a drink" → `drink_delivered` true). This is the `chained_eval`
  philosophy with the LLM *generating* the chain instead of a hardcoded list.

## Reuse (do not duplicate)

- `chained_eval.run_leg` / `_select_action` — navigator leg execution + readouts.
- `evaluate.load_q_model`, `process_observation` — checkpoint loading + obs.
- `goal_geometry` / `goal_to_coordinates` — coordinate resolution.
- The 314 champion checkpoint + `softmax_rel` readout (temp_rel=0.1).
- `HomeBot2D-V1` non-goal env (same config as `chained_eval`).

## File structure

New `planner/` module in the sac-homebot repo:
- `planner/world_model.py` — WorldModel
- `planner/navigator_tool.py` — NavigatorTool
- `planner/llm_client.py` — LLMClient + system prompt + go_to schema
- `planner/agent_loop.py` — handle_utterance / ReAct loop
- `planner/run_planner.py` — CLI: run an utterance end-to-end in sim
- `planner/eval_planner.py` — NL-task suite + scoring
- `tests/test_planner_loop.py` (mock LLM), `tests/test_world_model.py`,
  `tests/test_navigator_tool.py`

## Scope / YAGNI (explicit deferrals)

- **Interaction tools** (pickup/deliver/drop) — deferred to the richer env where
  they're trained as goals; sim auto-handles them on proximity for now.
- **Vision / real perception** — WorldModel is a sim stub; the VLM perception
  module is future, dropped into the WorldModel slot.
- **Voice** (ASR/TTS/wake-word/barge-in) — adapters outside the planner, added
  later; the conversational I/O boundary is the only provision made now.
- **Multiple maps** — single default house.

## Success criteria

- Mock-LLM loop tests pass (loop, budget, recovery, I/O contract).
- NavigatorTool reaches known destinations at the navigator's measured rates.
- Integration suite: the real LLM completes a set of NL home tasks in sim at a
  rate consistent with the navigator's per-leg reliability (i.e. the planner adds
  no avoidable failures; chain completion ≈ product of per-leg reach rates).
