# Project File Reference

All commands run from the project root (`agent_gpt/`).

---

## Setup (first time on any machine)

```bash
# Install all dependencies including the spaCy model
uv sync

# Start the local LLM (required for main.py and evaluations)
ollama pull llama3.2

# Copy .env and add your API key
cp .env.example .env   # then edit OPENAI_API_KEY=...
```

`uv sync` handles everything: Python packages + `en_core_web_sm` (spaCy model). No separate `spacy download` step needed.

---

## Entry Points

| File | Description | Run |
|------|-------------|-----|
| `main.py` | Interactive agent loop. Prompts the user for a task, runs HybridAgent, and logs results. | `uv run python main.py` |
| `agents/run_comparison.py` | Given a task, builds the naive payload via LC, applies PrivScope + Presidio + PEP, sends all 4 to CLM, shows payloads + responses + summary table. | `uv run python agents/run_comparison.py` `uv run python agents/run_comparison.py --task "book a dentist"` |

---

## Evaluation

| File | Description | Run |
|------|-------------|-----|
| `evaluation/run_evaluation.py` | Full evaluation pipeline. Loads tasks, builds naive payloads via LC, applies all 4 methods (naive/PrivScope/Presidio/PEP), sends to CLM, computes LR/LRatio/RLR/RLRatio/URR/TSR/PR/Latency. Saves per-task + aggregate results to `run_evaluation.json`. | `uv run python evaluation/run_evaluation.py` `uv run python evaluation/run_evaluation.py --n-tasks 20` |

---

## Agents

| File | Description | Run |
|------|-------------|-----|
| `agents/hybrid_agent.py` | Core agent. Four-phase pipeline: LC reasoning → privacy baseline → cloud search → form actuation. Supports modes 1–5 (naive through all baselines). | imported by `main.py` |
| `agents/vanilla_agent.py` | Minimal baseline agent. Sends the raw task directly to the cloud with no local reasoning or privacy handling. | `uv run python agents/vanilla_agent.py` |
| `agents/langgraph_demo.py` | Educational demo only. Reimplements the same 4-phase pipeline as a LangGraph `StateGraph`. For comparison with the native implementation. | `uv run python agents/langgraph_demo.py` `uv run python agents/langgraph_demo.py --task "book a dentist"` |

---

## LLM Wrappers

| File | Description | Run |
|------|-------------|-----|
| `llm/local_llm.py` | `LocalLLM` — wraps Ollama REST API (`http://localhost:11434`). Default model: `llama3.2`. | — |
| `llm/cloud_router.py` | `CloudLLM` — uniform `generate()` / `chat()` interface over OpenAI, Anthropic, and Gemini. Provider selected via `config.py`. | — |

---

## Privacy Baselines

| File | Description | Run |
|------|-------------|-----|
| `privacy/privacyscope.py` | **Original PrivacyScope** (4-stage: span extraction → scope control → classification → transformation). Currently used in the full evaluation pipeline. **Do not modify.** | — |
| `privacy/privscope.py` | **New PrivScope** governor. Stages 1–3b active. Full pipeline debug runner. | `uv run python privacy/privscope.py` |
| `privacy/span_extractor.py` | Stage 1 of the new PrivScope. Partitions payload into U_loc (withheld identifiers), U_med (mediation candidates), and C_t (context). Three-layer extractor: profile matching → structured regex → spaCy. | `uv run python privacy/span_extractor.py` |
| `privacy/scope_control.py` | Stage 2 of the new PrivScope. Filters U_med spans using Rel + κ + TaskGain. Runs Stage 1 + Stage 2 end-to-end and shows sanitized output. | `uv run python privacy/scope_control.py` |
| `privacy/span_classification.py` | Stage 3a of the new PrivScope. Joint PTH/CSS classification of retained spans using a single local-LM call. Runs Stage 1 + Stage 2 + Stage 3a end-to-end. | `uv run python privacy/span_classification.py` |
| `privacy/span_abstraction.py` | Stage 3b of the new PrivScope. Semantic type inference + calibrated abstraction for 35+ types. Runs Stage 1–3b end-to-end. | `uv run python privacy/span_abstraction.py` |
| `privacy/abstraction_policy.py` | Calibrated abstraction policy π_ψ: type hierarchies and levels for all 35+ semantic types. | — |
| `privacy/presidio.py` | **Presidio baseline.** PII detection and redaction using Microsoft Presidio. Replaces the old NER-REDACT baseline. | `uv run python privacy/presidio.py` |
| `privacy/pep.py` | **PEP baseline.** Local LLM light-touch filter: removes personal identifiers while preserving task-relevant content. | `uv run python privacy/pep.py` |
| `privacy/legacy/ner_redact.py` | **Archived.** Old NER-REDACT baseline (spaCy NER + regex). Replaced by Presidio. Kept for reference only. | — |

---

## State

| File | Description | Run |
|------|-------------|-----|
| `state/state_io.py` | Central state I/O. Merges `profile_state.json` + `working_trace.json`. Public API: `load_state()`, `save_state()`, `append_trace()`, `reset_trace()`. | Reset trace: `uv run python -c "from state.state_io import reset_trace; reset_trace()"` |
| `state/profile_state.json` | Static on-device user profile (Bob Smith). Edit directly to change the persona. Fields include name, DOB, address, phone, insurance, SSN, IDs, loyalty numbers. | — |
| `state/working_trace.json` | Dynamic trace store. Appended after every tool call. Grows across workflows — source of cross-workflow residual leakage. | — |

---

## Mock Services

| File | Description | Run |
|------|-------------|-----|
| `mock_services/server.py` | Flask server on port 8000. Serves static and dynamically registered service endpoints. Agents submit booking forms here. | `uv run python mock_services/server.py` |

---

## Task Generation

| File | Description | Run |
|------|-------------|-----|
| `task_generated/task_generator.py` | Generates task prompts across 3 domains (medical, travel, restaurant) × seeds × variants. Saves to `task_prompts.json`. | `uv run python task_generated/task_generator.py --mode openai --num-seeds 30 --variants 4` |
| `task_generated/task_prompts.json` | Generated prompts with `sensitive_info` ground truth for evaluation. | — |

---

## Evaluation

| File | Description | Run |
|------|-------------|-----|
| `evaluation/warmup.py` | Runs N seed tasks per domain through the naive agent to populate `working_trace.json` with cross-workflow memory before evaluation. | `uv run python evaluation/warmup.py --seeds-per-domain 3` |
| `evaluation/run_eval.py` | **Main privacy eval.** Generates payloads via LC only (no cloud), applies all baselines, computes LR / LRatio / RLR / RLRatio / PR / Latency. | `uv run python evaluation/run_eval.py --n-tasks 10 --seed 42` |
| `evaluation/domain_eval.py` | Same as `run_eval.py` but stratified by domain (N tasks per domain). | `uv run python evaluation/domain_eval.py --n-tasks-per-domain 10` |
| `evaluation/utility_eval.py` | Utility evaluation. Sends payloads to cloud and measures URR (Utility Retention Rate) vs naive ground truth. | `uv run python evaluation/utility_eval.py --n-tasks 10 --seed 42` |
| `evaluation/domain_utility_eval.py` | Domain-stratified utility evaluation. | `uv run python evaluation/domain_utility_eval.py --n-tasks-per-domain 10 --seed 42` |
| `evaluation/model_sensitivity_eval.py` | Sweeps 5 local Ollama models and reports all metrics per model. | `uv run python evaluation/model_sensitivity_eval.py --n-tasks 10 --seed 42` |
| `evaluation/privacy_eval.py` | Post-hoc privacy metrics over `results/task_results.json` from interactive runs via `main.py`. | `uv run python evaluation/privacy_eval.py --mode 2` |

---

## Results

| File | Description | Run |
|------|-------------|-----|
| `results/task_logger.py` | Append-only logger. Writes each task run (prompt + all payloads + CLM responses) to `task_results.json`. | Reset: `uv run python -c "from results.task_logger import reset; reset()"` |
| `results/task_results.json` | Full task log from interactive `main.py` runs. | — |

---

## Config

| File | Description |
|------|-------------|
| `config.py` | Cloud provider selection (`PROVIDER`), model names (`MODEL_CONFIG`), default local model (`LOCAL_MODEL`), and the 5-model sweep list (`EVAL_LOCAL_MODELS`). |

---

## Quick Reset

```bash
# Clear memory traces (start fresh between experiments)
uv run python -c "from state.state_io import reset_trace; reset_trace()"

# Clear all logged task results
uv run python -c "from results.task_logger import reset; reset()"
```
