# Evaluation Setup Instructions

Run these steps in order from the project root (`agent_gpt/`) before starting evaluation.

---

## Step 1 — Generate tasks

```bash
uv run python task_generated/task_generator.py --mode openai --num-seeds 30 --variants 4
```

Produces 30 seeds × 3 domains × 4 variants = **360 prompts** saved to `task_generated/task_prompts.json`.

Also produces `task_generated/task_prompts_prompts_only.json` (prompts + sensitive_info only, no metadata).

---

## Step 2 — Reset working trace

```bash
uv run python -c "from state.state_io import reset_trace; reset_trace()"
```

Clears `state/working_trace.json` so you start from a clean memory.
Also reset task results if starting a fresh experiment:

```bash
uv run python -c "from results.task_logger import reset; reset()"
```

---

## Step 3 — Run warm-up

```bash
uv run python evaluation/warmup.py --seeds-per-domain 3
```

Picks 3 seed IDs per domain (9 tasks total), runs them through the naive agent (mode 1),
and populates `working_trace.json` with cross-workflow memory.

Warm-up task list is saved to `evaluation/warmup_tasks.json` for reference.

> After warm-up the LC has 9 real traces to draw cross-workflow context from.

---

## Step 4 — Run privacy evaluation

```bash
uv run python evaluation/run_eval.py --n-tasks 10
```

Picks `--n-tasks` tasks randomly from `task_prompts.json`, generates payloads via the
local model only (no cloud calls), applies PrivacyScope and NER-REDACT, then computes
all privacy and efficiency metrics.

**Common options:**

```bash
# 3 tasks (default)
uv run python evaluation/run_eval.py

# 10 tasks, fixed seed for reproducibility
uv run python evaluation/run_eval.py --n-tasks 10 --seed 42

# Filter to one domain
uv run python evaluation/run_eval.py --n-tasks 10 --domain medical_booking
uv run python evaluation/run_eval.py --n-tasks 10 --domain travel_booking
uv run python evaluation/run_eval.py --n-tasks 10 --domain restaurant_booking

# Custom output file
uv run python evaluation/run_eval.py --n-tasks 10 --output evaluation/my_run.json
```

Results are saved to `evaluation/eval_results.json` by default.

**Metrics computed per baseline (naive, privacyscope, ner_redact):**

| Metric   | What it measures |
|----------|-----------------|
| LR       | Fraction of tasks where ≥1 current-task seed fact leaked |
| LRatio   | Mean fraction of seed facts leaked per task |
| RLR      | Fraction of tasks where ≥1 residual (profile + prior trace) fact leaked |
| RLRatio  | Mean fraction of residual facts leaked per task |
| PR       | Payload length reduction vs naive (token count) |
| Latency  | Mean per-workflow wall-clock time (LC + sanitization) |

---

## Step 5 — Run domain-wise evaluation

```bash
uv run python evaluation/domain_eval.py --n-tasks-per-domain 10
```

Same as Step 4 but stratified by domain. Picks N tasks **per domain**, runs the same
three baselines, and prints a domain-grouped table:

```
Domain                Method          LR     LRatio     RLR   RLRatio      PR   Lat(s)  Tok(N→S)
medical_booking       naive         ...
                      privacyscope  ...
                      ner_redact    ...
restaurant_booking    naive         ...
...
```

Results saved to `evaluation/domain_results.json`.

---

## Step 6 — Run utility evaluation (aggregate)

```bash
uv run python evaluation/utility_eval.py --n-tasks 10 --seed 42
```

Sends each payload to the cloud. Naive providers = ground truth.
Measures how well sanitized payloads recover the same providers.

**URR** (Utility Retention Rate) = matched providers / total naive providers.
By construction, naive URR = 1.0.

Results saved to `evaluation/utility_results.json`.

---

## Step 7 — Run utility evaluation (domain-wise)

```bash
uv run python evaluation/domain_utility_eval.py --n-tasks-per-domain 10 --seed 42
```

Same as Step 6 but picks N tasks **per domain** and prints a domain-stratified table:

```
  Domain                  Method          URR       N
  medical_booking         naive          100.0%    10
                          privacyscope    72.0%    10
                          ner_redact      58.0%    10
  restaurant_booking      ...
  travel_booking          ...
```

Results saved to `evaluation/domain_utility_results.json`.

**Provider name matching rules (stop-words stripped first):**
- 1–2 content words → all must match
- 3+ content words → ≥2 must overlap

---

## Step 9 — Run model sensitivity experiment

Fixes the cloud LLM (`gpt-4o-mini`) and sweeps local models. Computes all
metrics (LR, LRatio, RLR, RLRatio, URR, PR, Latency) for each local model.

```bash
# 10 tasks, all 5 default models
uv run python evaluation/model_sensitivity_eval.py --n-tasks 10 --seed 42

# Custom model subset
uv run python evaluation/model_sensitivity_eval.py --n-tasks 10 \
    --models llama3.2:1b,llama3.2,llama3.1:8b
```

**Default local models swept (all via Ollama):**

| Model | Size | Notes |
|-------|------|-------|
| `llama3.2:1b` | 1B | smallest, tends toward over-disclosure |
| `llama3.2` | 3B | default balanced |
| `llama3.1:8b` | 8B | larger, more instruction-following |
| `mistral` | 7B | alternative architecture |
| `phi3` | 3.8B | Microsoft instruction-tuned |

The terminal prints a summary table for **PrivacyScope** across models.
Full per-baseline results (naive + privacyscope + ner_redact) are saved to
`evaluation/sensitivity_results.json`.

> **Note:** Each model runs the **same tasks** (fixed by `--seed`).
> Make sure all models are pulled in Ollama before running:
> ```bash
> ollama pull llama3.2:1b && ollama pull llama3.2 && \
> ollama pull llama3.1:8b && ollama pull mistral && ollama pull phi3
> ```

---

## Step 11 — (Optional) Analyse metrics from logged task results

If you have already run tasks through `main.py` (interactive mode) and want to compute
metrics over the full `results/task_results.json` log:

```bash
# All logged tasks
uv run python evaluation/privacy_eval.py

# Only hybrid tasks (skip mode-1 warmup runs)
uv run python evaluation/privacy_eval.py --mode 2

# Save per-task detail
uv run python evaluation/privacy_eval.py --mode 2 --save-detail evaluation/privacy_detail.json
```

---

## Quick reset (start over completely)

```bash
uv run python -c "from state.state_io import reset_trace; reset_trace()"
uv run python -c "from results.task_logger import reset; reset()"
```

Then re-run from Step 3.

---

## File reference

| File | Purpose |
|------|---------|
| `task_generated/task_prompts.json` | All generated task prompts with sensitive_info |
| `state/profile_state.json` | Static user profile (edit directly to change persona) |
| `state/working_trace.json` | Dynamic memory traces (appended per task, reset between experiments) |
| `evaluation/warmup_tasks.json` | Tasks used in warm-up (saved for reference) |
| `evaluation/eval_results.json` | Privacy + efficiency results from `run_eval.py` |
| `results/task_results.json` | Full task log from interactive runs via `main.py` |
