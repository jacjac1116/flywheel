# Flywheel

A self-improving analyst agent: a tool-using LLM that writes pandas code, runs it in an isolated sandbox against real data, grades its own answer against ground truth, and iterates on its failures — wrapped in an evaluation harness that measures whether changes actually help.

## Why This Exists

Most "data analyst agent" demos are a single prompt-and-pray call: ask a question, get an answer, hope it's right. There's no execution, no ground truth, and no way to tell whether a prompt tweak improved anything or just moved the noise around.

Flywheel treats analysis as a closed loop. The agent doesn't just describe what it would do — it writes real pandas code, executes it against the actual dataset, compares its result to a known correct answer, and uses the failures to drive the next round of improvement. The point of the project is not the agent; it's the **measurement discipline** around it — knowing, with evidence, whether the thing is getting better.

## What Makes This Different

The differentiator is the evaluation and self-improvement layer, not the agent loop itself:

* **Real code execution, not description** — the agent writes pandas and runs it in an isolated sandbox against real data, so answers are grounded in computation rather than plausible-sounding text.
* **Self-grading against ground truth** — each answer is scored against a known-correct result, so "did it actually solve the task" is a measured fact, not a vibe.
* **Multi-sample evaluation** — every task is run *k* times and scored as a per-task solve rate, so run-to-run sampling noise is quantified and separated from real changes before any claim is made.
* **Failure-mode analysis** — failures are cross-tabulated against outcomes to isolate *why* the agent fails, turning a low score into a specific, fixable diagnosis.
* **Rejection-sampling fine-tuning** *(in progress)* — successful trajectories are collected as training data to raise the base solve rate.

## How the Loop Works

1. **Plan** — the agent receives a task and decides what to compute.
2. **Write** — it emits pandas code as a tool call.
3. **Execute** — the code runs in an isolated sandbox against the real dataset.
4. **Grade** — the result is scored against ground truth.
5. **Iterate** — on failure, the error and context are fed back and the agent tries again.

Every run is logged as a trajectory (see `reports/`) so individual successes and failures can be inspected and replayed.

## Results

Diagnosing failure modes rather than eyeballing outputs paid off directly: cross-tabulating failures against outcomes surfaced a tool-calling regression, and a single targeted fix drove the **malformed tool-call rate from ~45% to ~0%**, recovering suite performance. Rejection-sampling fine-tuning to push the solve rate higher is in progress.

## Repository Layout

| Path | Contents |
|---|---|
| `src/analyst_agent/` | Core package — agent loop, tool definitions, sandbox execution, grading, and the evaluation harness. |
| `configs/` | Run configuration (model, sampling parameters, number of samples *k*, task sets). |
| `scripts/` | Entry points for running the agent and the evaluation harness. |
| `data/` | Datasets the agent analyses, plus ground-truth answers. |
| `reports/` | Logged agent runs (plan → code → execution → grade) for inspection and replay. 
| `tests/` | Logged agent runs (plan → code → execution → grade) for inspection and replay. |
| `.github/workflows/` | CI (tests run on push). |
| `notebook.ipynb` | Exploratory / demo notebook. |
| `pyproject.toml` | Package definition and dependencies. |

## Requirements

* **Apple Silicon** — inference runs locally via [`mlx-lm`](https://github.com/ml-explore/mlx-lm) (Qwen3), which targets Apple's MLX framework.
* Python 3.10+

## Setup

```bash
pip install -e .
```

## Usage

The agent and evaluation harness are driven from `scripts/`, configured via files in `configs/`. For example:

```bash
# Adjust the script name and config path to match the entry points in scripts/
python scripts/<run_eval>.py --config configs/<your_config>.yaml
```

Results are written per task as solve rates across *k* samples; individual runs are saved to `trajectories/` for inspection.

> **TODO:** replace `<run_eval>` and `<your_config>` above with the actual entry point and config once confirmed, and add a one-line description + topics to the repo's GitHub "About" panel.

## Project Status

* **Phase 1** — Tool-using agent loop with sandboxed pandas execution ✅
* **Phase 2** — Self-grading against ground truth ✅
* **Phase 3** — Multi-sample evaluation harness (per-task solve rate, variance quantification) ✅
* **Phase 4** — Failure-mode analysis and targeted fixes ✅
* **Phase 5** — Rejection-sampling fine-tuning to raise solve rate 🔜
