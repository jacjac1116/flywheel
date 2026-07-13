A self-improving data-analyst agent. A small open LLM, running locally on my Mac, gets one tool — a sandboxed Python/pandas execution environment — and must answer analytical questions about public UK energy data (carbon intensity API data, NESO open data portal, Elexon) by writing code, running it, reading results/errors, and iterating to a final answer.

Around it, I build the improvement flywheel:


A task generator creates questions from templates over the real datasets AND computes each ground-truth answer with reference code (the answer key). Difficulty tiers: single-series lookups → filtered aggregations → multi-dataset joins.
The agent attempts batches of questions. Scoring is pure arithmetic — final answer matches ground truth within tolerance, pass/fail. No LLM-as-judge for the core metric. Solve rate is THE number.
Rejection-sampling SFT: collect successful trajectories, fine-tune the model on its own wins (LoRA). Later rounds: DPO on passed-vs-failed trajectory pairs — genuine preference post-training.
Re-evaluate on held-out questions. Promotion gate: the new checkpoint replaces the old ONLY if solve rate improves; otherwise roll back. Model registry tracks versions.
A diagnostic agent (Claude API — needs real judgement) analyses failed trajectories per round and reports failure patterns ("collapses on date-filtering questions"), which drives what the task generator emphasises next round. Failure analysis → targeted data → retraining: the flywheel.


Headline result I'm chasing: solve rate climbing round over round — e.g. "untuned 3B model solved 12% of held-out tasks; after N rounds of rejection-sampling SFT + DPO it solves X% — and the loop that did it runs end-to-end with one command."

Why this design (so future sessions hold the frame)


The agent IS the model being trained — not a wrapper around an API. Post-training an agent on verifiable rewards is exactly what the JDs I'm targeting describe.
Automatic verification (my answer key) means unlimited free training/eval data, no labeling costs, no judge ambiguity.
The day-job connection is the task SHAPE (ad-hoc analytical questions over energy data — my actual work), not employer data. All data is public. The deployment story for interviews: utilities can't send internal data to external APIs, so a small self-hosted analyst model has a genuine niche.
Small model matters to the story: local, private, cheap — and "a 1.7–3B model doubled its solve rate" is the impressive claim.


Hardware & cost decisions (already made)


Mac: M3 Pro, 18GB unified memory. Model: Qwen3-1.7B or Llama-3.2-3B-Instruct — final pick by early inference tests; SAME base model locally and on cloud so local iteration transfers.
Attempts (trajectory generation) run FREE locally via mlx-lm — batches overnight is fine; wall-clock time is the cost, not money.
Training rounds: local MLX where feasible; rented NVIDIA GPU (RTX 4090 / A100 spot) for speed and for the CUDA/PEFT/TRL stack interviewers name. One vLLM serving demo on cloud at the end (quantisation, batching, throughput/latency measurements). Kill idle pods; spot for training, on-demand for the demo. Total cloud budget: under ~£50.
Diagnostic agent uses Claude API: pennies per round, negligible.


Phases


Environment + task generator (local, free — this is ~a third of the project, treat it as real engineering) — data download/caching for 2–3 public energy datasets; sandboxed code execution (subprocess isolation, timeouts, resource limits — solved-but-fiddly, do it properly); task templates + reference-answer computation; difficulty tiers; held-out eval split that stays frozen.
Agent loop + baseline (local, free) — multi-step loop (prompt → code → execute → observe → retry up to K steps → final answer) via mlx-lm; run the untuned model on the eval set. Baseline solve rate + trajectory logs. Also baseline a frontier model (Claude) on the same eval set for the comparison story.
First training round — SFT (local MLX → cloud) — my first real deep-model training. Collect winning trajectories, format for SFT, LoRA fine-tune, learn the loop hands-on (loss curves, epochs, learning rate, overfitting on small data). Re-eval. Built as a repeatable pipeline: versioned data, config-driven, MLflow-tracked, one command.
The flywheel (local + cloud rounds) — iterate: generate → score → collect wins → retrain → gate → promote/rollback. Add the diagnostic agent driving targeted task generation. Add DPO once SFT rounds plateau. Registry + promotion gates formalised.
Serve + prove (cloud) — quantise best checkpoint, serve with vLLM behind FastAPI, measure throughput/latency; the final comparison table (untuned vs tuned vs frontier; solve rate, cost per question, latency).
Automation + writeup — the whole loop end-to-end with one command; seeded-failure tests of the gate and diagnostic agent (verify the machinery catches known-bad rounds); README with the solve-rate-over-rounds chart front and centre.


Production hygiene throughout: Docker, CI, MLflow, pytest.

Critical design rules (hold me to these)


Thin slice first. Before anything is good: 10 template questions, the sandbox, the untuned model attempting them, a printed solve rate. One command. Only then widen.
The eval set is sacred. Held-out questions are never trained on, never regenerated mid-project; report solve rate only on it.
Track schema/format validity as a first-class metric — what fraction of attempts produced runnable code and a parseable final answer, separate from correctness.
Trajectory logs are an asset — store every attempt (prompt, code, outputs, errors, verdict) from day one; SFT data, DPO pairs, and diagnostics all come from them.
Don't let the task generator overfit to what the model finds easy. Difficulty mix is fixed by config, drifted only deliberately in response to diagnostics.
Sandbox is non-negotiable — model-generated code never runs unisolated.


How I want you to work with me


I learn by building, not reading theory. Teach me the WHY behind decisions, not just the how.
I tend to overcomplicate — push back HARD when I do.
Don't solve things for me unless I explicitly ask. Hold firm when I want shortcuts on things that matter.
I sometimes have Python fundamentals gaps — teach me when you spot them.
Explain jargon in plain language the first time it appears (assume I don't know post-training vocabulary: SFT, DPO, rejection sampling, trajectories, reward — build these up as we meet them). I want to genuinely understand well enough to discuss naturally in interviews.
Production standards throughout — flag tradeoffs explicitly, never default to "easy".
Motivation matters: I nearly abandoned a previous design because it felt dead. The climbing solve-rate number is the engine of this project — get me to a scored baseline as fast as responsibly possible.


Interview narrative I'm building toward

"I built an agent environment with automatically verifiable rewards — an analyst agent that answers questions about energy data by writing and running code, scored against ground truth I compute independently. Then I post-trained a small open model on its own successful trajectories — rejection-sampling SFT, then DPO — and raised its solve rate from X% to Y% over N rounds. The whole loop runs end-to-end with one command: generate, score, retrain, and a promotion gate that only ships checkpoints that beat the incumbent, with a diagnostic agent that turns failure patterns into targeted training data. I served the final model with vLLM and benchmarked it against a frontier API on solve rate, cost, and latency."