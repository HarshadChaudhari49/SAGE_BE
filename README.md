---
title: SAGE Scheduler Dashboard
sdk: docker
app_port: 7860
---

<p align="center">
  <h1 align="center">🧠 SAGE — Self-Adaptive Guided Explainable Scheduler</h1>
  <p align="center">
    <em>An AI-driven, explainable task scheduler for heterogeneous computing environments</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/RL-PPO%20(SB3)-green" alt="PPO">
  <img src="https://img.shields.io/badge/ML-Gradient%20Boosting-orange" alt="GBR">
  <img src="https://img.shields.io/badge/XAI-SHAP-red" alt="SHAP">
  <img src="https://img.shields.io/badge/framework-Gymnasium-purple" alt="Gymnasium">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Modules in Detail](#modules-in-detail)
  - [Gymnasium Environment](#1-gymnasium-environment-srcenvssage_envpy)
  - [Digital Twin Predictor](#2-digital-twin-predictor-srcdtpredictorpy)
  - [PPO Agent Training](#3-ppo-agent-training-srcagentstrain_ppopy)
  - [SAGE Agent](#4-sage-agent-srcagentssage_agentpy)
  - [Explainability Module](#5-explainability-module-srcagentsexplainerpy)
  - [Baseline Schedulers](#6-baseline-schedulers-srcsimulatorschedulerspy)
  - [Simulator](#7-simulator-srcsimulatorsimulatorpy)
  - [Metrics Collector](#8-metrics-collector-srcutilsmetricspy)
  - [Data Collector](#9-data-collector-srcutilsdata_collectorpy)
  - [Experiment Runner](#10-experiment-runner--visualisation-srcutilsexperimentpy)
- [Objective Function](#objective-function)
- [Self-Adaptation Loop](#self-adaptation-loop)
- [Experiment Results](#experiment-results)
- [CLI Reference](#cli-reference)
- [Running Tests](#running-tests)
- [TensorBoard Monitoring](#tensorboard-monitoring)
- [Configuration & Hyperparameters](#configuration--hyperparameters)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

**SAGE** addresses the challenge of scheduling computational tasks across heterogeneous resources (CPUs, GPUs, edge nodes) with competing objectives — latency, energy consumption, monetary cost, and SLA compliance. Traditional heuristic schedulers (Round-Robin, Min-Min) lack adaptability, while black-box RL agents lack transparency.

SAGE solves both problems by combining three key components:

| Component | Role | Technology |
|-----------|------|------------|
| **PPO Policy** | Proposes candidate resource assignments | Stable-Baselines3, PyTorch |
| **Digital Twin** | Evaluates & re-ranks candidates via predicted outcomes | Scikit-learn Gradient Boosting |
| **Explainer** | Provides human-readable justification for every decision | SHAP + contrastive reasoning |

The system is **self-adaptive** — the Digital Twin continuously retrains on observed outcomes to handle concept drift (workload changes, resource degradation, etc.).

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
                         │            SAGE Agent                    │
                         │                                          │
  Observation     ┌──────┴──────┐                                   │
  ─────────────→  │  PPO Policy │                                   │
  (task features  │  (MlpPolicy │──→ Top-K candidate actions        │
   + resource     │   128×128)  │         │                         │
   states)        └─────────────┘         ▼                         │
                                  ┌───────────────┐                 │
                                  │  Digital Twin  │                 │
                                  │  (3× GBR:     │──→ Predicted    │
                                  │   runtime,     │   outcomes      │
                                  │   energy,      │       │         │
                                  │   cost)        │       ▼         │
                                  └───────────────┘  Scalarise &    │
                                                     rank            │
                                                       │             │
                                                       ▼             │
                                                ┌────────────┐      │
                                                │ Explainer  │      │
                                                │ (SHAP +    │      │
                                                │ contrastive│      │
                                                │ reasoning) │      │
                                                └─────┬──────┘      │
                         └────────────────────────────┼─────────────┘
                                                      │
                                     ┌────────────────┼────────────────┐
                                     ▼                ▼                ▼
                               Best Action      Explanation      Self-Adaptation
                               (resource ID)    (JSON dict)      (retrain DT on
                                     │                            new outcomes)
                                     ▼
                            ┌─────────────────┐
                            │    SageEnv      │
                            │  (Gymnasium)    │
                            │  ┌───────────┐  │
                            │  │ Simulator  │  │
                            │  │ (heterog.  │  │
                            │  │  compute)  │  │
                            │  └───────────┘  │
                            └─────────────────┘
```

### Decision Flow (per task)

1. **Observe** — the environment provides a state vector encoding the current task's features and all resource states.
2. **Propose** — the PPO policy outputs action probabilities; the top-K most likely resources are selected as candidates.
3. **Evaluate** — the Digital Twin predicts runtime, energy, and cost for each candidate assignment.
4. **Rank** — candidates are scored using a weighted scalarised objective; the best is selected.
5. **Explain** — SHAP feature attributions + contrastive comparison (chosen vs. alternatives) are generated.
6. **Execute** — the chosen action is applied in the environment; the outcome is recorded for self-adaptation.

---

## Project Structure

```
SAGE/
├── main.py                          # 🚀 Pipeline orchestrator (entry point)
├── requirements.txt                 # 📦 Python dependencies
├── steps.txt                        # 📋 Development roadmap
├── README.md                        # 📖 This file
│
├── src/
│   ├── envs/
│   │   ├── __init__.py
│   │   └── sage_env.py              # 🎮 Gymnasium RL environment
│   │
│   ├── dt/
│   │   ├── __init__.py
│   │   └── predictor.py             # 🔮 Digital Twin predictor (GBR)
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── train_ppo.py             # 🏋️ PPO training script
│   │   ├── sage_agent.py            # 🧠 Full SAGE agent (PPO + DT + Explainer)
│   │   └── explainer.py             # 💡 SHAP + contrastive explainability
│   │
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── simulator.py             # ⚙️ Discrete task simulator
│   │   ├── schedulers.py            # 📊 Baseline scheduler implementations
│   │   └── sim.py                   # ⏱️ SimPy event-driven simulator
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py               # 📈 CSV metrics logger & aggregator
│   │   ├── data_collector.py        # 📥 Baseline data collection for DT
│   │   └── experiment.py            # 🧪 Experiment runner & plot generator
│   │
│   └── tests/
│       ├── __init__.py
│       └── test_sage.py             # ✅ 14 unit tests (pytest)
│
├── models/                          # 💾 Saved models (generated at runtime)
│   ├── dt_predictor.joblib          #    Trained Digital Twin
│   └── ppo_sage/
│       ├── best_model.zip           #    Best PPO model (by eval reward)
│       ├── ppo_sage_final.zip       #    Final PPO model
│       └── evaluations.npz          #    SB3 evaluation logs
│
└── logs/                            # 📁 Logs & outputs (generated at runtime)
    ├── metrics.csv                  #    Per-task metrics from simulator
    ├── experiments/
    │   ├── comparison.png           #    Bar chart: all schedulers compared
    │   ├── reward_curves.png        #    Per-episode reward curves
    │   ├── Random.csv               #    Per-episode results per policy
    │   ├── RoundRobin.csv
    │   ├── ShortestQueue.csv
    │   ├── FastestResource.csv
    │   ├── PPO‑SAGE.csv
    │   └── SAGE.csv
    └── tb_sage/                     #    TensorBoard event logs
```

---

## Installation

### Prerequisites

- **Python 3.10+** (tested with 3.13)
- **pip** package manager
- (Optional) NVIDIA GPU + CUDA for accelerated PyTorch training

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd SAGE

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Numerical computation |
| `pandas` | ≥2.0 | Data manipulation |
| `simpy` | ≥4.0 | Discrete-event simulation |
| `gymnasium` | ≥0.29 | RL environment interface |
| `stable-baselines3[extra]` | ≥2.1 | PPO algorithm implementation |
| `torch` | ≥2.0 | Neural network backend |
| `scikit-learn` | ≥1.3 | Digital Twin (GBR) models |
| `matplotlib` | ≥3.7 | Plotting & visualisation |
| `shap` | ≥0.42 | Explainability (feature attributions) |
| `flask` | ≥3.0 | REST API (for real-time mode) |
| `requests` | ≥2.31 | HTTP client |
| `pytest` | ≥7.4 | Unit testing |
| `joblib` | ≥1.3 | Model serialisation |
| `tensorboard` | ≥2.14 | Training visualisation |

---

## Quick Start

### Run the full SAGE pipeline (data → train → experiment → explain)

```bash
python main.py --mode full --timesteps 50000
```

This will:
1. Collect 2,000 baseline records (100 random-policy episodes × 20 tasks)
2. Train the Digital Twin predictor (3 Gradient Boosting models)
3. Train a PPO agent for 50,000 timesteps
4. Assemble the SAGE agent (PPO + DT + Explainer)
5. Run experiments comparing SAGE against 5 baselines
6. Print per-step explanations for a demo episode

### Run only the legacy simulator demo

```bash
python main.py --mode demo
```

### Run only experiments (after training)

```bash
python main.py --mode experiment
```

---

## Pipeline Stages

The `main.py` orchestrator executes 6 stages in sequence:

### Stage 1 — Baseline Data Collection

Runs random-assignment episodes on the Gymnasium environment to collect `(task, resource) → outcome` records. These records bootstrap the Digital Twin.

```
Stage 1 — Collecting baseline data for Digital Twin
Collected 2000 records from 100 episodes
```

### Stage 2 — Digital Twin Bootstrap

Fits three independent Gradient Boosting Regressors on the collected data:

| Target | Input Features | Typical R² |
|--------|---------------|------------|
| Runtime | workload, deadline, speed, energy_rate, cost_rate | 0.9999 |
| Energy | workload, deadline, speed, energy_rate, cost_rate | 0.9997 |
| Cost | workload, deadline, speed, energy_rate, cost_rate | 0.9999 |

```
Stage 2 — Training Digital Twin predictor
  runtime: MAE=0.0271  R²=0.9999
  energy:  MAE=0.3758  R²=0.9997
  cost:    MAE=0.1563  R²=0.9999
```

### Stage 3 — PPO Training

Trains a Proximal Policy Optimisation agent using Stable-Baselines3:

- **Policy**: 2-layer MLP (128 × 128) for both actor and critic
- **Environment**: 4 parallel vectorised `SageEnv` instances
- **Hyperparameters**: lr=3e-4, batch_size=64, n_epochs=10, γ=0.99
- **Logging**: TensorBoard + custom SAGE metrics callback

### Stage 4 — SAGE Assembly

Composes the three components into a single `SageAgent`:

```python
SageAgent(
    ppo_model   = trained_ppo,       # proposes candidates
    dt_predictor = trained_dt,        # evaluates candidates
    explainer    = SageExplainer(),   # explains decisions
    top_k        = 3,                 # candidates to evaluate
)
```

### Stage 5 — Comparative Experiments

Runs all 6 policies over N episodes and produces:
- Per-policy CSV results in `logs/experiments/`
- Bar chart comparison (`comparison.png`)
- Reward curve plot (`reward_curves.png`)
- Summary table printed to console

### Stage 6 — Explanation Demo

Runs one complete episode printing the SAGE agent's reasoning for every scheduling decision:

```
Task T0 → R4  |  pred_runtime=7.78s  pred_energy=174.57J  sla_miss=False
  → Preferred R4 over R2: runtime +1.91s, energy -19.27J, cost -19.38$
  → Preferred R4 over R3: runtime +5.15s, energy -58.16J, cost -38.79$
```

---

## Modules in Detail

### 1. Gymnasium Environment (`src/envs/sage_env.py`)

The `SageEnv` wraps the heterogeneous computing simulator as a standard Gymnasium environment compatible with any RL library.

#### Observation Space

A continuous `Box(0, 1)` vector of dimension `3 + 4 × num_resources` (default: 19):

| Features | Count | Description |
|----------|-------|-------------|
| Task features | 3 | `workload_norm`, `deadline_norm`, `slack_norm` |
| Per-resource features | 4 × N | `speed_norm`, `energy_rate_norm`, `cost_rate_norm`, `load_norm` |

All values are normalised to [0, 1].

#### Action Space

`Discrete(num_resources)` — the index of the resource to assign the current task to.

#### Reward

Negative scalarised objective (higher is better):

```
reward = -(α·latency_norm + β·energy_norm + γ·cost_norm + δ·sla_miss)
```

#### Episode Structure

- Each episode schedules `num_tasks` tasks sequentially
- Episode terminates after all tasks are assigned
- Final `info` dict contains episode summary metrics

#### Default Resource Pool

| Resource | Speed | Energy Rate | Cost Rate |
|----------|-------|-------------|-----------|
| R1 | 10 | 0.5 | 0.2 |
| R2 | 20 | 0.8 | 0.4 |
| R3 | 15 | 0.6 | 0.3 |
| R4 | 25 | 0.9 | 0.5 |

---

### 2. Digital Twin Predictor (`src/dt/predictor.py`)

The Digital Twin models the relationship between (task, resource) characteristics and execution outcomes, enabling **look-ahead evaluation** of candidate assignments before committing them.

#### Feature Vector (5 dimensions)

```
[task_workload, task_deadline, resource_speed, resource_energy_rate, resource_cost_rate]
```

#### Three Independent Models

| Model | Target | Algorithm |
|-------|--------|-----------|
| Runtime | `task_workload / resource_speed` + noise | GradientBoostingRegressor |
| Energy | `task_workload × resource_energy_rate` | GradientBoostingRegressor |
| Cost | `task_workload × resource_cost_rate` | GradientBoostingRegressor |

#### Key Methods

```python
dt = DTPredictor(n_estimators=200, max_depth=5)

# Fit from simulator records
dt.fit_from_records(records, resources_map)

# Predict for one (task, resource) pair
preds = dt.predict_single(task, resource)
# → {"runtime": 4.2, "energy": 90.0, "cost": 50.0}

# Score & rank all resources for a task
candidates = dt.evaluate_candidates(task, resources, resource_loads)
# → sorted list of dicts (best-first)

# Persistence
dt.save("models/dt_predictor.joblib")
dt.load("models/dt_predictor.joblib")
```

---

### 3. PPO Agent Training (`src/agents/train_ppo.py`)

Trains a PPO agent on `SageEnv` using Stable-Baselines3.

#### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 50,000 | Total environment steps |
| `n_envs` | 4 | Parallel vectorised environments |
| `learning_rate` | 3e-4 | Adam optimiser LR |
| `n_steps` | 256 | Steps per rollout |
| `batch_size` | 64 | Minibatch size |
| `n_epochs` | 10 | PPO update epochs per rollout |
| `gamma` | 0.99 | Discount factor |
| `net_arch` | pi=[128,128], vf=[128,128] | Actor-critic network shape |

#### Callbacks

- **`MetricsCallback`** — logs SAGE-specific metrics (latency, energy, cost, SLA miss rate, makespan) to TensorBoard
- **`EvalCallback`** — periodic evaluation, saves best model
- **`CheckpointCallback`** — periodic model checkpoints

#### Standalone Usage

```bash
# Train
python -m src.agents.train_ppo --timesteps 100000 --save-path models/ppo_sage

# Evaluate a saved model
python -m src.agents.train_ppo --evaluate-only models/ppo_sage/ppo_sage_final
```

---

### 4. SAGE Agent (`src/agents/sage_agent.py`)

The full SAGE decision pipeline, combining PPO + Digital Twin + Explainer.

#### Decision Algorithm

```
Input:  observation vector, environment state
Output: (action, explanation)

1. Extract action probabilities from PPO policy (softmax)
2. Select top-K candidate resources (highest probability)
3. For each candidate:
   a. DT predicts runtime, energy, cost
   b. Compute expected finish time (resource_load + predicted_runtime)
   c. Determine SLA miss (finish_time > deadline?)
   d. Compute scalarised score: α·runtime + β·energy + γ·cost + δ·sla_miss
4. Select candidate with lowest score
5. Generate explanation (contrastive + SHAP)
6. Return (best_action, explanation)
```

#### Self-Adaptation

The agent accumulates observed outcomes and periodically retrains the Digital Twin:

```python
agent.record_outcome(record)       # feed observed outcome
agent.maybe_adapt(resources_map)   # retrain DT if ≥50 records accumulated
```

This enables the system to adapt to:
- Changing workload distributions
- Resource degradation or upgrades
- Shifting deadline patterns

---

### 5. Explainability Module (`src/agents/explainer.py`)

Provides two complementary explanation types:

#### Contrastive Explanation

Compares the chosen resource against alternatives with quantified trade-offs:

```
Preferred R4 over R2: runtime +1.91s, energy -19.27J, cost -19.38$
Preferred R4 over R3: runtime +5.15s, energy -58.16J, cost -38.79$
```

#### SHAP Feature Attribution

Uses `TreeExplainer` on the Digital Twin's GBR models to attribute each feature's contribution:

```json
{
  "shap": {
    "runtime": {"workload": 0.42, "deadline": 0.01, "speed": -0.35, ...},
    "energy":  {"workload": 0.55, "energy_rate": 0.38, ...},
    "cost":    {"workload": 0.52, "cost_rate": 0.41, ...}
  }
}
```

#### Explanation Output Structure

```json
{
  "task_id": "T0",
  "chosen_resource": "R4",
  "reason": [
    "Preferred R4 over R2: runtime +1.91s, energy -19.27J, cost -19.38$"
  ],
  "candidates": [
    {"resource_id": "R4", "score": 8.2, "predicted_runtime": 4.0, ...},
    {"resource_id": "R2", "score": 12.1, "predicted_runtime": 5.9, ...}
  ],
  "shap": { ... },
  "summary": "Task T0 → R4 | pred_runtime=4.00s pred_energy=90.00J sla_miss=False"
}
```

---

### 6. Baseline Schedulers (`src/simulator/schedulers.py`)

Four baseline schedulers used for comparison:

| Scheduler | Strategy |
|-----------|----------|
| `RoundRobinScheduler` | Cycles through resources in order |
| `RandomScheduler` | Selects a random resource |
| `ShortestExpectedRuntimeScheduler` | Picks the fastest resource (lowest `workload/speed`) |
| `MinMinScheduler` | Batch assignment — each task gets its fastest resource |

---

### 7. Simulator (`src/simulator/simulator.py`)

A discrete task execution simulator with `Resource`, `Task`, and `Simulator` classes:

```python
resources = [Resource("R1", speed=10, energy_rate=0.5, cost_rate=0.2), ...]
tasks = [Task("T0", workload=150, deadline=20), ...]
sim = Simulator(tasks, resources, scheduler)
result = sim.run()
# → {"L_lat": 7.5, "L_energy": 85.0, "L_cost": 42.0, "L_sla": 0.3, "Objective": 8.2}
```

Additionally, `src/simulator/sim.py` provides a **SimPy-based discrete-event simulator** with arrival-time modelling, resource contention (queuing), and CPU/GPU affinity.

---

### 8. Metrics Collector (`src/utils/metrics.py`)

Logs per-task execution records and computes aggregate metrics:

| Metric | Formula | Description |
|--------|---------|-------------|
| `L_lat` | mean(latency) | Average task completion time |
| `L_energy` | mean(energy) | Average energy consumption |
| `L_cost` | mean(cost) | Average monetary cost |
| `L_sla` | count(sla_miss) / total | SLA violation rate |
| `Objective` | α·L_lat + β·L_energy + γ·L_cost + δ·L_sla | Scalarised objective |

---

### 9. Data Collector (`src/utils/data_collector.py`)

Collects training data for the Digital Twin by running random-policy episodes:

```python
records = collect_baseline_data(num_episodes=100, num_tasks=20, seed=42)
# → 2000 enriched records with task/resource features + outcomes
```

Each record contains:
`task_workload`, `task_deadline`, `resource_id`, `latency`, `energy`, `cost`, `sla_miss`, `speed`, `energy_rate`, `cost_rate`

---

### 10. Experiment Runner & Visualisation (`src/utils/experiment.py`)

Runs all schedulers head-to-head and generates publication-ready plots:

```python
policies = [RandomPolicy(4), RoundRobinPolicy(4), ..., SageFullPolicy(agent)]
results = run_experiment(policies, num_episodes=50, num_tasks=20)
print_summary_table(results)
plot_comparison(results)       # → logs/experiments/comparison.png
plot_reward_curves(results)    # → logs/experiments/reward_curves.png
```

**Generated Outputs:**
- `comparison.png` — bar charts with error bars for 5 metrics across all policies
- `reward_curves.png` — per-episode total reward line plot
- Per-policy CSV files with detailed per-episode results

---

## Objective Function

SAGE optimises a weighted multi-objective function:

$$J = \alpha \cdot L_{\text{lat}} + \beta \cdot L_{\text{energy}} + \gamma \cdot L_{\text{cost}} + \delta \cdot L_{\text{sla}}$$

| Weight | Default | Objective | Description |
|--------|---------|-----------|-------------|
| α | 1.0 | Latency | Task execution time (seconds) |
| β | 0.01 | Energy | Energy consumption (joules) |
| γ | 0.01 | Cost | Monetary cost ($) |
| δ | 5.0 | SLA Miss | Deadline violation penalty (binary) |

The RL reward is the **negative** of this objective (normalised), so the agent learns to minimise all components simultaneously. Weights can be tuned to prioritise different objectives depending on the deployment context.

---

## Self-Adaptation Loop

SAGE implements online self-adaptation through continuous Digital Twin retraining:

```
┌───────────────────────────────────────────────────────┐
│                                                       │
│   ┌──────────┐    observe     ┌──────────────────┐   │
│   │  SAGE    │──────────────→ │  Record Buffer   │   │
│   │  Agent   │                │  (recent         │   │
│   │          │←──────────────── outcomes)         │   │
│   └──────────┘    retrained   └────────┬─────────┘   │
│                   DT                    │              │
│                                    ≥ 50 records?      │
│                                         │              │
│                                    ┌────▼────────┐    │
│                                    │  Retrain DT │    │
│                                    │  on new data│    │
│                                    └─────────────┘    │
│                                                       │
└───────────────────────────────────────────────────────┘
```

**Trigger**: Every `adaptation_interval` (default: 50) observed outcomes.  
**Action**: Retrain all three GBR models on accumulated records.  
**Effect**: DT predictions improve as it learns from actual scheduling outcomes, compensating for concept drift.

---

## Experiment Results

Sample results from running the full pipeline (20,000 PPO training steps, 30 evaluation episodes):

| Policy | Avg Latency | Avg Energy | Avg Cost | SLA Miss Rate | Makespan | Total Reward |
|--------|-------------|------------|----------|---------------|----------|-------------|
| Random | 8.14 | 90.0 | 45.1 | 48.2% | 69.0 | -52.4 |
| RoundRobin | 8.17 | 89.6 | 44.8 | 44.5% | 61.8 | -48.8 |
| ShortestQueue | 7.40 | 95.2 | 48.8 | 40.3% | 41.9 | -44.2 |
| FastestResource | 5.11 | 115.1 | 63.9 | 78.8% | 102.3 | -81.6 |
| PPO-SAGE | 6.24 | 102.9 | 54.4 | 49.2% | 48.1 | -52.5 |
| **SAGE** | **5.96** | **105.9** | **56.8** | **28.0%** | **60.9** | **-31.2** |

### Key Findings

- **SAGE achieves the lowest SLA miss rate (28.0%)** — nearly half that of other methods.
- **SAGE achieves the best total reward (-31.2)** — significantly outperforming all baselines.
- The DT look-ahead enables SAGE to outperform raw PPO by **avoiding SLA-violating assignments** that appear locally optimal but lead to deadline misses.
- FastestResource has the lowest latency but **worst SLA compliance** (78.8% miss) due to overloading a single fast resource.

---

## CLI Reference

```bash
python main.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | choice | `full` | `full` · `train` · `eval` · `experiment` · `demo` |
| `--timesteps` | int | 50,000 | PPO training timesteps |
| `--num-tasks` | int | 20 | Tasks per episode |
| `--num-episodes` | int | 50 | Evaluation/experiment episodes |
| `--baseline-episodes` | int | 100 | Episodes for DT data collection |
| `--seed` | int | 42 | Random seed |
| `--model-dir` | str | `models/ppo_sage` | Model save/load directory |
| `--output-dir` | str | `logs/experiments` | Experiment output directory |

### Mode Descriptions

| Mode | What it does |
|------|-------------|
| `full` | Runs all 6 stages end-to-end |
| `train` | Stages 1-3 only (data collection + DT + PPO training) |
| `eval` | Loads saved models, assembles SAGE, runs explanation demo |
| `experiment` | Loads saved models, runs comparative experiments with plots |
| `demo` | Runs legacy simulator with baseline schedulers (no RL) |

### Standalone PPO Training

```bash
python -m src.agents.train_ppo --timesteps 100000 --save-path models/ppo_sage
python -m src.agents.train_ppo --evaluate-only models/ppo_sage/ppo_sage_final
```

---

## Running Tests

```bash
# Run all 14 tests with verbose output
pytest src/tests/test_sage.py -v

# Run with coverage (install pytest-cov first)
pytest src/tests/test_sage.py -v --cov=src
```

### Test Coverage

| Test Class | Tests | Components Verified |
|------------|-------|-------------------|
| `TestSimulator` | 3 | RoundRobin, ShortestRuntime, Random schedulers |
| `TestSageEnv` | 4 | Reset shape, step types, episode completion, obs bounds |
| `TestDTPredictor` | 3 | Fit & predict, single prediction, candidate ranking |
| `TestMetricsCollector` | 2 | Logging + aggregation, CSV persistence |
| `TestExplainer` | 1 | Explanation generation (contrastive + summary) |
| `TestDataCollector` | 1 | Correct record count and schema |

---

## TensorBoard Monitoring

PPO training logs SAGE-specific metrics to TensorBoard:

```bash
tensorboard --logdir logs/tb_sage
```

**Available Metrics:**

| Metric Path | Description |
|-------------|-------------|
| `sage/avg_latency` | Mean task latency per episode |
| `sage/avg_energy` | Mean energy consumption per episode |
| `sage/avg_cost` | Mean cost per episode |
| `sage/sla_miss_rate` | Fraction of SLA violations per episode |
| `sage/makespan` | Total schedule length per episode |
| `rollout/ep_rew_mean` | Mean episode reward (standard SB3) |
| `train/loss` | PPO training loss |

---

## Configuration & Hyperparameters

### Environment (`SageEnv`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_resources` | 4 | Number of heterogeneous resources |
| `num_tasks` | 20 | Tasks per episode |
| `workload_range` | (50, 200) | Random workload range |
| `deadline_range` | (10, 40) | Random deadline range |
| `alpha` | 1.0 | Latency weight in reward |
| `beta` | 0.01 | Energy weight in reward |
| `gamma` | 0.01 | Cost weight in reward |
| `delta` | 5.0 | SLA-miss penalty in reward |

### Digital Twin (`DTPredictor`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 200 | Number of boosting iterations |
| `max_depth` | 5 | Maximum tree depth |
| `lr` | 0.1 | Learning rate (shrinkage) |

### SAGE Agent

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 3 | Candidate actions to evaluate |
| `adaptation_interval` | 50 | Retrain DT every N observations |

---

## Future Work

As outlined in `steps.txt`, planned extensions include:

- **Real-time deployment** — Docker worker containers with REST API for live task submission (Step 10)
- **Kubernetes integration** — Shadow-mode scheduling alongside the default K8s scheduler (Step 10)
- **Monitoring dashboards** — Prometheus metrics + Grafana visualisation (Step 11)
- **CI/CD pipeline** — GitHub Actions for automated testing (Step 12)
- **Advanced DT** — PyTorch MLP with quantile regression for uncertainty estimates (Step 6)
- **Multi-agent scheduling** — Extend to distributed, cooperative scheduling
- **Dynamic resource pools** — Handle resource addition/removal at runtime

---

## License

This project is developed as part of a B.E. academic research project.

---

<p align="center">
  <em>Built with ❤️ using PyTorch, Stable-Baselines3, Scikit-learn, SHAP, and Gymnasium</em>
</p>
