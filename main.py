"""
╔═══════════════════════════════════════════════════════════════════════╗
║  SAGE — Self‑Adaptive Guided Explainable Scheduler                  ║
║  Main entry point                                                   ║
║                                                                     ║
║  Stages executed in order:                                          ║
║    1. Collect baseline data (Random policy on SageEnv)              ║
║    2. Bootstrap the Digital‑Twin predictor from that data           ║
║    3. Train a PPO agent on SageEnv                                  ║
║    4. Build the full SAGE agent (PPO + DT + Explainer)              ║
║    5. Run comparative experiments (baselines vs SAGE)               ║
║    6. Produce plots & summary table                                 ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import random
import numpy as np

# ── Ensure project root is on sys.path ─────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.envs.sage_env import SageEnv
from src.dt.predictor import DTPredictor
from src.agents.train_ppo import train as train_ppo, evaluate as evaluate_ppo
from src.agents.sage_agent import SageAgent
from src.agents.explainer import SageExplainer
from src.utils.data_collector import collect_baseline_data, build_resources_map
from src.utils.experiment import (
    run_experiment,
    plot_comparison,
    plot_reward_curves,
    print_summary_table,
    RandomPolicy,
    RoundRobinPolicy,
    ShortestQueuePolicy,
    FastestResourcePolicy,
    PPOPolicy,
    SageFullPolicy,
)

# Also keep the legacy simulator demo available
from src.simulator.simulator import Simulator, Task, Resource
from src.simulator.schedulers import (
    RoundRobinScheduler,
    RandomScheduler,
    ShortestExpectedRuntimeScheduler,
    MinMinScheduler,
)


def parse_args():
    p = argparse.ArgumentParser(description="SAGE — Self‑Adaptive Guided Explainable Scheduler")
    p.add_argument("--mode", choices=["full", "train", "eval", "experiment", "demo"],
                    default="full", help="Pipeline stage to run")
    p.add_argument("--timesteps", type=int, default=50_000,
                    help="PPO training timesteps")
    p.add_argument("--num-tasks", type=int, default=20,
                    help="Tasks per episode")
    p.add_argument("--num-episodes", type=int, default=50,
                    help="Episodes for experiments")
    p.add_argument("--baseline-episodes", type=int, default=100,
                    help="Episodes to collect for DT bootstrap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-dir", type=str, default="models/ppo_sage")
    p.add_argument("--output-dir", type=str, default="logs/experiments")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  Stage 1: Collect baseline data
# ═══════════════════════════════════════════════════════════════════
def stage_collect_data(args):
    print("\n" + "=" * 60)
    print("  Stage 1 — Collecting baseline data for Digital Twin")
    print("=" * 60)
    records = collect_baseline_data(
        num_episodes=args.baseline_episodes,
        num_tasks=args.num_tasks,
        seed=args.seed,
    )
    print(f"  Collected {len(records)} records from {args.baseline_episodes} episodes")
    return records


# ═══════════════════════════════════════════════════════════════════
#  Stage 2: Bootstrap Digital Twin
# ═══════════════════════════════════════════════════════════════════
def stage_bootstrap_dt(records, env):
    print("\n" + "=" * 60)
    print("  Stage 2 — Training Digital Twin predictor")
    print("=" * 60)
    resources_map = build_resources_map(env)
    dt = DTPredictor(n_estimators=200, max_depth=5)
    results = dt.fit_from_records(records, resources_map)
    print("  DT fit results:")
    for tgt, metrics in results.items():
        print(f"    {tgt}: MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")
    dt.save("models/dt_predictor.joblib")
    print("  ✓ DT saved to models/dt_predictor.joblib")
    return dt


# ═══════════════════════════════════════════════════════════════════
#  Stage 3: Train PPO
# ═══════════════════════════════════════════════════════════════════
def stage_train_ppo(args):
    print("\n" + "=" * 60)
    print("  Stage 3 — Training PPO agent")
    print("=" * 60)
    model = train_ppo(
        total_timesteps=args.timesteps,
        n_envs=4,
        save_path=args.model_dir,
        num_tasks=args.num_tasks,
        seed=args.seed,
    )
    return model


# ═══════════════════════════════════════════════════════════════════
#  Stage 4: Build full SAGE agent
# ═══════════════════════════════════════════════════════════════════
def stage_build_sage(ppo_model, dt):
    print("\n" + "=" * 60)
    print("  Stage 4 — Assembling SAGE agent (PPO + DT + Explainer)")
    print("=" * 60)
    explainer = SageExplainer(use_shap=True)
    if dt.is_fitted:
        explainer.build_shap_explainers(dt)
    agent = SageAgent(
        ppo_model=ppo_model,
        dt_predictor=dt,
        explainer=explainer,
        top_k=3,
    )
    print("  ✓ SAGE agent assembled")
    return agent


# ═══════════════════════════════════════════════════════════════════
#  Stage 5: Run comparative experiments
# ═══════════════════════════════════════════════════════════════════
def stage_experiments(args, ppo_model, sage_agent):
    print("\n" + "=" * 60)
    print("  Stage 5 — Comparative experiments")
    print("=" * 60)
    num_res = 4
    policies = [
        RandomPolicy(num_res),
        RoundRobinPolicy(num_res),
        ShortestQueuePolicy(num_res),
        FastestResourcePolicy(num_res),
        PPOPolicy(ppo_model),
        SageFullPolicy(sage_agent),
    ]
    results = run_experiment(
        policies,
        num_episodes=args.num_episodes,
        num_tasks=args.num_tasks,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print_summary_table(results)
    plot_comparison(results, output_dir=args.output_dir)
    plot_reward_curves(results, output_dir=args.output_dir)
    return results


# ═══════════════════════════════════════════════════════════════════
#  Stage 6: Demonstrate one episode with explanations
# ═══════════════════════════════════════════════════════════════════
def stage_demo_explanations(sage_agent, num_tasks=10):
    print("\n" + "=" * 60)
    print("  Stage 6 — SAGE decision explanations (1 episode)")
    print("=" * 60)
    env = SageEnv(num_tasks=num_tasks, seed=99)
    obs, _ = env.reset()
    done = False
    while not done:
        action, explanation = sage_agent.decide(obs, env, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  {explanation.get('summary', '')}")
        if explanation.get("reason"):
            for r in explanation["reason"][:2]:
                print(f"    -> {r}")
        done = terminated or truncated
    if "avg_latency" in info:
        print(f"\n  Episode summary: latency={info['avg_latency']:.2f}  "
              f"energy={info['avg_energy']:.2f}  cost={info['avg_cost']:.2f}  "
              f"SLA-miss={info['sla_miss_rate']:.2%}  makespan={info['makespan']:.2f}")


# ═══════════════════════════════════════════════════════════════════
#  Legacy demo (original main.py behaviour)
# ═══════════════════════════════════════════════════════════════════
def run_legacy_demo():
    print("\n" + "=" * 60)
    print("  Legacy Simulator Demo (baseline schedulers)")
    print("=" * 60)
    resources = [
        Resource("R1", speed=10, energy_rate=0.5, cost_rate=0.2),
        Resource("R2", speed=20, energy_rate=0.8, cost_rate=0.4),
        Resource("R3", speed=15, energy_rate=0.6, cost_rate=0.3),
        Resource("R4", speed=25, energy_rate=0.9, cost_rate=0.5),
    ]
    tasks = [Task(f"T{i}", workload=random.randint(50, 200),
                  deadline=random.randint(10, 40)) for i in range(10)]

    for SchedClass, label in [
        (RoundRobinScheduler, "Round-Robin"),
        (ShortestExpectedRuntimeScheduler, "Shortest Expected Runtime"),
        (RandomScheduler, "Random"),
    ]:
        sched = SchedClass(resources)
        sim = Simulator(tasks, resources, sched)
        result = sim.run()
        print(f"  {label}: {result}")


# ═══════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/experiments", exist_ok=True)

    env = SageEnv(num_tasks=args.num_tasks, seed=args.seed)

    if args.mode == "demo":
        run_legacy_demo()
        return

    # ── Full pipeline ──────────────────────────────────────────
    if args.mode in ("full", "train"):
        records = stage_collect_data(args)
        dt = stage_bootstrap_dt(records, env)
        ppo_model = stage_train_ppo(args)
    elif args.mode == "eval":
        from stable_baselines3 import PPO
        model_path = os.path.join(args.model_dir, "ppo_sage_final")
        ppo_model = PPO.load(model_path)
        dt = DTPredictor()
        dt_path = "models/dt_predictor.joblib"
        if os.path.exists(dt_path):
            dt.load(dt_path)
    elif args.mode == "experiment":
        from stable_baselines3 import PPO
        model_path = os.path.join(args.model_dir, "ppo_sage_final")
        ppo_model = PPO.load(model_path)
        dt = DTPredictor()
        dt_path = "models/dt_predictor.joblib"
        if os.path.exists(dt_path):
            dt.load(dt_path)

    sage_agent = stage_build_sage(ppo_model, dt)

    if args.mode in ("full", "experiment"):
        stage_experiments(args, ppo_model, sage_agent)

    if args.mode in ("full", "eval"):
        stage_demo_explanations(sage_agent, num_tasks=args.num_tasks)

    print("\n✓ SAGE pipeline complete.")


if __name__ == "__main__":
    main()