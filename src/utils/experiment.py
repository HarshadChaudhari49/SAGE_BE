"""
Experiment Runner & Visualisation
==================================
Runs N episodes for each scheduler (baseline + SAGE) and produces
comparison plots and aggregate statistics.
"""

import os
import csv
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non‑interactive backend — safe for servers
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from statistics import mean

from src.envs.sage_env import SageEnv


# ── Baseline scheduler wrappers (act on env) ──────────────────────

class RandomPolicy:
    name = "Random"

    def __init__(self, num_resources):
        self.n = num_resources

    def select(self, obs, env):
        return random.randint(0, self.n - 1)


class RoundRobinPolicy:
    name = "RoundRobin"

    def __init__(self, num_resources):
        self.n = num_resources
        self._idx = 0

    def select(self, obs, env):
        action = self._idx % self.n
        self._idx += 1
        return action


class ShortestQueuePolicy:
    name = "ShortestQueue"

    def __init__(self, num_resources):
        self.n = num_resources

    def select(self, obs, env):
        return int(np.argmin(env.resource_finish_times))


class FastestResourcePolicy:
    name = "FastestResource"

    def __init__(self, num_resources):
        self.n = num_resources
        self.speeds = None

    def select(self, obs, env):
        if self.speeds is None:
            self.speeds = np.array([r["speed"] for r in env.resource_configs])
        return int(np.argmax(self.speeds))


class PPOPolicy:
    name = "PPO‑SAGE"

    def __init__(self, model):
        self.model = model

    def select(self, obs, env):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


class SageFullPolicy:
    """Full SAGE pipeline (PPO + DT look‑ahead + explainability)."""
    name = "SAGE"

    def __init__(self, sage_agent):
        self.agent = sage_agent

    def select(self, obs, env):
        action, explanation = self.agent.decide(obs, env, deterministic=True)
        return action


# ── Run experiments ────────────────────────────────────────────────

def run_policy(policy, env: SageEnv, num_episodes: int = 50, seed: int = 0):
    """Run a policy on the env for N episodes and return summary dicts."""
    summaries = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        while not done:
            action = policy.select(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        summary = {
            "episode": ep,
            "total_reward": total_reward,
        }
        if "avg_latency" in info:
            summary.update(info)
        summaries.append(summary)
    return summaries


def run_experiment(
    policies: List,
    num_episodes: int = 50,
    num_tasks: int = 20,
    seed: int = 42,
    output_dir: str = "logs/experiments",
):
    """
    Run all policies, save CSVs, return results dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    env = SageEnv(num_tasks=num_tasks, seed=seed)
    all_results: Dict[str, List[dict]] = {}

    for policy in policies:
        name = getattr(policy, "name", type(policy).__name__)
        print(f"Running {name} …")
        summaries = run_policy(policy, env, num_episodes=num_episodes, seed=seed)
        all_results[name] = summaries

        # Save per‑policy CSV
        csv_path = os.path.join(output_dir, f"{name.replace(' ', '_')}.csv")
        if summaries:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
                writer.writeheader()
                writer.writerows(summaries)
            print(f"  → saved {csv_path}")

    return all_results


# ── Plotting ───────────────────────────────────────────────────────

def plot_comparison(
    results: Dict[str, List[dict]],
    output_dir: str = "logs/experiments",
    show: bool = False,
):
    """Generate bar charts comparing schedulers on key metrics."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = ["avg_latency", "avg_energy", "avg_cost", "sla_miss_rate", "makespan"]
    labels = list(results.keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        for name in labels:
            vals = [s.get(metric, 0) for s in results[name] if metric in s]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        bars = ax.bar(labels, means, yerr=stds, capsize=4, alpha=0.8)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison.png")
    plt.savefig(path, dpi=150)
    print(f"✓ Comparison plot saved to {path}")
    if show:
        plt.show()
    plt.close()


def plot_reward_curves(
    results: Dict[str, List[dict]],
    output_dir: str = "logs/experiments",
    show: bool = False,
):
    """Plot per‑episode total reward for each policy."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, summaries in results.items():
        rewards = [s["total_reward"] for s in summaries]
        ax.plot(rewards, label=name, alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Reward Comparison")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "reward_curves.png")
    plt.savefig(path, dpi=150)
    print(f"✓ Reward curves saved to {path}")
    if show:
        plt.show()
    plt.close()


def print_summary_table(results: Dict[str, List[dict]]):
    """Print a quick text summary table to stdout."""
    metrics = ["avg_latency", "avg_energy", "avg_cost", "sla_miss_rate", "makespan", "total_reward"]
    header = f"{'Policy':<20}" + "".join(f"{m:<16}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, summaries in results.items():
        row = f"{name:<20}"
        for m in metrics:
            vals = [s.get(m, 0) for s in summaries if m in s]
            avg = np.mean(vals) if vals else 0
            row += f"{avg:<16.4f}"
        print(row)
    print("=" * len(header) + "\n")
