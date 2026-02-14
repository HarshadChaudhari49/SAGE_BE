"""
Data Collector
==============
Runs baseline schedulers on the simulator to collect (task, resource) → outcome
records that are used to **bootstrap** the Digital Twin predictor before RL
training begins.
"""

import random
import numpy as np
from typing import List, Dict

from src.envs.sage_env import SageEnv


def collect_baseline_data(
    num_episodes: int = 100,
    num_tasks: int = 20,
    seed: int = 42,
) -> List[dict]:
    """
    Run random‑assignment episodes on SageEnv and collect per‑task records.

    Returns a flat list of dicts, each containing:
        task_workload, task_deadline, resource_id, latency, energy, cost,
        sla_miss, speed, energy_rate, cost_rate
    """
    rng = random.Random(seed)
    env = SageEnv(num_tasks=num_tasks, seed=seed)
    all_records: List[dict] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = rng.randint(0, env.num_resources - 1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        # Gather episode records enriched with task / resource info
        for i, rec in enumerate(env.episode_records):
            task = env.tasks[i]
            res_cfg = env.resource_configs[
                next(j for j, r in enumerate(env.resource_configs) if r["id"] == rec["resource_id"])
            ]
            all_records.append({
                "task_workload": task["workload"],
                "task_deadline": task["deadline"],
                "resource_id": rec["resource_id"],
                "latency": rec["latency"],
                "energy": rec["energy"],
                "cost": rec["cost"],
                "sla_miss": rec["sla_miss"],
                "speed": res_cfg["speed"],
                "energy_rate": res_cfg["energy_rate"],
                "cost_rate": res_cfg["cost_rate"],
            })

    return all_records


def build_resources_map(env: SageEnv) -> Dict[str, dict]:
    """Return a resource_id → config dict map from the environment."""
    return {r["id"]: r for r in env.resource_configs}
