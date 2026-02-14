"""
SAGE Gymnasium Environment
==========================
Wraps the heterogeneous‑computing simulator as a Gymnasium RL environment.

Observation (per step):
    A flat vector concatenating:
    - Current task features  [workload_norm, deadline_norm, slack_norm]
    - Per-resource features  [speed_norm, energy_rate_norm, cost_rate_norm, load_norm]
        repeated for each resource

Action:
    Discrete — index of the resource to assign the current task to.

Reward:
    Negative scalarised objective delta (lower latency / energy / cost / SLA miss → higher reward).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import List, Callable, Optional, Dict


class SageEnv(gym.Env):
    """Gymnasium environment for the SAGE scheduler."""

    metadata = {"render_modes": ["human"]}

    # ── construction ────────────────────────────────────────────────
    def __init__(
        self,
        num_resources: int = 4,
        num_tasks: int = 20,
        resource_configs: Optional[List[Dict]] = None,
        workload_range: tuple = (50, 200),
        deadline_range: tuple = (10, 40),
        alpha: float = 1.0,
        beta: float = 0.01,
        gamma: float = 0.01,
        delta: float = 5.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.num_resources = num_resources
        self.num_tasks = num_tasks
        self.workload_range = workload_range
        self.deadline_range = deadline_range

        # Scalarisation weights for reward
        self.alpha = alpha   # latency
        self.beta = beta     # energy
        self.gamma = gamma   # cost
        self.delta = delta   # SLA‑miss penalty

        # Resource pool (can be overridden)
        if resource_configs is None:
            self.resource_configs = [
                {"id": "R1", "speed": 10, "energy_rate": 0.5, "cost_rate": 0.2},
                {"id": "R2", "speed": 20, "energy_rate": 0.8, "cost_rate": 0.4},
                {"id": "R3", "speed": 15, "energy_rate": 0.6, "cost_rate": 0.3},
                {"id": "R4", "speed": 25, "energy_rate": 0.9, "cost_rate": 0.5},
            ]
        else:
            self.resource_configs = resource_configs
        self.num_resources = len(self.resource_configs)

        # Normalisation constants (derived from resource pool)
        self._max_speed = max(r["speed"] for r in self.resource_configs)
        self._max_energy = max(r["energy_rate"] for r in self.resource_configs)
        self._max_cost = max(r["cost_rate"] for r in self.resource_configs)
        self._max_workload = float(workload_range[1])
        self._max_deadline = float(deadline_range[1])

        # Spaces
        # obs = 3 task features + 4 per‑resource features * num_resources
        obs_dim = 3 + 4 * self.num_resources
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_resources)

        # Internal state (set in reset)
        self.tasks = []
        self.current_task_idx = 0
        self.resource_finish_times = np.zeros(self.num_resources)
        self.episode_records = []

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ── reset ───────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random if hasattr(self, "np_random") and self.np_random is not None else random

        self.tasks = []
        for i in range(self.num_tasks):
            wl = rng.integers(self.workload_range[0], self.workload_range[1] + 1) \
                if hasattr(rng, "integers") else rng.randint(self.workload_range[0], self.workload_range[1])
            dl = rng.integers(self.deadline_range[0], self.deadline_range[1] + 1) \
                if hasattr(rng, "integers") else rng.randint(self.deadline_range[0], self.deadline_range[1])
            self.tasks.append({"id": f"T{i}", "workload": int(wl), "deadline": int(dl)})

        self.current_task_idx = 0
        self.resource_finish_times = np.zeros(self.num_resources, dtype=np.float64)
        self.episode_records = []

        return self._get_obs(), {}

    # ── step ────────────────────────────────────────────────────────
    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        task = self.tasks[self.current_task_idx]
        res = self.resource_configs[action]

        # Compute execution on chosen resource
        exec_time = task["workload"] / res["speed"]
        start_time = self.resource_finish_times[action]
        finish_time = start_time + exec_time
        self.resource_finish_times[action] = finish_time

        energy = task["workload"] * res["energy_rate"]
        cost = task["workload"] * res["cost_rate"]
        sla_miss = 1.0 if finish_time > task["deadline"] else 0.0

        # Record
        record = {
            "task_id": task["id"],
            "resource_id": res["id"],
            "start_time": start_time,
            "finish_time": finish_time,
            "latency": exec_time,
            "energy": energy,
            "cost": cost,
            "deadline": task["deadline"],
            "sla_miss": bool(sla_miss),
        }
        self.episode_records.append(record)

        # Reward: negative weighted objective (minimise → maximise reward)
        reward = -(
            self.alpha * (exec_time / self._max_deadline)
            + self.beta * (energy / (self._max_workload * self._max_energy))
            + self.gamma * (cost / (self._max_workload * self._max_cost))
            + self.delta * sla_miss
        )

        self.current_task_idx += 1
        terminated = self.current_task_idx >= len(self.tasks)
        truncated = False

        info = record if not terminated else self._episode_summary()

        return self._get_obs(), float(reward), terminated, truncated, info

    # ── observation builder ─────────────────────────────────────────
    def _get_obs(self):
        if self.current_task_idx >= len(self.tasks):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        task = self.tasks[self.current_task_idx]
        remaining = len(self.tasks) - self.current_task_idx

        # Task features
        workload_norm = task["workload"] / self._max_workload
        deadline_norm = task["deadline"] / self._max_deadline
        slack_norm = max(0, task["deadline"] - task["workload"] / self._max_speed) / self._max_deadline

        obs = [workload_norm, deadline_norm, slack_norm]

        # Per‑resource features
        max_ft = max(self.resource_finish_times.max(), 1.0)
        for idx, res in enumerate(self.resource_configs):
            obs.append(res["speed"] / self._max_speed)
            obs.append(res["energy_rate"] / self._max_energy)
            obs.append(res["cost_rate"] / self._max_cost)
            obs.append(self.resource_finish_times[idx] / max_ft if max_ft > 0 else 0.0)

        return np.array(obs, dtype=np.float32)

    # ── helpers ─────────────────────────────────────────────────────
    def _episode_summary(self) -> dict:
        from statistics import mean
        records = self.episode_records
        if not records:
            return {}
        return {
            "avg_latency": mean(r["latency"] for r in records),
            "avg_energy": mean(r["energy"] for r in records),
            "avg_cost": mean(r["cost"] for r in records),
            "sla_miss_rate": sum(1 for r in records if r["sla_miss"]) / len(records),
            "makespan": max(r["finish_time"] for r in records),
            "num_tasks": len(records),
        }

    def get_episode_records(self):
        """Return raw per‑task records for the last episode."""
        return list(self.episode_records)

    def render(self, mode="human"):
        if self.current_task_idx < len(self.tasks):
            task = self.tasks[self.current_task_idx]
            print(f"[Step {self.current_task_idx}] Task {task['id']}  "
                  f"wl={task['workload']}  dl={task['deadline']}  "
                  f"res_loads={np.round(self.resource_finish_times, 2)}")
