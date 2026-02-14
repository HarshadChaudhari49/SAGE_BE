"""
Unit tests for SAGE components.
Run with:  pytest src/tests/ -v
"""

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════
#  1. Simulator / Scheduler tests
# ═══════════════════════════════════════════════════════════════════
from src.simulator.simulator import Simulator, Task, Resource
from src.simulator.schedulers import (
    RoundRobinScheduler,
    RandomScheduler,
    ShortestExpectedRuntimeScheduler,
)


class TestSimulator:
    def _make_resources(self):
        return [
            Resource("R1", speed=10, energy_rate=0.5, cost_rate=0.2),
            Resource("R2", speed=20, energy_rate=0.8, cost_rate=0.4),
        ]

    def _make_tasks(self, n=5):
        return [Task(f"T{i}", workload=100, deadline=20) for i in range(n)]

    def test_round_robin_runs(self):
        resources = self._make_resources()
        tasks = self._make_tasks()
        sched = RoundRobinScheduler(resources)
        sim = Simulator(tasks, resources, sched)
        result = sim.run()
        assert "L_lat" in result
        assert result["L_lat"] > 0

    def test_shortest_runtime_scheduler(self):
        resources = self._make_resources()
        tasks = self._make_tasks(3)
        sched = ShortestExpectedRuntimeScheduler(resources)
        sim = Simulator(tasks, resources, sched)
        result = sim.run()
        # Fastest resource (R2, speed=20) → runtime = 100/20 = 5
        assert result["L_lat"] == pytest.approx(5.0)

    def test_random_scheduler_runs(self):
        resources = self._make_resources()
        tasks = self._make_tasks()
        sched = RandomScheduler(resources)
        sim = Simulator(tasks, resources, sched)
        result = sim.run()
        assert result["L_lat"] > 0


# ═══════════════════════════════════════════════════════════════════
#  2. Gymnasium Environment tests
# ═══════════════════════════════════════════════════════════════════
from src.envs.sage_env import SageEnv


class TestSageEnv:
    def test_reset_returns_correct_shape(self):
        env = SageEnv(num_tasks=5, seed=42)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)

    def test_step_returns_correct_types(self):
        env = SageEnv(num_tasks=5, seed=42)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert obs2.shape == obs.shape

    def test_episode_completes(self):
        env = SageEnv(num_tasks=10, seed=0)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps == 10
        assert "avg_latency" in info  # episode summary

    def test_observation_in_bounds(self):
        env = SageEnv(num_tasks=20, seed=7)
        obs, _ = env.reset()
        for _ in range(20):
            assert env.observation_space.contains(obs)
            action = env.action_space.sample()
            obs, *_ = env.step(action)


# ═══════════════════════════════════════════════════════════════════
#  3. Digital Twin Predictor tests
# ═══════════════════════════════════════════════════════════════════
from src.dt.predictor import DTPredictor


class TestDTPredictor:
    def _make_data(self, n=200):
        rng = np.random.RandomState(42)
        X = rng.rand(n, 5) * [200, 40, 25, 1.0, 0.5]  # scale
        y_rt = X[:, 0] / (X[:, 2] + 1e-6)
        y_en = X[:, 0] * X[:, 3]
        y_co = X[:, 0] * X[:, 4]
        return X, {"runtime": y_rt, "energy": y_en, "cost": y_co}

    def test_fit_and_predict(self):
        X, y = self._make_data()
        dt = DTPredictor(n_estimators=100)
        results = dt.fit(X, y)
        assert dt.is_fitted
        for tgt in DTPredictor.TARGET_NAMES:
            assert tgt in results
            assert results[tgt]["mae"] >= 0  # model produces valid predictions

    def test_predict_single(self):
        X, y = self._make_data()
        dt = DTPredictor(n_estimators=50)
        dt.fit(X, y)
        task = {"workload": 100, "deadline": 20}
        res = {"id": "R1", "speed": 15, "energy_rate": 0.6, "cost_rate": 0.3}
        preds = dt.predict_single(task, res)
        assert "runtime" in preds
        assert preds["runtime"] > 0

    def test_evaluate_candidates(self):
        X, y = self._make_data()
        dt = DTPredictor(n_estimators=50)
        dt.fit(X, y)
        task = {"workload": 120, "deadline": 15}
        resources = [
            {"id": "R1", "speed": 10, "energy_rate": 0.5, "cost_rate": 0.2},
            {"id": "R2", "speed": 25, "energy_rate": 0.9, "cost_rate": 0.5},
        ]
        loads = np.zeros(2)
        cands = dt.evaluate_candidates(task, resources, loads)
        assert len(cands) == 2
        assert cands[0]["score"] <= cands[1]["score"]  # sorted


# ═══════════════════════════════════════════════════════════════════
#  4. Metrics Collector tests
# ═══════════════════════════════════════════════════════════════════
from src.utils.metrics import MetricsCollector


class TestMetricsCollector:
    def test_log_and_compute(self, tmp_path):
        mc = MetricsCollector(filename=str(tmp_path / "test_metrics.csv"))
        mc.log_task("T0", "R1", 0, 5, 10, 3, 10, False)
        mc.log_task("T1", "R2", 0, 10, 20, 6, 8, True)
        result = mc.compute_metrics()
        assert result["L_lat"] == pytest.approx(7.5)
        assert result["L_sla"] == pytest.approx(0.5)

    def test_save_creates_file(self, tmp_path):
        fp = tmp_path / "m.csv"
        mc = MetricsCollector(filename=str(fp))
        mc.log_task("T0", "R1", 0, 5, 10, 3, 10, False)
        mc.save()
        assert fp.exists()


# ═══════════════════════════════════════════════════════════════════
#  5. Explainer tests
# ═══════════════════════════════════════════════════════════════════
from src.agents.explainer import SageExplainer


class TestExplainer:
    def test_explain_without_dt(self):
        exp = SageExplainer(use_shap=False)
        task = {"id": "T0", "workload": 100, "deadline": 20}
        resources = [
            {"id": "R1", "speed": 10, "energy_rate": 0.5, "cost_rate": 0.2},
            {"id": "R2", "speed": 25, "energy_rate": 0.9, "cost_rate": 0.5},
        ]
        candidates = [
            {"resource_idx": 1, "resource_id": "R2", "ppo_prob": 0.7,
             "predicted_runtime": 4.0, "predicted_energy": 90, "predicted_cost": 50,
             "score": 5.0},
            {"resource_idx": 0, "resource_id": "R1", "ppo_prob": 0.3,
             "predicted_runtime": 10.0, "predicted_energy": 50, "predicted_cost": 20,
             "score": 12.0},
        ]
        expl = exp.explain_decision(task, resources, 1, candidates)
        assert "summary" in expl
        assert "reason" in expl
        assert len(expl["reason"]) > 0


# ═══════════════════════════════════════════════════════════════════
#  6. Data collector test
# ═══════════════════════════════════════════════════════════════════
from src.utils.data_collector import collect_baseline_data


class TestDataCollector:
    def test_collect_returns_records(self):
        records = collect_baseline_data(num_episodes=5, num_tasks=10, seed=0)
        assert len(records) == 50  # 5 episodes * 10 tasks
        assert "task_workload" in records[0]
        assert "latency" in records[0]
