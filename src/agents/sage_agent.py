"""
SAGE Agent — Self‑Adaptive Guided Explainable Scheduler
========================================================
Combines:
    1. **PPO policy** — proposes top‑K candidate resource assignments.
    2. **Digital‑Twin predictor** — evaluates each candidate's predicted
       runtime / energy / cost and SLA‑miss.
    3. **Explainer** — attributes feature importance to the final decision
       using SHAP on the DT model.
    4. **Self‑adaptation** — periodically retrains DT on recent records;
       optionally fine‑tunes PPO when objective drifts.

Usage::

    agent = SageAgent.from_pretrained("models/ppo_sage/ppo_sage_final")
    env   = SageEnv()
    obs, _ = env.reset()
    while True:
        action, explanation = agent.decide(obs, env)
        obs, reward, done, _, info = env.step(action)
        if done:
            break
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from stable_baselines3 import PPO

from src.dt.predictor import DTPredictor
from src.agents.explainer import SageExplainer


class SageAgent:
    """Full SAGE decision pipeline."""

    def __init__(
        self,
        ppo_model: PPO,
        dt_predictor: Optional[DTPredictor] = None,
        explainer: Optional[SageExplainer] = None,
        top_k: int = 3,
        alpha: float = 1.0,
        beta: float = 0.01,
        gamma: float = 0.01,
        delta: float = 5.0,
        adaptation_interval: int = 50,
    ):
        self.ppo = ppo_model
        self.dt = dt_predictor or DTPredictor()
        self.explainer = explainer or SageExplainer()
        self.top_k = top_k

        # Scalarisation weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Self‑adaptation bookkeeping
        self.adaptation_interval = adaptation_interval
        self._decision_count = 0
        self._recent_records: List[dict] = []

    # ── factory ─────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        ppo_path: str,
        dt_path: Optional[str] = None,
        **kwargs,
    ) -> "SageAgent":
        ppo = PPO.load(ppo_path)
        dt = DTPredictor()
        if dt_path:
            dt.load(dt_path)
        return cls(ppo_model=ppo, dt_predictor=dt, **kwargs)

    # ── main decision method ────────────────────────────────────────
    def decide(
        self,
        obs: np.ndarray,
        env,
        deterministic: bool = False,
    ) -> Tuple[int, dict]:
        """
        1. Ask PPO for action probabilities.
        2. Pick top‑K candidate resources.
        3. If DT is fitted, evaluate each candidate and pick the best.
        4. Generate explanation.
        """
        task = env.tasks[env.current_task_idx]
        resources = env.resource_configs
        loads = env.resource_finish_times.copy()

        # Step 1 — PPO action distribution
        action_probs = self._get_action_probs(obs)
        top_k_indices = np.argsort(action_probs)[::-1][: self.top_k]

        # Step 2 — DT look‑ahead (if fitted)
        if self.dt.is_fitted:
            candidates = []
            for idx in top_k_indices:
                res = resources[idx]
                preds = self.dt.predict_single(task, res)
                finish = loads[idx] + preds["runtime"]
                sla_miss = 1.0 if finish > task["deadline"] else 0.0
                score = (
                    self.alpha * preds["runtime"]
                    + self.beta * preds["energy"]
                    + self.gamma * preds["cost"]
                    + self.delta * sla_miss
                )
                candidates.append({
                    "resource_idx": int(idx),
                    "resource_id": res["id"],
                    "ppo_prob": float(action_probs[idx]),
                    "predicted_runtime": preds["runtime"],
                    "predicted_energy": preds["energy"],
                    "predicted_cost": preds["cost"],
                    "predicted_finish": finish,
                    "sla_miss": bool(sla_miss),
                    "score": score,
                })
            candidates.sort(key=lambda c: c["score"])
            chosen = candidates[0]
            action = chosen["resource_idx"]
        else:
            # No DT — fall back to PPO greedy / stochastic
            if deterministic:
                action = int(np.argmax(action_probs))
            else:
                action, _ = self.ppo.predict(obs, deterministic=False)
                action = int(action)
            candidates = [{"resource_idx": action, "resource_id": resources[action]["id"],
                           "ppo_prob": float(action_probs[action])}]
            chosen = candidates[0]

        # Step 3 — explanation
        explanation = self.explainer.explain_decision(
            task=task,
            resources=resources,
            chosen_idx=action,
            candidates=candidates,
            dt_predictor=self.dt if self.dt.is_fitted else None,
        )

        self._decision_count += 1
        return action, explanation

    # ── action probabilities from PPO ───────────────────────────────
    def _get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Extract softmax action probabilities from the PPO policy."""
        import torch
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.ppo.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.cpu().numpy().flatten()
        return probs

    # ── self‑adaptation ─────────────────────────────────────────────
    def record_outcome(self, record: dict):
        """Feed an observed outcome back for DT retraining."""
        self._recent_records.append(record)

    def maybe_adapt(self, resources_map: Dict[str, dict]):
        """
        If enough new records have been collected, retrain the DT on the
        accumulated data.  Called at the end of each episode or periodically.
        """
        if len(self._recent_records) >= self.adaptation_interval:
            enriched = []
            for rec in self._recent_records:
                enriched.append({
                    "task_workload": rec.get("workload", rec.get("task_workload", 0)),
                    "task_deadline": rec.get("deadline", rec.get("task_deadline", 0)),
                    "resource_id": rec["resource_id"],
                    "latency": rec["latency"],
                    "energy": rec["energy"],
                    "cost": rec["cost"],
                })
            X, y = DTPredictor.featurize_batch(enriched, resources_map)
            results = self.dt.fit(X, y)
            print(f"[SAGE‑Adapt] DT retrained on {len(self._recent_records)} records — {results}")
            self._recent_records.clear()

    # ── serialisation ───────────────────────────────────────────────
    def save_dt(self, path: str = "models/dt_predictor.joblib"):
        self.dt.save(path)

    def load_dt(self, path: str = "models/dt_predictor.joblib"):
        self.dt.load(path)
