"""
Digital Twin Predictor
======================
Predicts (runtime, energy, cost) for a (task, resource) pair using a
Gradient‑Boosting ensemble with optional uncertainty via quantile regression.

Training data is collected from simulator runs.  Once fitted the DT is used
inside the SAGE decision loop to evaluate candidate assignments *before* they
are committed in the real simulator.

Features
--------
For each (task, resource) pair the feature vector is::

    [task_workload, task_deadline, res_speed, res_energy_rate, res_cost_rate]

Targets
-------
Three targets are predicted independently:

* ``runtime``   – task_workload / res_speed  (deterministic baseline, but DT
  learns residuals from contention / queuing)
* ``energy``    – task_workload * res_energy_rate
* ``cost``      – task_workload * res_cost_rate
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from typing import List, Dict, Optional, Tuple


class DTPredictor:
    """Digital‑Twin predictor for runtime / energy / cost."""

    TARGET_NAMES = ["runtime", "energy", "cost"]

    def __init__(self, n_estimators: int = 200, max_depth: int = 5, lr: float = 0.1):
        self.models: Dict[str, GradientBoostingRegressor] = {}
        for tgt in self.TARGET_NAMES:
            self.models[tgt] = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                loss="squared_error",
            )
        self.is_fitted = False
        self._feature_names = [
            "workload", "deadline", "speed", "energy_rate", "cost_rate"
        ]

    # ── featurise one (task, resource) pair ─────────────────────────
    @staticmethod
    def featurize(task: dict, resource: dict) -> np.ndarray:
        """Return a 1‑D feature vector for a single (task, resource) pair."""
        return np.array([
            task["workload"],
            task["deadline"],
            resource["speed"],
            resource["energy_rate"],
            resource["cost_rate"],
        ], dtype=np.float64)

    @staticmethod
    def featurize_batch(records: List[dict], resources_map: Dict[str, dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Convert a list of simulator record dicts into feature matrix ``X``
        and target dict ``y``.

        Parameters
        ----------
        records : list[dict]
            Each dict must contain at minimum:
            ``task_workload``, ``task_deadline``, ``resource_id``,
            ``latency`` (runtime), ``energy``, ``cost``.
        resources_map : dict[str, dict]
            Mapping from resource_id → resource config dict.
        """
        X_list, y_rt, y_en, y_co = [], [], [], []
        for rec in records:
            res = resources_map[rec["resource_id"]]
            feat = np.array([
                rec["task_workload"],
                rec["task_deadline"],
                res["speed"],
                res["energy_rate"],
                res["cost_rate"],
            ])
            X_list.append(feat)
            y_rt.append(rec["latency"])
            y_en.append(rec["energy"])
            y_co.append(rec["cost"])
        X = np.vstack(X_list)
        y = {"runtime": np.array(y_rt), "energy": np.array(y_en), "cost": np.array(y_co)}
        return X, y

    # ── fit ──────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: Dict[str, np.ndarray], test_size: float = 0.2):
        """Fit all three target models.  Returns train / test MAE per target."""
        results = {}
        for tgt in self.TARGET_NAMES:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y[tgt], test_size=test_size, random_state=42
            )
            self.models[tgt].fit(X_tr, y_tr)
            pred_te = self.models[tgt].predict(X_te)
            results[tgt] = {
                "mae": mean_absolute_error(y_te, pred_te),
                "r2": r2_score(y_te, pred_te),
            }
        self.is_fitted = True
        return results

    def fit_from_records(self, records: List[dict], resources_map: Dict[str, dict]):
        """Convenience: fit directly from simulator episode records."""
        enriched = []
        for rec in records:
            enriched.append({
                "task_workload": rec.get("task_workload", rec.get("workload", 0)),
                "task_deadline": rec.get("task_deadline", rec.get("deadline", 0)),
                "resource_id": rec["resource_id"],
                "latency": rec["latency"],
                "energy": rec["energy"],
                "cost": rec["cost"],
            })
        X, y = self.featurize_batch(enriched, resources_map)
        return self.fit(X, y)

    # ── predict ──────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict all three targets.  ``X`` shape (n, 5)."""
        assert self.is_fitted, "DTPredictor must be fitted before predict()"
        return {tgt: self.models[tgt].predict(X) for tgt in self.TARGET_NAMES}

    def predict_single(self, task: dict, resource: dict) -> Dict[str, float]:
        """Predict runtime / energy / cost for *one* (task, resource) pair."""
        feat = self.featurize(task, resource).reshape(1, -1)
        preds = self.predict(feat)
        return {k: float(v[0]) for k, v in preds.items()}

    def evaluate_candidates(
        self,
        task: dict,
        resources: List[dict],
        resource_loads: np.ndarray,
        alpha: float = 1.0,
        beta: float = 0.01,
        gamma: float = 0.01,
        delta: float = 5.0,
    ) -> List[dict]:
        """
        Score every resource for the given task and return a list of dicts
        sorted best‑first (lowest scalarised cost).
        """
        candidates = []
        for idx, res in enumerate(resources):
            preds = self.predict_single(task, res)
            finish = resource_loads[idx] + preds["runtime"]
            sla_miss = 1.0 if finish > task["deadline"] else 0.0
            score = (
                alpha * preds["runtime"]
                + beta * preds["energy"]
                + gamma * preds["cost"]
                + delta * sla_miss
            )
            candidates.append({
                "resource_idx": idx,
                "resource_id": res["id"],
                "predicted_runtime": preds["runtime"],
                "predicted_energy": preds["energy"],
                "predicted_cost": preds["cost"],
                "predicted_finish": finish,
                "sla_miss": bool(sla_miss),
                "score": score,
            })
        candidates.sort(key=lambda c: c["score"])
        return candidates

    # ── persistence ──────────────────────────────────────────────────
    def save(self, path: str = "models/dt_predictor.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"models": self.models, "fitted": self.is_fitted}, path)

    def load(self, path: str = "models/dt_predictor.joblib"):
        data = joblib.load(path)
        self.models = data["models"]
        self.is_fitted = data["fitted"]
