"""
SAGE Explainability Module
===========================
Provides human‑readable explanations for scheduling decisions.

Two complementary approaches:

1. **SHAP‑based feature importance** — uses the DT Gradient‑Boosting models
   to attribute each input feature's contribution to the predicted runtime /
   energy / cost for the chosen assignment.

2. **Contrastive explanation** — compares the chosen resource against the
   best alternative and summarises the trade‑offs.

Output is a JSON‑serialisable dict that can be logged, displayed in a
dashboard, or returned via the scheduler REST API.
"""

import numpy as np
from typing import Dict, List, Optional

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


class SageExplainer:
    """Generates human‑readable explanations for SAGE decisions."""

    FEATURE_NAMES = ["workload", "deadline", "speed", "energy_rate", "cost_rate"]

    def __init__(self, use_shap: bool = True):
        self.use_shap = use_shap and _SHAP_AVAILABLE
        self._shap_explainers: Dict[str, object] = {}

    # ── build SHAP explainers from a fitted DT ─────────────────────
    def build_shap_explainers(self, dt_predictor):
        """
        Build TreeExplainer objects for each target model in the DT.
        Call this once after the DT has been fitted.
        """
        if not _SHAP_AVAILABLE:
            return
        for tgt, model in dt_predictor.models.items():
            self._shap_explainers[tgt] = shap.TreeExplainer(model)

    # ── main explain method ─────────────────────────────────────────
    def explain_decision(
        self,
        task: dict,
        resources: List[dict],
        chosen_idx: int,
        candidates: List[dict],
        dt_predictor=None,
    ) -> dict:
        """
        Produce an explanation dict for why ``resources[chosen_idx]`` was
        selected for ``task``.
        """
        chosen_res = resources[chosen_idx]
        explanation: dict = {
            "task_id": task.get("id", "?"),
            "chosen_resource": chosen_res["id"],
            "reason": [],
        }

        # ── contrastive explanation ─────────────────────────────────
        if len(candidates) > 1:
            best = candidates[0]
            for alt in candidates[1:]:
                delta_rt = alt.get("predicted_runtime", 0) - best.get("predicted_runtime", 0)
                delta_en = alt.get("predicted_energy", 0) - best.get("predicted_energy", 0)
                delta_co = alt.get("predicted_cost", 0) - best.get("predicted_cost", 0)
                explanation["reason"].append(
                    f"Preferred {best.get('resource_id','?')} over "
                    f"{alt.get('resource_id','?')}: "
                    f"runtime {delta_rt:+.2f}s, energy {delta_en:+.2f}J, cost {delta_co:+.2f}$"
                )
            explanation["candidates"] = candidates

        # ── SHAP feature importance ─────────────────────────────────
        if dt_predictor and dt_predictor.is_fitted and self.use_shap and self._shap_explainers:
            feat = dt_predictor.featurize(task, chosen_res).reshape(1, -1)
            shap_values = {}
            for tgt, explainer in self._shap_explainers.items():
                sv = explainer.shap_values(feat)[0]
                contributions = {
                    name: float(sv[i]) for i, name in enumerate(self.FEATURE_NAMES)
                }
                shap_values[tgt] = contributions
            explanation["shap"] = shap_values
        elif dt_predictor and dt_predictor.is_fitted:
            # Fallback: feature importance from GBR
            explanation["feature_importance"] = {}
            for tgt, model in dt_predictor.models.items():
                imp = model.feature_importances_
                explanation["feature_importance"][tgt] = {
                    name: float(imp[i]) for i, name in enumerate(self.FEATURE_NAMES)
                }

        # ── human‑readable summary ──────────────────────────────────
        chosen_cand = next(
            (c for c in candidates if c.get("resource_idx") == chosen_idx),
            None,
        )
        if chosen_cand and "predicted_runtime" in chosen_cand:
            explanation["summary"] = (
                f"Task {task.get('id','?')} → {chosen_res['id']}  |  "
                f"pred_runtime={chosen_cand['predicted_runtime']:.2f}s  "
                f"pred_energy={chosen_cand['predicted_energy']:.2f}J  "
                f"sla_miss={chosen_cand.get('sla_miss', '?')}"
            )
        else:
            explanation["summary"] = (
                f"Task {task.get('id','?')} → {chosen_res['id']} (PPO policy choice)"
            )

        return explanation

    # ── batch SHAP plot (for analysis notebooks) ────────────────────
    def plot_shap_summary(self, dt_predictor, X: np.ndarray, target: str = "runtime"):
        """
        Generate a SHAP summary plot for the specified target model.
        ``X`` should be the feature matrix used during training / evaluation.
        """
        if not _SHAP_AVAILABLE:
            print("SHAP is not installed — cannot generate plots.")
            return
        if target not in self._shap_explainers:
            self.build_shap_explainers(dt_predictor)
        explainer = self._shap_explainers.get(target)
        if explainer is None:
            print(f"No explainer for target '{target}'")
            return
        sv = explainer.shap_values(X)
        shap.summary_plot(sv, X, feature_names=self.FEATURE_NAMES, show=True)
