"""
PPO Training Script for SAGE
=============================
Trains a Proximal Policy Optimisation (PPO) agent on the SageEnv Gymnasium
environment.  The trained policy learns to map (task features, resource states)
→ resource assignment that minimises a weighted combination of latency, energy,
cost, and SLA‑miss rate.

Usage
-----
    python -m src.agents.train_ppo --timesteps 50000 --save-path models/ppo_sage
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)

from src.envs.sage_env import SageEnv


# ── Custom logging callback ────────────────────────────────────────
class MetricsCallback(BaseCallback):
    """Logs episode summary metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "avg_latency" in info:
                self.logger.record("sage/avg_latency", info["avg_latency"])
                self.logger.record("sage/avg_energy", info["avg_energy"])
                self.logger.record("sage/avg_cost", info["avg_cost"])
                self.logger.record("sage/sla_miss_rate", info["sla_miss_rate"])
                self.logger.record("sage/makespan", info["makespan"])
        return True


# ── Environment factory ────────────────────────────────────────────
def make_sage_env(**kwargs):
    """Return a callable that creates a SageEnv with the given kwargs."""
    def _init():
        return SageEnv(**kwargs)
    return _init


# ── Training ───────────────────────────────────────────────────────
def train(
    total_timesteps: int = 50_000,
    n_envs: int = 4,
    save_path: str = "models/ppo_sage",
    eval_freq: int = 2_000,
    num_tasks: int = 20,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    verbose: int = 1,
):
    """Train a PPO agent on SageEnv."""

    env_kwargs = dict(num_tasks=num_tasks, seed=seed)

    # Vectorised training envs
    train_env = make_vec_env(make_sage_env(**env_kwargs), n_envs=n_envs)
    # Single eval env
    eval_env = make_vec_env(make_sage_env(**env_kwargs), n_envs=1)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=verbose,
        tensorboard_log="logs/tb_sage",
        seed=seed,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
    )

    # Callbacks
    os.makedirs(save_path, exist_ok=True)
    callbacks = [
        MetricsCallback(),
        EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=eval_freq,
            n_eval_episodes=5,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(total_timesteps // 5, 1000),
            save_path=save_path,
            name_prefix="ppo_sage",
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    final_path = os.path.join(save_path, "ppo_sage_final")
    model.save(final_path)
    print(f"✓ Model saved to {final_path}")
    return model


# ── Evaluation helper ──────────────────────────────────────────────
def evaluate(model_path: str, num_episodes: int = 10, num_tasks: int = 20):
    """Load a saved PPO model and evaluate it."""
    env = SageEnv(num_tasks=num_tasks)
    model = PPO.load(model_path)

    all_summaries = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        if "avg_latency" in info:
            all_summaries.append(info)

    # Aggregate
    if all_summaries:
        avg = {k: np.mean([s[k] for s in all_summaries]) for k in all_summaries[0]}
        print("Evaluation results (mean over episodes):")
        for k, v in avg.items():
            print(f"  {k}: {v:.4f}")
        return avg
    return {}


# ── CLI ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train SAGE PPO agent")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="models/ppo_sage")
    parser.add_argument("--eval-freq", type=int, default=2_000)
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--evaluate-only", type=str, default=None,
                        help="Path to model .zip to evaluate instead of training")
    args = parser.parse_args()

    if args.evaluate_only:
        evaluate(args.evaluate_only, num_tasks=args.num_tasks)
    else:
        train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_path=args.save_path,
            eval_freq=args.eval_freq,
            num_tasks=args.num_tasks,
            seed=args.seed,
            learning_rate=args.lr,
        )


if __name__ == "__main__":
    main()
