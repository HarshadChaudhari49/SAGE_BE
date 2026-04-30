"""
SAGE Real-Time Simulation Server
==================================
Runs a continuous simulation of task scheduling using the SAGE agent and
streams live analytics to a web dashboard via Server-Sent Events (SSE).

Features:
  - Continuously generates random tasks at configurable intervals
  - Schedules each task using the trained SAGE agent (PPO + DT + Explainer)
  - Streams per-task results, per-resource stats, and global metrics live
  - Serves a web dashboard at http://localhost:5000

Usage:
    python simulation_server.py
    # then open http://localhost:5000 in your browser
"""

import os
import sys
import json
import time
import random
import threading
import queue
from datetime import datetime
from collections import deque
from statistics import mean

from flask import Flask, Response, jsonify, render_template, request

# ── project imports ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.envs.sage_env import SageEnv
from src.dt.predictor import DTPredictor
from src.agents.sage_agent import SageAgent
from src.agents.explainer import SageExplainer
from stable_baselines3 import PPO

# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

TASK_INTERVAL_MIN = 0.5        # seconds between tasks (min)
TASK_INTERVAL_MAX = 2.0        # seconds between tasks (max)
TASKS_PER_EPISODE = 20         # tasks before episode reset
MAX_HISTORY = 200              # rolling window for charts

RESOURCE_CONFIGS = [
    {"id": "CPU-Node-1",  "speed": 10, "energy_rate": 0.5, "cost_rate": 0.2,
     "type": "CPU",  "location": "Rack-A"},
    {"id": "GPU-Node-1",  "speed": 25, "energy_rate": 0.9, "cost_rate": 0.5,
     "type": "GPU",  "location": "Rack-B"},
    {"id": "CPU-Node-2",  "speed": 15, "energy_rate": 0.6, "cost_rate": 0.3,
     "type": "CPU",  "location": "Rack-A"},
    {"id": "Edge-Node-1", "speed": 20, "energy_rate": 0.8, "cost_rate": 0.4,
     "type": "Edge", "location": "Edge-Site-1"},
]

TASK_TYPES = ["inference", "training", "data-pipeline", "batch-job", "web-request"]

# ═══════════════════════════════════════════════════════════════════
#  Shared State
# ═══════════════════════════════════════════════════════════════════

class SimulationState:
    """Thread-safe shared state between simulation thread and Flask."""

    def __init__(self):
        self.lock = threading.Lock()
        self.task_history = deque(maxlen=MAX_HISTORY)
        self.resource_stats = {}
        self.global_metrics = {
            "total_tasks": 0,
            "total_episodes": 0,
            "avg_latency": 0,
            "avg_energy": 0,
            "avg_cost": 0,
            "sla_miss_rate": 0,
            "throughput": 0,
        }
        self.latency_series = deque(maxlen=MAX_HISTORY)
        self.energy_series = deque(maxlen=MAX_HISTORY)
        self.cost_series = deque(maxlen=MAX_HISTORY)
        self.sla_series = deque(maxlen=MAX_HISTORY)
        self.throughput_series = deque(maxlen=MAX_HISTORY)
        self.resource_load_series = {r["id"]: deque(maxlen=MAX_HISTORY) for r in RESOURCE_CONFIGS}

        # SSE event queue (multiple clients can connect)
        self.event_queues: list[queue.Queue] = []
        self.running = False
        self.speed_multiplier = 1.0

        # Initialise resource stats
        for r in RESOURCE_CONFIGS:
            self.resource_stats[r["id"]] = {
                "id": r["id"],
                "type": r["type"],
                "location": r["location"],
                "speed": r["speed"],
                "tasks_handled": 0,
                "total_latency": 0,
                "total_energy": 0,
                "total_cost": 0,
                "sla_misses": 0,
                "current_load": 0.0,
                "utilisation": 0.0,
            }

    def broadcast(self, event_type: str, data: dict):
        """Push an SSE event to all connected clients."""
        msg = f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"
        dead = []
        for q in self.event_queues:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            self.event_queues.remove(q)


state = SimulationState()

# ═══════════════════════════════════════════════════════════════════
#  SAGE Agent Loader
# ═══════════════════════════════════════════════════════════════════

def load_sage_agent():
    """Load the trained PPO + DT into a SageAgent."""
    ppo_path = os.path.join(PROJECT_ROOT, "models", "ppo_sage", "ppo_sage_final")
    dt_path = os.path.join(PROJECT_ROOT, "models", "dt_predictor.joblib")

    ppo_model = PPO.load(ppo_path)

    dt = DTPredictor()
    if os.path.exists(dt_path):
        dt.load(dt_path)

    explainer = SageExplainer(use_shap=False)  # SHAP off for speed
    if dt.is_fitted:
        pass  # skip SHAP tree explainers for real-time performance

    agent = SageAgent(
        ppo_model=ppo_model,
        dt_predictor=dt,
        explainer=explainer,
        top_k=3,
    )
    return agent


# ═══════════════════════════════════════════════════════════════════
#  Simulation Thread
# ═══════════════════════════════════════════════════════════════════

def simulation_loop(agent: SageAgent):
    """Continuously generate tasks, schedule with SAGE, and broadcast results."""
    global state
    state.running = True
    task_counter = 0
    episode_counter = 0
    start_wall = time.time()
    rng = random.Random(int(time.time()))

    env = SageEnv(
        num_tasks=TASKS_PER_EPISODE,
        resource_configs=RESOURCE_CONFIGS,
        seed=None,
    )

    while state.running:
        # ── start new episode ───────────────────────────────────
        episode_counter += 1
        obs, _ = env.reset()
        done = False
        episode_tasks = []

        state.broadcast("episode_start", {
            "episode": episode_counter,
            "timestamp": datetime.now().isoformat(),
        })

        while not done and state.running:
            task_counter += 1
            task = env.tasks[env.current_task_idx]

            # Assign a random task type for realism
            task_type = rng.choice(TASK_TYPES)

            # ── SAGE decision ─────────────────────────────────
            action, explanation = agent.decide(obs, env, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get the record from the env
            record = env.episode_records[-1]

            # Build rich event data
            now = datetime.now()
            event = {
                "task_id": f"TASK-{task_counter:05d}",
                "task_type": task_type,
                "workload": task["workload"],
                "deadline": task["deadline"],
                "resource_id": record["resource_id"],
                "resource_type": next(r["type"] for r in RESOURCE_CONFIGS if r["id"] == record["resource_id"]),
                "start_time": round(record["start_time"], 2),
                "finish_time": round(record["finish_time"], 2),
                "latency": round(record["latency"], 2),
                "energy": round(record["energy"], 2),
                "cost": round(record["cost"], 2),
                "deadline_val": record["deadline"],
                "sla_miss": record["sla_miss"],
                "reward": round(reward, 4),
                "explanation": explanation.get("summary", ""),
                "reasons": explanation.get("reason", []),
                "timestamp": now.isoformat(),
                "episode": episode_counter,
            }

            # ── update shared state ─────────────────────────
            with state.lock:
                state.task_history.append(event)

                # Update resource stats
                rs = state.resource_stats[record["resource_id"]]
                rs["tasks_handled"] += 1
                rs["total_latency"] += record["latency"]
                rs["total_energy"] += record["energy"]
                rs["total_cost"] += record["cost"]
                rs["current_load"] = round(env.resource_finish_times[action], 2)
                if record["sla_miss"]:
                    rs["sla_misses"] += 1
                if rs["tasks_handled"] > 0:
                    rs["utilisation"] = round(
                        rs["total_latency"] / (rs["tasks_handled"] * (task["workload"] / rs["speed"] + 0.01)) * 100, 1
                    )

                # Update global metrics
                total = task_counter
                state.global_metrics["total_tasks"] = total
                state.global_metrics["total_episodes"] = episode_counter

                recent = list(state.task_history)[-50:]
                if recent:
                    state.global_metrics["avg_latency"] = round(mean(t["latency"] for t in recent), 2)
                    state.global_metrics["avg_energy"] = round(mean(t["energy"] for t in recent), 2)
                    state.global_metrics["avg_cost"] = round(mean(t["cost"] for t in recent), 2)
                    state.global_metrics["sla_miss_rate"] = round(
                        sum(1 for t in recent if t["sla_miss"]) / len(recent) * 100, 1
                    )

                elapsed = time.time() - start_wall
                state.global_metrics["throughput"] = round(total / max(elapsed, 1), 2)

                # Time series
                state.latency_series.append({"x": total, "y": record["latency"]})
                state.energy_series.append({"x": total, "y": record["energy"]})
                state.cost_series.append({"x": total, "y": record["cost"]})
                state.sla_series.append({"x": total, "y": 1 if record["sla_miss"] else 0})
                state.throughput_series.append({"x": total, "y": state.global_metrics["throughput"]})

                for r in RESOURCE_CONFIGS:
                    idx = next(i for i, rc in enumerate(RESOURCE_CONFIGS) if rc["id"] == r["id"])
                    state.resource_load_series[r["id"]].append({
                        "x": total, "y": round(env.resource_finish_times[idx], 2)
                    })

            # ── broadcast to dashboard ──────────────────────
            state.broadcast("task_scheduled", event)
            state.broadcast("metrics_update", state.global_metrics)
            state.broadcast("resource_update", {
                rid: dict(rs) for rid, rs in state.resource_stats.items()
            })

            # ── pace the simulation ─────────────────────────
            delay = rng.uniform(TASK_INTERVAL_MIN, TASK_INTERVAL_MAX) / state.speed_multiplier
            time.sleep(delay)

        # Episode finished
        summary = env._episode_summary()
        summary["episode"] = episode_counter
        state.broadcast("episode_end", summary)

    state.running = False


# ═══════════════════════════════════════════════════════════════════
#  Flask Application
# ═══════════════════════════════════════════════════════════════════

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("dashboard.html", resources=RESOURCE_CONFIGS)


@app.route("/api/stream")
def stream():
    """SSE endpoint — dashboard connects here for live updates."""
    q = queue.Queue(maxsize=200)
    state.event_queues.append(q)

    def generate():
        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    yield "event: ping\ndata: {}\n\n"
        except GeneratorExit:
            if q in state.event_queues:
                state.event_queues.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/state")
def get_state():
    """Full snapshot of current simulation state."""
    with state.lock:
        return jsonify({
            "global_metrics": state.global_metrics,
            "resource_stats": state.resource_stats,
            "recent_tasks": list(state.task_history)[-30:],
            "latency_series": list(state.latency_series),
            "energy_series": list(state.energy_series),
            "cost_series": list(state.cost_series),
            "sla_series": list(state.sla_series),
            "resource_load_series": {k: list(v) for k, v in state.resource_load_series.items()},
        })


@app.route("/api/speed", methods=["POST"])
def set_speed():
    """Adjust simulation speed multiplier."""
    data = request.get_json(force=True)
    state.speed_multiplier = max(0.1, min(20.0, float(data.get("speed", 1.0))))
    return jsonify({"speed": state.speed_multiplier})


@app.route("/api/pause", methods=["POST"])
def pause():
    """Pause or resume the simulation."""
    state.running = not state.running
    return jsonify({"running": state.running})


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("=" * 60)
    print("  SAGE Real-Time Simulation Dashboard")
    print("=" * 60)
    print("  Loading SAGE agent...")

    agent = load_sage_agent()
    print("  ✓ Agent loaded (PPO + Digital Twin)")

    # Start simulation in background thread
    sim_thread = threading.Thread(target=simulation_loop, args=(agent,), daemon=True)
    sim_thread.start()
    print("  ✓ Simulation started")

    print()
    print(f"  ➜  Dashboard: http://localhost:{port}")
    print(f"  ➜  API state: http://localhost:{port}/api/state")
    print(f"  ➜  SSE stream: http://localhost:{port}/api/stream")
    print()

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
