import csv
import time
from statistics import mean

class MetricsCollector:
    def __init__(self, filename="logs/metrics.csv"):
        self.filename = filename
        self.records = []

    def log_task(self, task_id, resource_id, start_time, finish_time,
                 energy, cost, deadline, sla_miss):
        self.records.append({
            "task_id": task_id,
            "resource_id": resource_id,
            "start_time": start_time,
            "finish_time": finish_time,
            "latency": finish_time - start_time,
            "energy": energy,
            "cost": cost,
            "deadline": deadline,
            "sla_miss": sla_miss
        })

    def save(self):
        keys = self.records[0].keys()
        with open(self.filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)
    def compute_metrics(self, alpha=1, beta=1, gamma=1, delta=1):
        L_lat = mean([r["latency"] for r in self.records])
        L_energy = mean([r["energy"] for r in self.records])
        L_cost = mean([r["cost"] for r in self.records])
        L_sla = sum([1 for r in self.records if r["sla_miss"]]) / len(self.records)

        J = alpha*L_lat + beta*L_energy + gamma*L_cost + delta*L_sla
        return {
            "L_lat": L_lat,
            "L_energy": L_energy,
            "L_cost": L_cost,
            "L_sla": L_sla,
            "Objective": J
        }
