import random
import time
from src.simulator.schedulers import (
    RoundRobinScheduler,
    ShortestExpectedRuntimeScheduler,
    MinMinScheduler,
    RandomScheduler
)
from src.utils.metrics import MetricsCollector
class Resource:
    def __init__(self, resource_id, speed, energy_rate, cost_rate):
        self.id = resource_id
        self.speed = speed            
        self.energy_rate = energy_rate
        self.cost_rate = cost_rate


class Task:
    def __init__(self, task_id, workload, deadline):
        self.id = task_id
        self.workload = workload
        self.deadline = deadline

class Simulator:
    def __init__(self, tasks, resources, scheduler, metrics_file="logs/metrics.csv"):
        self.tasks = tasks
        self.resources = resources
        self.scheduler = scheduler
        self.metrics = MetricsCollector(metrics_file)

    def run(self):
        current_time = 0
        for task in self.tasks:
            if hasattr(self.scheduler, "schedule") and "task_list" in self.scheduler.schedule.__code__.co_varnames:
                assignment = self.scheduler.schedule(self.tasks)
                for task, res in assignment.items():
                    self._execute_task(task, res, current_time)
                break
            else:
                res = self.scheduler.schedule(task)
                self._execute_task(task, res, current_time)

        self.metrics.save()
        return self.metrics.compute_metrics()

    def _execute_task(self, task, res, current_time):
        exec_time = task.workload / res.speed
        finish_time = current_time + exec_time

        energy = task.workload * res.energy_rate
        cost = task.workload * res.cost_rate
        sla_miss = finish_time > task.deadline

        self.metrics.log_task(
            task_id=task.id,
            resource_id=res.id,
            start_time=current_time,
            finish_time=finish_time,
            energy=energy,
            cost=cost,
            deadline=task.deadline,
            sla_miss=sla_miss
        )

        return finish_time
