import itertools
import random

class RoundRobinScheduler:
    def __init__(self, resources):
        self.resources = resources
        self._cycle = itertools.cycle(resources)
    
    def schedule(self, task):
        return next(self._cycle)


class ShortestExpectedRuntimeScheduler:
    def __init__(self, resources):
        self.resources = resources
    
    def schedule(self, task):
        """
        runtime = task.workload / resource.speed
        """
        runtimes = [(r, task.workload / r.speed) for r in self.resources]
        return min(runtimes, key=lambda x: x[1])[0]


class MinMinScheduler:
    def __init__(self, resources):
        self.resources = resources
    
    def schedule(self, task_list):
        assignment = {}
        for task in task_list:
            runtimes = [(r, task.workload / r.speed) for r in self.resources]
            best_res = min(runtimes, key=lambda x: x[1])[0]
            assignment[task] = best_res
        return assignment


class RandomScheduler:
    def __init__(self, resources):
        self.resources = resources

    def schedule(self, task):
        return random.choice(self.resources)
