import simpy
import random
import time
from collections import namedtuple

Task = namedtuple('Task', ['id', 'arrival', 'work', 'deadline', 'affinity'])

class ResourceNode:
    def __init__(self, env, name, speed, power):
        self.env = env
        self.name = name
        self.speed = speed
        self.power = power
        self.proc = simpy.Resource(env, capacity=1)

def run_tasks(env, task, node, record):
    with node.proc.request() as req:
        yield req
        start = env.now
        exec_time = task.work / node.speed
        yield env.timeout(exec_time)
        finish = env.now
        energy = exec_time * node.power
        record.append({'task':task.id, 'node':node.name, 'start':start, 'finish':finish, 'energy':energy})

def generator(env, tasks, nodes, record, scheduler):
    for t in tasks:
        yield env.timeout(t.arrival - env.now)
        node = scheduler.choose_node(t, nodes, env, record)
        env.process(run_tasks(env, t, node, record))

class SimpleScheduler:
    def choose_node(self, task, nodes, env, record):
        candidates = [n for n in nodes if task.affinity in n.name]
        if not candidates: candidates = nodes
        return max(candidates, key=lambda n: n.speed)
    
def example():
    env = simpy.Environment()
    nodes = [ResourceNode(env, 'cpu', speed=1e6, power=50), ResourceNode(env, 'gpu', speed=5e6, power=200)]
    tasks = [Task(id=i, arrival=i*2, work=random.uniform(1e6, 5e6), deadline=(i+1)*10, affinity='gpu' if i%2==0 else 'cpu') for i in range(10)]
    record = []
    scheduler = SimpleScheduler()
    env.process(generator(env, tasks, nodes, record, scheduler))
    env.run()
    for r in record:
        print(r)

if __name__ == "__main__":
    example()