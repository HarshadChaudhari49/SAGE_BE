from src.simulator.simulator import Simulator, Task, Resource
import random
from src.simulator.schedulers import RoundRobinScheduler, RandomScheduler, ShortestExpectedRuntimeScheduler, MinMinScheduler

# Define resources
resources = [
    Resource("R1", speed=10, energy_rate=0.5, cost_rate=0.2),
    Resource("R2", speed=20, energy_rate=0.8, cost_rate=0.4),
    Resource("R3", speed=15, energy_rate=0.6, cost_rate=0.3),
]

# Generate random tasks
tasks = [Task(f"T{i}", workload=random.randint(50, 200), deadline=random.randint(10, 40)) for i in range(10)]

# Choose scheduler
scheduler = RoundRobinScheduler(resources)

# Run simulation
sim = Simulator(tasks, resources, scheduler)
results = sim.run()
print("Simulation Results:", results)
