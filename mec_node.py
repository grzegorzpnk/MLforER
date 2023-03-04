
class MecNode:
    def __init__(self, id, cpu_capacity, memory_capacity, cpu_utilization, memory_utilization, latency_matrix, placement_cost):
        self.id = id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.cpu_utilization = cpu_utilization
        self.memory_utilization = memory_utilization
        self.latency_matrix = latency_matrix
        self.placement_cost = placement_cost
