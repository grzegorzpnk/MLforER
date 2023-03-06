
class MecNode:
    #utilization -> precentage
    def __init__(self, id, cpu_capacity, memory_capacity, cpu_utilization, memory_utilization, latency_matrix,
                 placement_cost):
        self.id = id
        self.cpu_capacity = cpu_capacity
        self.cpu_utilization = cpu_utilization
        self.cpu_available = cpu_capacity - cpu_utilization / 100 * cpu_capacity
        self.memory_capacity = memory_capacity
        self.memory_utilization = memory_utilization
        self.memory_available = memory_capacity - memory_utilization / 100 * memory_capacity
        self.latency_matrix = latency_matrix
        self.placement_cost = placement_cost

