import random
from mec_node import MecNode
import app


def selectStartingUEPosition(n_rans):
    return random.randint(1, n_rans)


class MecApp:
    def __init__(self, app_req_cpu, app_req_memory, app_req_latency, tau, n_rans):
        self.app_req_cpu = app_req_cpu
        self.app_req_memory = app_req_memory
        self.app_req_latency = app_req_latency
        self.tau = tau
        self.user_position = selectStartingUEPosition(n_rans)

    def LatencyOK(self, mec):
        if mec.latency_matrix[self.user_position] < self.app_req_latency:
            return True
        else:
            return False

    def ResourcesOK(self, mec):
        if mec.memory_available < self.app_req_memory < self.tau * mec.memory_capacity and mec.cpu_available < self.app_req_cpu < self.tau * mec.cpu_capacity:
            return True
        else:
            return False




