from mec_node import MecNode
import app

class MecApp:
    def __init__(self, app_req_cpu, app_req_memory, app_req_latency, tau, user_position):
        self.app_req_cpu = app_req_cpu
        self.app_req_memory = app_req_memory
        self.app_req_latency = app_req_latency
        self.tau = tau
        self.user_position = user_position
        self.current_MEC = None

    def LatencyOK(self, mec):
        '''
        This is supportive funtion to check latency conditions, used only for initial (for init state) placement of our main app.
        This func is not used to check conditions during the relocation, since it;s responisibility of agent
        :param mec:
        :return:
        '''
        if mec.latency_array[self.user_position] < self.app_req_latency:
            return True
        else:
            return False

    def ResourcesOK(self, mec):
        '''
        This is supportive funtion to check resources conditions, used only for initial (for init state) placement of our main app.
        This func is not used to check conditions during the relocation, since it;s responisibility of agent
        :param mec:
        :return:
        '''
        if mec.memory_available < self.app_req_memory < self.tau * mec.memory_capacity and mec.cpu_available < self.app_req_cpu < self.tau * mec.cpu_capacity:
            return True
        else:
            return False




