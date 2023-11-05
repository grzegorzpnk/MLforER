import random
import numpy as np
import sys
import gym
import json

sys.modules["gym"] = gym


class EdgeRelEnv(gym.Env):

    def __init__(self, configPath, _min_trajectory_length, _max_trajectory_length, _initial_load):

        # read initial topology config from file
        self.cnt = None
        self.reward_episode = 0
        self.mec_nodes = self._readMECnodesConfig(configPath)
        self.number_of_RANs = len(self.mec_nodes[0].latency_array)
        self.current_step = 0
        self.apps = []
        self.current_app = None

        # self.mecApp = None
        self.state = {}
        self.mobilityStateMachine = self._generateStateMachine()
        self.min_trajectory_length = _min_trajectory_length
        self.max_trajectory_length = _max_trajectory_length

        self.done = False
        self.episodes_counter = 0
        self.relocations_done = 0
        self.relocations_skipped = 0

        ################## ACTION SPACE ####################################
        self.action_space = gym.spaces.Discrete(n=len(self.mec_nodes))

        ################## OBSERVABILITY SPACE ####################################
        # : 1) CPU Capacity 2) CPU Utilization [%] 3) Memory Capacity 4) Memory Utilization [%] 5) Unit Cost
        low_bound_mec = np.zeros((len(self.mec_nodes), 5))  # initialize a 3x5 array filled with zeros
        low_bound_mec[:, 0] = 0  # Low bound of CPU Utilization is 0%
        low_bound_mec[:, 1] = 1  # App CPU req as a utilization %
        low_bound_mec[:, 2] = 0  # Low bound of Memory Utilization is 0%
        low_bound_mec[:, 3] = 1  # App MEM req as a utilization %
        low_bound_mec[:, 4] = 1  # offered latency towards current network Cell, todo: this should be continous float space, instead of int
        # low_bound_mec[:, 2] = 1  # Low bound of unit cost is 1 ( == 0.33 == city-level)

        high_bound_mec = np.ones((len(self.mec_nodes), 5))  # initialize a 3x5 array filled with zeros
        high_bound_mec[:, 0] = 100  # High bound of CPU Utilization is 100 [%]
        high_bound_mec[:, 1] = 25  # App CPU req as a utilization %
        high_bound_mec[:, 2] = 100  # High bound of MEM Utilization is 100 [%]
        high_bound_mec[:, 3] = 25  # App MEM req as a utilization %
        high_bound_mec[:, 4] = 30  # offered latency towards current network Cell
        # high_bound_mec[:, 2] = 3  # High bound of unit cost is 3 ( == 1 == international-level)

        # 1Dimension Box that represent single application attributea
        low_bound_app = np.ones((1, 1))  # initialize a 1x5 array filled with ones
        low_bound_app[:, 0] = 15

        high_bound_app = np.ones((1, 1))  # initialize a 1x5 array filled with ones
        high_bound_app[:, 0] = 30  # high bound of Required Latency 1:10, 2:15, 3:30
        # high_bound_app[:, 1] = len(self.mec_nodes)  # high bound of CurrentMEC
        # high_bound_app[:, 4] = self.number_of_RANs  # high bound of CurrentRAN

        self.observation_space = gym.spaces.dict.Dict(
            {
                # MEC(for MEC each)    : 1) CPU Utilization [%] 2) Memory Utilization [%] 3) Unit Cost 4) offered latency towards current network Cell
                # APP(for single app)  : 1) Required mvCPU 2) required Memory 3) Required Latency 4) Current MEC 5) Current RAN
                "space_MEC": gym.spaces.Box(shape=low_bound_mec.shape, dtype=np.int32, low=low_bound_mec,
                                            high=high_bound_mec),
                "space_App": gym.spaces.Box(shape=low_bound_app.shape, dtype=np.int32, low=low_bound_app,
                                            high=high_bound_app)
            }
        )

    def get_state(self):

        space_MEC = np.zeros((len(self.mec_nodes), 5))

        for i, mec_node in enumerate(self.mec_nodes):
            if mec_node.id == self.current_app.current_MEC.id:
                space_MEC[i, 0] = int(
                    mec_node.cpu_utilization - (self.current_app.app_req_cpu * 100 / mec_node.cpu_capacity))
                space_MEC[i, 2] = int(
                    mec_node.memory_utilization - (self.current_app.app_req_memory * 100 / mec_node.memory_capacity))
            else:
                space_MEC[i, 0] = mec_node.cpu_utilization
                space_MEC[i, 2] = mec_node.memory_utilization
            space_MEC[i, 1] = int(self.current_app.app_req_cpu * 100 / mec_node.cpu_capacity)
            space_MEC[i, 3] = int(self.current_app.app_req_memory * 100 / mec_node.memory_capacity)
            space_MEC[i, 4] = int(mec_node.latency_array[self.current_app.user_position - 1])

        space_App = np.ones((1, 1))
        space_App[0, 0] = self.current_app.app_req_latency

        self.state["space_App"] = space_App
        self.state["space_MEC"] = space_MEC

        return self.state

    def reset(self):

        # summarize what was achieved in previouse episode
        print("Cummulate reward:", self.reward_episode)
        print("Mobilities: ")
        for app in self.apps:
            print("APP ID: ", app.app_id, " -> ", app.trajectory)

        # let's recreate all environement
        self.done = False
        self.reward_episode = 0
        self.current_step = 0
        self.episodes_counter += 1
        print("\n")
        print(self.episodes_counter, "New episode")

        # generate initial load - apps and its clusters
        self._resetMECsState()
        self._generateInitialLoad()

        self.current_app = random.choice(self.apps)
        self.current_app.user_position = random.choice(
            self.mobilityStateMachine.get(str(self.current_app.user_position), []))
        self.current_app.trajectory.append(self.current_app.user_position)

        self.current_step += 1
        print("App: ", self.current_app.app_id, " has moved toward cell: ", self.current_app.user_position)

        self.state = self.get_state()

        return self.state

    def step(self, action):

        action += 1  # + 1 because gym implements action space as a discrete 0,1,2,.., we need to update for MEC ID: 1,2,3...
        print("Selected MEC: ", action)
        print(self.get_state())
        self.current_step += 1

        # check first if selected MEC is a current MEC
        currentNode = self._getMecNodeByID(self.current_app.current_MEC.id)
        targetNode = self._getMecNodeByID("mec" + str(action))

        # print(self.current_step,". Current node:", currentNode.id, " Cpu util: ", currentNode.cpu_utilization, "Latency: ", currentNode.latency_array[self.mecApp.user_position - 1],
        #       ", Selected node: ",targetNode.id , " Cpu util: ", targetNode.cpu_utilization, "Latency: ", targetNode.latency_array[self.mecApp.user_position - 1])

        # resourceCPUPenalty = max(0, self.mecApp.app_req_cpu - targetNode.cpu_available) / self.mecApp.app_req_cpu
        # resourceMEMPenalty = max(0, self.mecApp.app_req_memory - targetNode.memory_available) / self.mecApp.app_req_memory
        # latencyReward = (self.mecApp.app_req_latency - targetNode.latency_array[self.mecApp.user_position - 1]) / self.mecApp.app_req_latency

        # Penalty = latencyPenalty+resourceMEMPenalty+resourceCPUPenalty
        reward = 0
        # if self.current_app.CPU_OK(targetNode):
        #     reward += 0.5
        #     print("Selected Good CPU Cluster: MEC", action)
        # else:
        #     reward += -5
        #     print("Selected Bad CPU Cluster: MEC", action)
        # if self.current_app.MEM_OK(targetNode):
        #     reward += 0.5
        #     print("Selected Good MEM Cluster: MEC", action)
        # else:
        #     reward += -5
        #     print("Selected Bad MEM Cluster: MEC", action)
        if self.current_app.LatencyOK(targetNode):
            reward += 1
            print("Selected Good LAT Cluster: MEC", action)
        else:
            reward += -10
            print("Selected Bad LAT Cluster: MEC", action)
        # if reward == 2:
        #     self._relocateApplication(currentNode, targetNode)
        #
        if self.current_step >= self.min_trajectory_length:
            self.done = True
        else:
            self.current_app = random.choice(self.apps)
            self.current_app.user_position = random.choice(
                self.mobilityStateMachine.get(str(self.current_app.user_position), []))
            self.current_app.trajectory.append(self.current_app.user_position)
            self.current_step += 1
            print("App: ", self.current_app.app_id, " has moved toward cell: ", self.current_app.user_position)

        # Update the state of the environment
        self.state = self.get_state()

        self.reward_episode += reward
        # Return the new state, the reward, and whether the episode is finished
        return self.state, reward, self.done, {}


    def determineMecID(self, mecName):
        return int(mecName[3:])

    def _printAllMECs(self):
        print(self.mec_nodes)

    def _getMecNodeByID(self, id):
        for node in self.mec_nodes:
            if node.id == id:
                return node
        # if the mec node with the given ID is not found, return None
        return None

    def _printInintialLoad(self):
        print("initial load:")
        for mec in self.mec_nodes:
            print("mec: ", mec.id, "cpu: ", mec.cpu_utilization, mec.cpu_available, "memory: ", mec.memory_utilization,
                  mec.memory_available)

    def _readMECnodesConfig(self, filename):
        mec_nodes = []
        with open(filename, "r") as mec:
            config = json.load(mec)

            for item in config:
                mec_node = MecNode(
                    item['id'],
                    item['cpu_capacity'],
                    item['memory_capacity'],
                    item['cpu_utilization'],
                    item['memory_utilization'],
                    item['latency_array'],
                    item['placement_cost']
                )
                mec_nodes.append(mec_node)

        return mec_nodes

    def _resetMECsState(self):
        for mec in self.mec_nodes:
            if mec.placement_cost == 1:
                mec.cpu_utilization = 1552 * 100 / mec.cpu_capacity
                mec.memory_utilization = 1112 * 100 / mec.memory_capacity
            elif mec.placement_cost == 0.66667:
                mec.cpu_utilization = 1200 * 100 / mec.cpu_capacity
                mec.memory_utilization = 1080 * 100 / mec.memory_capacity
            elif mec.placement_cost == 0.33334:
                mec.cpu_utilization = 1548 * 100 / mec.cpu_capacity
                mec.memory_utilization = 1080 * 100 / mec.memory_capacity

            mec.cpu_used = int(mec.cpu_capacity * mec.cpu_utilization / 100)
            mec.cpu_available = mec.cpu_capacity - mec.cpu_used

            mec.memory_used = int(mec.memory_capacity * mec.memory_utilization / 100)
            mec.memory_available = mec.memory_capacity - mec.memory_used

    def _generateInitialLoad(self):


        # print("Iniital state:")
        # for mec in self.mec_nodes:
        #     print("MEC ID: ", mec.id, "CPU UTIL: ", mec.cpu_utilization, "CPU USED: ", mec.cpu_used, "CPU Ava:", mec.cpu_available, "MEM UTIL:", mec.memory_utilization)

        self.apps = self.generateApps()

        search = True
        failedCnt = 0
        while search:
            search = False
            for app in self.apps:
                candidates = self.findCandidates(app)
                if candidates:
                    selected_mec = random.choice(candidates)
                    self.instantiateApp2(selected_mec, app)
                else:
                    print("cannot find initial palcement, next iter")
                    self._resetMECsState()
                    search = True
                    failedCnt += 1
                    if failedCnt > 20:
                        print("new set of apps generated")
                        self.apps = self.generateApps()
                    break
        # print("Succesfully found initial iteration:")
        # for mec in self.mec_nodes:
        #     print("MEC ID: ", mec.id, "CPU UTIL: ", mec.cpu_utilization, "CPU USED: ", mec.cpu_used, "CPU Ava:", mec.cpu_available, "MEM UTIL:", mec.memory_utilization)

    def generateApps(self):
        apps = []
        for i in range(17):
            app_id = i + 1
            app_req_cpu = random.randint(531, 1050)
            app_req_memory = random.randint(531, 1050)
            app_req_latency = 10
            tau = 0.8
            user_position = random.randint(1, 42)
            mec_app_instance = MecApp(app_id, app_req_cpu, app_req_memory, app_req_latency, tau, user_position)
            apps.append(mec_app_instance)

        for i in range(17):
            app_id = i + 18
            app_req_cpu = random.randint(531, 1050)
            app_req_memory = random.randint(531, 1050)
            app_req_latency = 15
            tau = 0.8
            user_position = random.randint(1, 42)
            mec_app_instance = MecApp(app_id, app_req_cpu, app_req_memory, app_req_latency, tau, user_position)
            apps.append(mec_app_instance)

        for i in range(16):
            app_id = i + 35
            app_req_cpu = random.randint(531, 1050)
            app_req_memory = random.randint(531, 1050)
            app_req_latency = 30
            tau = 0.8
            user_position = random.randint(1, 42)
            mec_app_instance = MecApp(app_id, app_req_cpu, app_req_memory, app_req_latency, tau, user_position)
            apps.append(mec_app_instance)

        return apps

    def findCandidates(self, app):
        candidates = []
        for mec in self.mec_nodes:
            if app.LatencyOK(mec) and app.ResourcesOK(mec):
                candidates.append(mec)
                if mec.placement_cost == 0.66667:
                    candidates.append(mec)
                    candidates.append(mec)
                    candidates.append(mec)
                if mec.placement_cost == 0.33334:
                    candidates.append(mec)
                    candidates.append(mec)

        return candidates

    def _relocateApplication(self, currentNode, targetNode):
        # print("BEFORE REL:\n")
        # print(self.get_state())
        self.deleteApp2(currentNode, self.current_app)
        IsAppSuccasfullyInstantiated = self.instantiateApp2(targetNode, self.current_app)

        if IsAppSuccasfullyInstantiated:
            print("Relocated to MEC:  ", targetNode.id)
            return True

        # print("AFTER REL:\n")
        print(self.get_state())
        return False

    def instantiateApp2(self, mecNode, app):

        mecNode.cpu_used += app.app_req_cpu
        mecNode.cpu_available -= app.app_req_cpu
        mecNode.cpu_utilization = int(mecNode.cpu_used / mecNode.cpu_capacity * 100)

        mecNode.memory_used += app.app_req_memory
        mecNode.memory_available -= app.app_req_memory
        mecNode.memory_utilization = int(mecNode.memory_used / mecNode.memory_capacity * 100)

        if mecNode.memory_utilization > 100 | mecNode.cpu_utilization > 100:
            print("Error!", mecNode.id, "has not enough capacity to host this application!")
            self.deleteApp2(mecNode, app)
            return False

        app.current_MEC = mecNode
        return True


    def deleteApp2(self, mecNode, app):
        # print("BEFORE:\n")
        # print(self.get_state())
        mecNode.cpu_used -= app.app_req_cpu
        mecNode.cpu_available += app.app_req_cpu
        mecNode.cpu_utilization = int(mecNode.cpu_used / mecNode.cpu_capacity * 100)

        # take care of Memory
        mecNode.memory_used -= app.app_req_memory
        mecNode.memory_available += app.app_req_memory
        mecNode.memory_utilization = int(mecNode.memory_used / mecNode.memory_capacity * 100)

        app.current_MEC = ""

    def calculateReward(self):

        mec = self.mecApp.current_MEC
        cost = (mec.cpu_utilization + mec.memory_utilization)  # [0-100] + [0-100]
        normalized_cost = cost / 200  # [0-1]
        reward = (1 - normalized_cost) / mec.placement_cost  # inter: 0.333 , regional: 0.666,  city-level: 1

        return reward

    def _generateStateMachine(self):

        mobilityStateMachine = {
            "1": [3],
            "5": [2, 7],
            "8": [1],
            "2": [11, 14],
            "4": [11, 14],
            "6": [1, 2],
            "9": [4, 7],
            "3": [12],
            "7": [12, 15],
            "10": [8],
            "11": [13, 16],
            "14": [13, 16],
            "17": [6, 9],
            "20": [6, 9],
            "12": [22],
            "15": [22, 25],
            "18": [10],
            "13": [23, 26],
            "16": [23, 26],
            "19": [17, 20],
            "21": [17, 20],
            "22": [24],
            "25": [15, 18, 24, 27],
            "29": [18],
            "23": [32, 35],
            "26": [32, 35],
            "28": [19, 21],
            "30": [19, 21],
            "24": [33],
            "27": [25, 29],
            "31": [29],
            "32": [34],
            "35": [37],
            "38": [28, 30],
            "41": [28, 30],
            "33": [39],
            "36": [27, 31],
            "39": [31],
            "34": [36, 40],
            "37": [39, 42],
            "40": [38, 41],
            "42": [38, 41],
        }

        return mobilityStateMachine

    def print_summary(self):
        print("----------------------------------------------------")
        print(f"\t- Total no. of episodes: {self.episodes_counter}")
        print(f"\t- Total no. of done relocations: {self.relocations_done}")
        print(f"\t- Total no. of skipped relocations: {self.relocations_skipped}")
        print("----------------------------------------------------")


class MecNode:
    # utilization -> percentage
    def __init__(self, id, cpu_capacity, memory_capacity, cpu_utilization, memory_utilization, latency_array,
                 placement_cost):
        self.id = id
        self.cpu_capacity = cpu_capacity
        self.cpu_utilization = cpu_utilization
        self.cpu_used = int(self.cpu_capacity * self.cpu_utilization / 100)
        self.cpu_available = self.cpu_capacity - self.cpu_used

        self.memory_capacity = memory_capacity
        self.memory_utilization = memory_utilization
        self.memory_used = int(self.memory_capacity * self.memory_utilization / 100)
        self.memory_available = self.memory_capacity - self.memory_used

        self.latency_array = latency_array
        self.placement_cost = placement_cost


class MecApp:
    def __init__(self, app_id, app_req_cpu, app_req_memory, app_req_latency, tau, user_position):
        self.app_id = app_id
        self.app_req_cpu = app_req_cpu
        self.app_req_memory = app_req_memory
        self.app_req_latency = app_req_latency
        self.tau = tau
        self.user_position = user_position
        self.trajectory = []
        self.trajectory.append(user_position)
        self.current_MEC = None

    def LatencyOK(self, mec):
        # -1 cause latency_array[0] refers to the cell 1 etc..
        if mec.latency_array[self.user_position - 1] <= self.app_req_latency:
            # print(mec.id, "latencyOK. Lat oferred: ", mec.latency_array[self.user_position - 1], "while required by app: ", self.app_req_latency )
            return True
        else:
            # print(mec.id, "latencyNOToK. Lat oferred: ", mec.latency_array[self.user_position - 1], "while required by app: ", self.app_req_latency )
            return False

    def ResourcesOK(self, mec):
        # if considered mec is a current mec for app esources are definitely true, no need to check
        if mec == self.current_MEC:
            return True
        if mec.cpu_available <= self.app_req_cpu:
            # print("resources NOT OK. available CPU: ", mec.cpu_available, "while requested by app: ", self.app_req_cpu )
            # print("resources NOT OK. available MEM: ", mec.memory_available, "while requested by app: ", self.app_req_memory)
            return False
        if mec.memory_available <= self.app_req_memory:
            # print("resources NOT OK. available MEM: ", mec.memory_available, "while requested by app: ", self.app_req_memory )
            return False
        if (mec.cpu_utilization + self.app_req_cpu / mec.cpu_capacity * 100) <= self.tau * 100 and \
                (mec.memory_utilization + self.app_req_memory / mec.memory_capacity * 100) <= self.tau * 100:
            # print("resources OK. ")
            return True
        else:
            # print("resources NOT OK. ")
            return False

    def CPU_OK(self, mec):
        if mec == self.current_MEC:
            return True
        if (mec.cpu_utilization + self.app_req_cpu / mec.cpu_capacity * 100) <= self.tau * 100:
            return True
        else:
            # print("BAD CPU status")
            return False

    def MEM_OK(self, mec):
        if mec == self.current_MEC:
            return True
        if (mec.memory_utilization + self.app_req_memory / mec.memory_capacity * 100) <= self.tau * 100:
            return True
        else:
            # print("BAD MEM status")
            return False
# #
# max_trajectory_length = 25
# min_trajectory_length = 25
# initial_load = 'variable_load' # low (10-40%), medium(40-60%)), high(60-80%), random (10-80%), variable_load ( different initial load for each episode)
#
# # create environment
#
# erEnv = EdgeRelEnv("topoconfig.json", min_trajectory_length, max_trajectory_length, initial_load)
# # #
# # env = EdgeRelEnv("topoconfig.json")
# erEnv.reset()
# erEnv.step(1)
# erEnv.step(5)
# erEnv.step(10)
# # env.calculateReward2(True)
# # print(env.action_masks())

# Create an instance of the MecApp class
# app = MecApp(app_req_cpu=2, app_req_memory=2048, app_req_latency=10, tau=1, user_position=(10, 20))
#
# # Accessing object attributes
# print("CPU Requirement:", app.app_req_cpu)
# print("Memory Requirement:", app.app_req_memory)
# print("Latency Requirement:", app.app_req_latency)
# print("Tau:", app.tau)
# print("User Position:", app.user_position)
# print("Current MEC:", app.current_MEC)
