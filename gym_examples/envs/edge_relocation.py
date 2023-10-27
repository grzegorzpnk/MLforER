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

        self.trajectory = None
        self.mecApp = None
        self.state = {}
        self.mobilityStateMachine = self._generateStateMachine()
        self.min_trajectory_length = _min_trajectory_length
        self.max_trajectory_length = _max_trajectory_length
        self.initial_load = _initial_load

        self.done = False
        self.episodes_counter = 0
        self.relocations_done = 0
        self.relocations_skipped = 0

        ################## ACTION SPACE ####################################
        self.action_space = gym.spaces.Discrete(n=len(self.mec_nodes))

        ################## OBSERVABILITY SPACE ####################################

        # Multi-Dimension Box that represent every MEC host, each represented by 5 attributea
        # : 1) CPU Capacity 2) CPU Utilization [%] 3) Memory Capacity 4) Memory Utilization [%] 5) Unit Cost
        low_bound_mec = np.zeros((len(self.mec_nodes), 5))  # initialize a 3x5 array filled with zeros
        low_bound_mec[:, 0] = 1  # Low bound of CPU Capacity is 1 ( == 4000 mvCPU)
        low_bound_mec[:, 1] = 0  # Low bound of CPU Utilization is 0%
        low_bound_mec[:, 2] = 1  # Low bound of Memory Capacity is 1 ( == 4000 Mb RAM)
        low_bound_mec[:, 3] = 0  # Low bound of Memory Utilization is 0%
        low_bound_mec[:, 4] = 1  # Low bound of unit cost is 1 ( == 0.33 == city-level)

        high_bound_mec = np.ones((len(self.mec_nodes), 5))  # initialize a 3x5 array filled with zeros
        high_bound_mec[:, 0] = 3  # High bound of CPU Capacity is 3 ( == 12000 mvCPU)
        high_bound_mec[:, 1] = 100  # High bound of CPU Utilization is 100 [%]
        high_bound_mec[:, 2] = 3  # High bound of Memory Capacity is 3 ( == 12000 Mb RAM)
        high_bound_mec[:, 3] = 100  # High bound of Memory Utilization is 100 [%]
        high_bound_mec[:, 4] = 3  # High bound of unit cost is 3 ( == 1 == international-level)

        # 1Dimension Box that represent single application attributea
        # 1) Required mvCPU 2) required Memory 3) Required Latency 4) Current MEC 5) Current RAN
        low_bound_app = np.ones((1, 5))  # initialize a 1x5 array filled with ones
        high_bound_app = np.ones((1, 5))  # initialize a 1x5 array filled with ones
        high_bound_app[:, 0] = 500  # high bound of required mvCPU # 1:500, 2:600, 3:700... 6:1000
        high_bound_app[:, 1] = 500  # high bound of required Memory #1:500, 2:600, 3:700... 6:1000
        high_bound_app[:, 2] = 3  # high bound of Required Latency 1:10, 2:15, 3:30
        high_bound_app[:, 3] = len(self.mec_nodes)  # high bound of CurrentMEC
        high_bound_app[:, 4] = self.number_of_RANs  # high bound of CurrentRAN

        self.observation_space = gym.spaces.dict.Dict(
            {
                # MEC(for MEC each)    : 1) CPU Capacity 2) CPU Utilization [%] 3) Memory Capacity 4) Memory Utilization [%] 5) Unit Cost
                # APP(for single app)  : 1) Required mvCPU 2) required Memory 3) Required Latency 4) Current MEC 5) Current RAN
                "space_MEC": gym.spaces.Box(shape=low_bound_mec.shape, dtype=np.int32, low=low_bound_mec,
                                            high=high_bound_mec),
                "space_App": gym.spaces.Box(shape=low_bound_app.shape, dtype=np.int32, low=low_bound_app,
                                            high=high_bound_app)
            }
        )

    def get_state(self):

        space_MEC = np.zeros((len(self.mec_nodes), 5))

        # MEC  : 0) CPU Capacity 1) CPU Utilization [%] 2) Memory Capacity 3) Memory Utilization [%] 4) Unit Cost
        for i, mec_node in enumerate(self.mec_nodes):
            space_MEC[i, 0] = self.determineStateOfCapacity(mec_node.cpu_capacity)
            space_MEC[i, 1] = mec_node.cpu_utilization
            space_MEC[i, 2] = self.determineStateOfCapacity(mec_node.memory_capacity)
            space_MEC[i, 3] = mec_node.memory_utilization
            space_MEC[i, 4] = self.determineStateofCost(mec_node.placement_cost)

        # APP  : [0,1] Required mvCPU  [0,2] required Memory [0,3] Required Latency [0,4] Current MEC [0,5] Current RAN
        space_App = np.ones((1, 5))
        space_App[0, 0] = self.determineReqRes(self.mecApp.app_req_cpu)
        space_App[0, 1] = self.determineReqRes(self.mecApp.app_req_memory)
        space_App[0, 2] = self.determineStateofAppLatReq(self.mecApp.app_req_latency)
        space_App[0, 3] = self.determineMecID(self.mecApp.current_MEC.id)
        space_App[0, 4] = self.mecApp.user_position

        self.state["space_App"] = space_App
        self.state["space_MEC"] = space_MEC

        return self.state

    def determineReqRes(self, reqRes):
        return reqRes - 500
        # res_map = {500: 1, 600: 2, 700: 3, 800: 4, 900: 5, 1000: 6}
        # return res_map.get(reqRes, 0)

    def determineStateOfCapacity(self, capacityValue):
        capacity_map = {4000: 1, 8000: 2, 12000: 3}
        return capacity_map.get(capacityValue)

    def determineStateofCost(self, placement_cost):
        cost_map = {0.33334: 1, 0.66667: 2, 1: 3}
        return cost_map.get(placement_cost, 0)

    def determineStateofAppLatReq(self, latValue):
        # lat_map = {10: 1, 15: 2, 30: 3}
        lat_map = {10: 1, 15: 2, 30: 3}

        return lat_map.get(latValue, 0)

    def determineMecID(self, mecName):
        return int(mecName[3:])

    def _generateTrajectory(self):

        # generate initial UE position
        start_state = random.randint(1, self.number_of_RANs)
        trajectory = [start_state]
        current_state = start_state

        # generate length of trajectory
        trajectory_length = random.randint(self.min_trajectory_length, self.max_trajectory_length)
        for i in range(trajectory_length):
            next_states = self.mobilityStateMachine.get(str(current_state), [])
            if not next_states:
                break
            current_state = random.choice(next_states)
            trajectory.append(current_state)

        return trajectory

    def _generateMECApp(self):

        # Generate a value for required resources among given:
        # resources_req = [500, 600, 700, 800, 900, 1000]
        random_req_cpu = random.randint(501, 1000)
        random_req_mem = random.randint(501, 1000)
        # Define a list of three latency: 10, 15, 30
        allowed_latencies = [10, 15, 30]
        # Randomly select one number from the list
        random_latency = random.choice(allowed_latencies)

        return MecApp(random_req_cpu, random_req_mem, random_latency, 0.8, self.trajectory[0])

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

    def _generateInitialLoadForTopology(self):

        # current_scenario = self.initial_load
        current_scenario = "variable_load"

        _min = 0
        _max = 0

        if current_scenario == "variable_load":
            scenarios = ["low", "medium", "high", "random", "additional"]
            current_scenario = random.choice(scenarios)

        if current_scenario == "low":
            _min = 10
            _max = 40
        elif current_scenario == "medium":
            _min = 40
            _max = 60
        elif current_scenario == "high":
            _min = 60
            _max = 80
        elif current_scenario == "additional":
            _min = 40
            _max = 80
        elif current_scenario == "random":
            _min = 10
            _max = 90

        # print("selected min, max: ", _min, _max)
        print("Initial load:")
        for mec in self.mec_nodes:
            mec.cpu_utilization = random.randint(_min, _max)
            mec.cpu_used = int(mec.cpu_capacity * mec.cpu_utilization / 100)
            mec.cpu_available = int(mec.cpu_capacity - mec.cpu_used)
            if current_scenario == 'random':
                # this if refers only to a case where random scenario were selected.
                # I wanted to avoid situation, where cpu utilization is 10 % and mem util is 90 %,
                # so I decided to put the same utilization value for both is such a case
                mec.memory_utilization = mec.cpu_utilization
                mec.memory_used = int(mec.memory_capacity * mec.memory_utilization / 100)
                mec.memory_available = int(mec.memory_capacity - mec.memory_used)
            else:
                mec.memory_utilization = random.randint(_min, _max)
                mec.memory_used = int(mec.memory_capacity * mec.memory_utilization / 100)
                mec.memory_available = int(mec.memory_capacity - mec.memory_used)
            print(mec.id, ". CPU  Util:", mec.cpu_utilization, "CPU available: ", mec.cpu_available, "CPU Used: ",
                  mec.cpu_used, ". MEM  Util:", mec.memory_utilization, "MEM available: ", mec.memory_available,
                  "MEM Used: ", mec.memory_used)

    def _selectStartingNode(self):
        cnt = 0
        while True:
            randomMec = random.choice(self.mec_nodes)
            if self.mecApp.LatencyOK(randomMec) and self.mecApp.ResourcesOK(randomMec):
                self.instantiateApp(randomMec)
                print("Initial MEC:  ", randomMec.id)
                return randomMec
            if cnt > 1000:
                return None
            cnt += 1

    def reset(self):

        # summarize what was achieved in previouse episode
        print("Cummulate reward:", self.reward_episode)

        # let's recreate all environement
        self.done = False
        self.reward_episode = 0
        self.current_step = 1
        self.episodes_counter += 1
        print("\n")
        print(self.episodes_counter, "New episode")

        # generate initial load
        self._generateInitialLoadForTopology()

        # generate trajectory
        self.trajectory = self._generateTrajectory()

        # generateApp
        self.mecApp = self._generateMECApp()
        self.mecApp.current_MEC = self._selectStartingNode()
        while self.mecApp.current_MEC is None:
            print("Cannot find any initial cluster for app. Generating new initial load")
            self.trajectory = self._generateTrajectory()
            self.mecApp = self._generateMECApp()
            self.mecApp.current_MEC = self._selectStartingNode()

        self.mecApp.user_position = self.trajectory[self.current_step]
        print("Moved towards cell: ", self.trajectory[self.current_step])

        self.state = self.get_state()

        return self.state

    def step(self, action):

        action += 1  # + 1 because gym implements action space as a discrete 0,1,2,.., we need to update for MEC ID: 1,2,3...
        self.current_step += 1

        # check first if selected MEC is a current MEC
        currentNode = self._getMecNodeByID(self.mecApp.current_MEC.id)
        targetNode = self._getMecNodeByID("mec" + str(action))

        # print(self.current_step,". Current node:", currentNode.id, " Cpu util: ", currentNode.cpu_utilization, "Latency: ", currentNode.latency_array[self.mecApp.user_position - 1],
        #       ", Selected node: ",targetNode.id , " Cpu util: ", targetNode.cpu_utilization, "Latency: ", targetNode.latency_array[self.mecApp.user_position - 1])

        # resourceCPUPenalty = max(0, self.mecApp.app_req_cpu - targetNode.cpu_available) / self.mecApp.app_req_cpu
        # resourceMEMPenalty = max(0, self.mecApp.app_req_memory - targetNode.memory_available) / 500
        # latencyPenalty = max(0, targetNode.latency_array[self.mecApp.user_position-1]-self.mecApp.app_req_latency)/20
        # Penalty = latencyPenalty+resourceMEMPenalty+resourceCPUPenalty

        if currentNode == targetNode:
            if self.mecApp.LatencyOK(targetNode):
                print("Staying at the same MEC:", targetNode)
                reward = self.calculateReward()
            else:
                print("The same Cluster, Not relocated, bad latency")
                reward = -100
                self.done = True
        if currentNode != targetNode:
            if self.mecApp.ResourcesOK(targetNode):
                if self.mecApp.LatencyOK(targetNode):
                    self._relocateApplication(currentNode, targetNode)
                    reward = self.calculateReward()
                else:
                    print("New Cluster, Not relocated, bad latency")
                    self.done = True
                    reward = -100
            else:
                if self.mecApp.LatencyOK(targetNode):
                    print("New Cluster, Not relocated, bad resources")
                    self.done = True
                    reward = -100
                else:
                    print("New Cluster, Not relocated, bad latency, bad resources")
                    self.done = True
                    reward = -100

        if self.current_step >= len(self.trajectory):
            self.done = True
        else:
            self.mecApp.user_position = self.trajectory[self.current_step]
            print("Moved towards cell: ", self.trajectory[self.current_step])

        # Update the state of the environment
        self.state = self.get_state()

        self.reward_episode += reward
        # Return the new state, the reward, and whether the episode is finished
        return self.state, reward, self.done, {}

    def _relocateApplication(self, currentNode, targetNode):

        self.deleteApp(currentNode)
        IsAppSuccasfullyInstantiated = self.instantiateApp(targetNode)

        if IsAppSuccasfullyInstantiated:
            print("Relocated to MEC:  ", targetNode.id)
            return True

        return False

    def instantiateApp(self, mecNode):

        mecNode.cpu_used += self.mecApp.app_req_cpu
        mecNode.cpu_available -= self.mecApp.app_req_cpu
        mecNode.cpu_utilization = int(mecNode.cpu_used / mecNode.cpu_capacity * 100)

        mecNode.memory_used += self.mecApp.app_req_memory
        mecNode.memory_available -= self.mecApp.app_req_memory
        mecNode.memory_utilization = int(mecNode.memory_used / mecNode.memory_capacity * 100)

        if mecNode.memory_utilization > 100 | mecNode.cpu_utilization > 100:
            self.deleteApp(mecNode)
            print("Error!", mecNode.id, "has not enough capacity to host this application!")
            return False

        # Application update
        self.mecApp.current_MEC = mecNode
        return True

    def deleteApp(self, mecNode):
        mecNode.cpu_used -= self.mecApp.app_req_cpu
        mecNode.cpu_available += self.mecApp.app_req_cpu
        mecNode.cpu_utilization = int(mecNode.cpu_used / mecNode.cpu_capacity * 100)

        # take care of Memory
        mecNode.memory_used -= self.mecApp.app_req_memory
        mecNode.memory_available += self.mecApp.app_req_memory
        mecNode.memory_utilization = int(mecNode.memory_used / mecNode.memory_capacity * 100)

        self.mecApp.current_MEC = ""

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
    def __init__(self, app_req_cpu, app_req_memory, app_req_latency, tau, user_position):
        self.app_req_cpu = app_req_cpu
        self.app_req_memory = app_req_memory
        self.app_req_latency = app_req_latency
        self.tau = tau
        self.user_position = user_position
        self.current_MEC = None

    def LatencyOK(self, mec):
        """
        This is supportive funtion to check latency conditions, used only for initial (for init state) placement of our main app.
        This func is not used to check conditions during the relocation, since it;s responisibility of agent
        :param mec:
        :return:
        """
        # -1 cause latency_array[0] refers to the cell 1 etc..
        if mec.latency_array[self.user_position - 1] < self.app_req_latency:
            # print(mec.id, "latencyOK. Lat oferred: ", mec.latency_array[self.user_position - 1], "while required by app: ", self.app_req_latency )
            return True
        else:
            # print(mec.id, "latencyNOToK. Lat oferred: ", mec.latency_array[self.user_position - 1], "while required by app: ", self.app_req_latency )
            return False

    def ResourcesOK(self, mec):
        """
        This is supportive function to check resources conditions, used only for initial (for init state) placement of our main app.
        This func is not used to check conditions during the relocation, since it;s responisibility of agent
        :param mec:
        :return:
        """
        # if considered mec is a current mec for app esources are definitely true, no need to check
        if mec.id == self.current_MEC.id:
            return True
        if mec.cpu_available < self.app_req_cpu:
            # print("resources NOT OK. available CPU: ", mec.cpu_available, "while requested by app: ", self.app_req_cpu )
            # print("resources NOT OK. available MEM: ", mec.memory_available, "while requested by app: ", self.app_req_memory)
            return False
        if mec.memory_available < self.app_req_memory:
            # print("resources NOT OK. available MEM: ", mec.memory_available, "while requested by app: ", self.app_req_memory )
            return False
        if (mec.cpu_utilization + self.app_req_cpu / mec.cpu_capacity * 100) <= self.tau * 100 and \
                (mec.memory_utilization + self.app_req_memory / mec.memory_capacity * 100) <= self.tau * 100:
            # print("resources OK. ")
            return True
        else:
            # print("resources NOT OK. ")
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
