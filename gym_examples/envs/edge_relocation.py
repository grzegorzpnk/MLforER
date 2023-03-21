import random
from typing import Optional, Union, List

import requests
import numpy as np
import gym
from gym.core import RenderFrame
import json


class EdgeRelEnv(gym.Env):

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def __init__(self, configPath):

        # read initial topology config from external simulator
        # self.mec_nodes = self._fetchMECnodesConfig(endpoint)
        self.mec_nodes = self._readMECnodesConfig(configPath)
        self.number_of_RANs = len(self.mec_nodes[0].latency_array)
        self.step = 0
        self.trajectory = None
        self.mecApp = None

        # generate inputs: iniital load, application and starting position (MEC and cell), trajectory

        # generate initial load
        self._generateInitialLoadForTopology()

        # generate trajectory
        self.mobilityStateMachine = self._generateStateMachine()
        self.trajectory = self._generateTrajectory(5, 25)

        # generateApp
        self.mecApp = self._generateMECApp()
        self.mecApp.current_MEC = self._selectStartingNode()
        while self.mecApp.current_MEC is None:
            print("Cannot find any initial cluster for app. Generating new one")
            self.mecApp = self._generateMECApp()
            self.mecApp.current_MEC = self._selectStartingNode()

        # Define the action and observation space
        self.action_space = gym.spaces.Discrete(len(self.mec_nodes), start=1)

        low_bound = np.zeros((len(self.mec_nodes), 5))  # initialize a 3x5 array filled with zeros
        low_bound[:, 0] = 1  # Low bound of CPU Capacity is 1 ( == 4000 mvCPU)
        low_bound[:, 1] = 0  # Low bound of CPU Utilization is 0%
        low_bound[:, 2] = 1  # Low bound of Memory Capacity is 1 ( == 4000 Mb RAM)
        low_bound[:, 3] = 0  # Low bound of Memory Utilization is 0%
        low_bound[:, 4] = 1  # Low bound of unit cost is 1 ( == 0.33 == city-level)

        high_bound = np.ones((len(self.mec_nodes), 5))  # initialize a 3x5 array filled with zeros
        high_bound[:, 0] = 3  # High bound of CPU Capacity is 3 ( == 12000 mvCPU)
        high_bound[:, 1] = 100  # High bound of CPU Utilization is 100 [%]
        high_bound[:, 2] = 3  # High bound of Memory Capacity is 3 ( == 12000 Mb RAM)
        high_bound[:, 3] = 100  # High bound of Memory Utilization is 100 [%]
        high_bound[:, 4] = 3  # High bound of unit cost is 3 ( == 1 == international-level)

        # MEC  : 1) CPU Capacity 2) CPU Utilization [%] 3) Memory Capacity 4) Memory Utilization [%] 5) Unit Cost
        space_MEC = gym.spaces.Box(shape=low_bound.shape, dtype=np.int32, low=low_bound, high=high_bound)
        # APP  : 1) Required mvCPU 2) required Memory 3) Required Latency 4) Current MEC 5) Current RAN
        space_APP = gym.spaces.Tuple((gym.spaces.Discrete(6, start=1),
                                      gym.spaces.Discrete(6, start=1),
                                      gym.spaces.Discrete(3, start=1),
                                      gym.spaces.Discrete(len(self.mec_nodes), start=1),
                                      gym.spaces.Discrete(self.number_of_RANs, start=1),))

        self.observation_space = gym.spaces.Tuple((space_MEC, space_APP))
        self.state = self._get_state()

    def _get_state(self):
        space_MEC = np.zeros((len(self.mec_nodes), 5))

        # MEC  : 0) CPU Capacity 1) CPU Utilization [%] 2) Memory Capacity 3) Memory Utilization [%] 4) Unit Cost
        for i, mec_node in enumerate(self.mec_nodes):
            space_MEC[i, 0] = self.determineStateOfCapacity(mec_node.cpu_capacity)
            space_MEC[i, 1] = mec_node.cpu_utilization
            space_MEC[i, 2] = self.determineStateOfCapacity(mec_node.memory_capacity)
            space_MEC[i, 3] = mec_node.memory_utilization
            space_MEC[i, 4] = self.determineStateofCost(mec_node.placement_cost)

        # APP  : 1) Required mvCPU 2) required Memory 3) Required Latency 4) Current MEC 5) Current RAN
        space_APP = gym.spaces.Tuple((self.determineStateofAppReq(self.mecApp.app_req_cpu),
                                      self.determineStateofAppReq(self.mecApp.app_req_memory),
                                      self.determineStateofAppLatReq(self.mecApp.app_req_latency),
                                      self.mecApp.current_MEC.id,
                                      self.mecApp.user_position))

        state = gym.spaces.Tuple((space_MEC, space_APP))
        return state

    def determineStateOfCapacity(self, capacityValue):
        if capacityValue == 4000:
            return 1
        if capacityValue == 8000:
            return 2
        if capacityValue == 12000:
            return 3

    def determineStateofCost(self, placement_cost):
        if placement_cost == 0.33333:
            return 1
        if placement_cost == 0.6667:
            return 2
        if placement_cost == 1:
            return 3

    def determineStateofAppReq(self, reqValue):
        if reqValue == 500:
            return 1
        if reqValue == 600:
            return 2
        if reqValue == 700:
            return 3
        if reqValue == 800:
            return 4
        if reqValue == 900:
            return 5
        if reqValue == 1000:
            return 6

    def determineStateofAppLatReq(self, latValue):
        if latValue == 10:
            return 1
        if latValue == 15:
            return 2
        if latValue == 25:
            return 3

    def _generateTrajectory(self, min_length, max_length):

        # generate initial UE position
        start_state = random.randint(1, self.number_of_RANs)
        trajectory = [start_state]
        current_state = start_state

        # generate length of trajectory
        trajectory_length = random.randint(min_length, max_length)
        for i in range(trajectory_length):
            next_states = self.mobilityStateMachine.get(str(current_state), [])
            if not next_states:
                break
            current_state = random.choice(next_states)
            trajectory.append(current_state)

        return trajectory

    def _generateMECApp(self):

        # Generate a value for required resources among given:
        resources_req = [500, 600, 700, 800, 900, 1000]
        random_req_cpu = random.choice(resources_req)
        random_req_mem = random.choice(resources_req)
        # Define a list of three latency: 10, 15, 25
        allowed_latencies = [10, 15, 25]
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

        print(mec_nodes)
        return mec_nodes

    def _generateInitialLoadForTopology(self):

        loads = [30, 40, 50, 60, 70, 80, 90]
        middle_of_range = random.choice(loads) # 70%
        low_boundries = middle_of_range - 10 #60
        high_boundries = middle_of_range + 10 #80

        for mec in self.mec_nodes:
            mec.cpu_utilization = random.randint(low_boundries, high_boundries)
            mec.cpu_available = int(mec.cpu_capacity - mec.cpu_capacity * mec.cpu_utilization / 100)
            mec.memory_utilization = random.randint(low_boundries, high_boundries)
            mec.memory_available = int(mec.memory_capacity - mec.memory_capacity * mec.memory_utilization / 100)

    def _selectStartingNode(self):
        cnt = 0
        while True:
            randomMecId = random.randint(1, len(self.mec_nodes))
            randomMecName = "mec" + str(randomMecId)
            if self.mecApp.LatencyOK(self._getMecNodeByID(randomMecName)) and self.mecApp.ResourcesOK(self._getMecNodeByID(randomMecName)):
                self._getMecNodeByID(randomMecName).cpu_utilization += self.mecApp.app_req_cpu/self._getMecNodeByID(randomMecName).cpu_capacity
                self._getMecNodeByID(randomMecName).cpu_available = self._getMecNodeByID(randomMecName).cpu_capacity - self._getMecNodeByID(randomMecName).cpu_capacity * self._getMecNodeByID(randomMecName).cpu_utilization

                self._getMecNodeByID(randomMecName).memory_utilization += self.mecApp.app_req_memory / self._getMecNodeByID(randomMecName).memory_capacity
                self._getMecNodeByID(randomMecName).memory_available = self._getMecNodeByID(randomMecName).memory_capacity - self._getMecNodeByID(randomMecName).memory_capacity * self._getMecNodeByID(randomMecName).memory_utilization

                return self._getMecNodeByID(randomMecName)
            if cnt > 1000:
                return None
            cnt += 1

    def reset(self):

        # todo: check if seed change is needed here
        super().reset()

        self.step = 0

        # generate inputs: inital load, application, trajectory

        # generate initial load
        self._generateInitialLoadForTopology()

        # generate trajectory
        self.trajectory = self._generateTrajectory(5, 25)

        # generateApp
        self.mecApp = self._generateMECApp()
        self.mecApp.current_MEC = self._selectStartingNode()
        while self.mecApp.current_MEC is None:
            print("Cannot find any initial cluster for app. Generating new one")
            self.mecApp = self._generateMECApp()
            self.mecApp.current_MEC = self._selectStartingNode()

        self.state = self._get_state()

        return self.state

    def step(self, action):
        '''
        # We are assuming that the constraints are already checked by agent, and actions are masked -> here we need to move application only and update the state
        :param action:  ID of mec done where the application is relocated
        :param paramWeights: weights of particulary parts of Reward function ( should be declared at agent or env side?)
        :return:
        '''

        # Check that the action is within the action space
        assert self.action_space.contains(action)

        self.step += 1

        relocation_done = self._relocateApplication(action)

        # Update the state of the environment
        self.state = self._get_state()

        # Calculate the reward based on the new state
        # reward = self._calculate_reward(state)
        reward = self.calculateReward(relocation_done)

        # Determine whether the episode is finished
        done = False
        if self.step == len(self.trajectory):
            done = True

        # Return the new state, the reward, and whether the episode is finished
        return self.state, reward, done, {}

    def _relocateApplication(self, action):
        '''
        :we are assuming that relocation MUST be finished with success, since the constraints are checked by agents and only allowed actions( latency OK, enough resources) are taken
        :todo: check what is under "action", seems that actions are int within range [0, length(mec_nodes)], but should be : [1 , length(mec_nodes)], maybe some dictionary for action space?
        :param action: currently action means the id of cluster where to relocate
        :return: true if app has been moved, false if app stayed at the same cluster
        '''
        # check first if selected MEC is a current MEC
        currentNode = self._getMecNodeByID(self.mecApp.current_MEC.id)
        targetNode = self._getMecNodeByID(action.targetNode)

        if currentNode == targetNode:
            print("No relocation, since selected cluster is the same as a current")
            return False

        # OLD NODE
        # take care of CPU
        currentNode.cpu_available -= self.mecApp.app_req_cpu
        currentNode.cpu_utilization /= currentNode.cpu_capacity * 100

        # take care of Memory
        currentNode.memory_available -= self.mecApp.app_req_memory
        currentNode.memory_utilization /= currentNode.memory_capacity * 100

        # NEW NODE
        # take care of CPU
        targetNode.cpu_available += self.mecApp.app_req_cpu
        targetNode.cpu_utilization /= targetNode.cpu_capacity * 100

        # take care of Memory
        targetNode.memory_available += self.mecApp.app_req_memory
        targetNode.memory_utilization /= targetNode.memory_capacity * 100

        # Application udpate
        self.mecApp.current_MEC = targetNode
        self.mecApp.user_position = self.trajectory[self.step]

        return True

    def calculateReward(self, is_relocation_done):
        '''
        func to calculate reward after each step
        :param is_relocation_done: this params refers to
        :return:
        '''

        if not is_relocation_done:
            reward = 1
        else:
            reward = 0
        #########################

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


class MecNode:
    # utilization -> percentage
    def __init__(self, id, cpu_capacity, memory_capacity, cpu_utilization, memory_utilization, latency_array,
                 placement_cost):
        self.id = id
        self.cpu_capacity = cpu_capacity
        self.cpu_utilization = cpu_utilization
        self.cpu_available = cpu_capacity - cpu_utilization / 100 * cpu_capacity
        self.memory_capacity = memory_capacity
        self.memory_utilization = memory_utilization
        self.memory_available = memory_capacity - memory_utilization / 100 * memory_capacity
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
        '''
        This is supportive funtion to check latency conditions, used only for initial (for init state) placement of our main app.
        This func is not used to check conditions during the relocation, since it;s responisibility of agent
        :param mec:
        :return:
        '''
        # -1 cause latency_array[0] refers to the cell 1 etc..
        if mec.latency_array[self.user_position-1] < self.app_req_latency:
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


env = EdgeRelEnv("topoconfig.json")
print(env._printAllMECs())
