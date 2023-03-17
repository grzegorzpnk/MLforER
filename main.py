import random
import requests
import numpy as np
import gym
from gym.spaces import Discrete

from mec_node import MecNode
from app import MecApp



class MECEnv(gym.Env):

    def __init__(self, mecApp):



        # Initialize the MEC nodes part of a state
        self.mec_nodes = self._fetchMECnodesConfig()

        # ran specific
        self.number_of_RANs = len(self.mec_nodes[0].latency_array)

        #todo: set of predefined apps needs to be defined here
        self.mecApp = MecApp(mecApp.app_req_cpu, mecApp.app_req_memory, mecApp.app_req_latency, 0.8, self.number_of_RANs)
        self.mecApp.current_MEC = self._selectStartingNode()
        if self.mecApp.current_MEC is None:
            print("Cannot find any initial cluster for app")

        # Define the action and observation space

        self.action_space = gym.spaces.Discrete(len(self.mec_nodes))

        low_bound = np.zeros((len(self.mec_nodes), 5))  # initialize a 3x5 array filled with zeros
        low_bound[:, 0] = 1  # Low bound of CPU Capacity is 1 ( == 4000 mvCPU)
        low_bound[:, 1] = 0  # Low bound of CPU Utilization is 0%
        low_bound[:, 2] = 1  # Low bound of Memory Capacity is 1 ( == 4000 mvCPU)
        low_bound[:, 3] = 0  # Low bound of Memory Utilization is 0%
        low_bound[:, 4] = 1  # Low bound of unit cost is 1 ( == 0.33 == city-level)

        high_bound = np.ones((len(self.mec_nodes), 5))  # initialize a 3x5 array filled with zeros
        high_bound[:, 0] = 3  # High bound of CPU Capacity is 3 ( == 12000 mvCPU)
        high_bound[:, 1] = 100  # High bound of CPU Utilization is 100 [%]
        high_bound[:, 2] = 3  # High bound of Memory Capacity is 3 ( == 12000 mvCPU)
        high_bound[:, 3] = 100  # High bound of Memory Utilization is 100 [%]
        high_bound[:, 4] = 3  # High bound of unit cost is 3 ( == 1 == international-level)

        # MEC  : 1) CPU Capacity 2) CPU Utilization [%] 3) Memory Capacity 4) Memory Utilization [%] 5) Unit Cost
        space_MEC = gym.spaces.Box(shape=low_bound.shape, dtype=np.int32, low=low_bound, high=high_bound)
        # APP  : 1) Required mvCPU 2) required Memory 3) Required Latency 4) Current MEC 5) Current RAN
        space_APP = gym.spaces.Tuple((gym.spaces.Discrete(3, start=2),
                                      gym.spaces.Discrete(3, start=1),
                                      gym.spaces.Discrete(3, start=1),
                                      gym.spaces.Discrete(len(self.mec_nodes), start=1),
                                      gym.spaces.Discrete(self.number_of_RANs, start=1),))

        obs_box = gym.spaces.Tuple((space_MEC, space_APP))

        self.state = None

    def _get_state(self):
        space_MEC = np.zeros((len(self.mec_nodes), 5))

        # MEC  : 0) CPU Capacity 1) CPU Utilization [%] 2) Memory Capacity 3) Memory Utilization [%] 4) Unit Cost
        for i, mec_node in enumerate(self.mec_nodes):
            space_MEC[i, 0] = self.determineStateOfCPUCapacity(mec_node.cpu_capacity)
            space_MEC[i, 1] = mec_node.cpu_utilization
            space_MEC[i, 2] = self.determineStateOfMemoryCapacity(mec_node.memory_capacity)
            space_MEC[i, 3] = mec_node.memory_utilization
            space_MEC[i, 4] = self.determineStateofCost(mec_node.placement_cost)

        # APP  : 1) Required mvCPU 2) required Memory 3) Required Latency 4) Current MEC 5) Current RAN
        space_APP = (self.mecApp.app_req_cpu, self.mecApp.app_req_memory, self.mecApp.app_req_latency, self.mecApp.current_MEC, self.mecApp.user_position)

        state = gym.spaces.Tuple((space_MEC, space_APP))
        return state


    def determineStateOfCPUCapacity(cpu_capacity):
        if cpu_capacity == 4000:
            return 1
        if cpu_capacity == 8000:
            return 2
        if cpu_capacity == 12000:
            return 3

    def determineStateOfMemoryCapacity(memory_capacity):
        if memory_capacity == 4000:
            return 1
        if memory_capacity == 8000:
            return 2
        if memory_capacity == 12000:
            return 3

    def determineStateofCost(placement_cost):
        if placement_cost == 0.33333:
            return 1
        if placement_cost == 0.6667:
            return 2
        if placement_cost == 1:
            return 3


    def _printAllMECs(self):
        print(self.mec_nodes)

    def _getMecNodeByID(self, id):
        for node in self.mec_nodes:
            if node.id == id:
                return node
        # if the mec node with the given ID is not found, return None
        return None

    def _fetchMECnodesConfig(self):
        mec_nodes = []
        url = "http://127.0.0.1:8282/v1/topology/ml/InitialConfig"
        response = requests.get(url)

        if response.status_code == 200:
            json_data = response.json()
            for item in json_data:
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
        else:
            print('Error:', response.status_code)
        print(mec_nodes)
        return mec_nodes



    def _generateInitialLoadForTopology(self):

        loads = [30, 40, 50, 60, 70, 80, 90]
        middle_of_range = random.choice(loads)
        low_boundries = middle_of_range - 10
        high_boundries = middle_of_range + 10

        for mec in self.mec_nodes:
            mec.cpu_utilization = random.randint(low_boundries, high_boundries)
            mec.cpu_available = mec.cpu_capacity - mec.cpu_capacity * mec.cpu_utilization / 100
            mec.memory_utilization = random.randint(low_boundries, high_boundries)
            mec.memory_available = mec.memory_capacity - mec.memory_capacity * mec.memory_utilization / 100



    def _selectStartingNode(self):
        cnt = 0
        while True:
            randomMecId = random.randint(1, len(self.mec_nodes))
            if self.mecApp.LatencyOK(self._getMecNodeByID(randomMecId)) and self.mecApp.ResourcesOK(self._getMecNodeByID(randomMecId)):
                return self._getMecNodeByID(randomMecId)
            if cnt > 1000:
                return None
            cnt += 1


    def reset(self, mecApp):

            self.step = 0
            self._generateInitialLoadForTopology()

                        # app specific
            #todo: if the paths will be defined, so the starting cell will be known as well
            self.mecApp = MecApp(mecApp.app_req_cpu, mecApp.app_req_memory, mecApp.app_req_latency, mecApp.tau, self.number_of_RANs)
            self.mecApp.current_MEC = self._selectStartingNode()
            if self.mecApp.current_MEC is None:
                print("Cannot find any initial cluster for app")

            return self._get_state()

    def step(self, action, paramWeights):
        '''
        # We are assuming that the constraints are already checked by agent, and actions are masked -> here we need to move application only and update the state
        :param action:  ID of mec done where the application is relocated
        :param paramWeights: weights of particulary parts of Reward function ( should be declared at agent or env side?)
        :return:
        '''


        # Check that the action is within the action space
        assert self.action_space.contains(action)

        check_if_relocated = self._relocateApplication(action)

        # # Update the state of the environment
        # state = self._get_state()

        # Calculate the reward based on the new state
        # reward = self._calculate_reward(state)
        reward = self.calculateReward(paramWeights, check_if_relocated)

        # Determine whether the episode is finished
        done = False
        self.step += self.step

        if self.step > self.maxNumberOfSteps:
            done = True

        state = self._get_state()

        # Return the new state, the reward, and whether the episode is finished
        return state, reward, done, {}


    def _relocateApplication(self, action):
        '''
        :we are assuming that relocation MUST be finished with success, since the constraints are checked by agents and only allowed actions( latency OK, enough resources) are taken
        :todo: check what is under "action", seems that actions are int within range [0, length(mec_nodes)], but should be : [1 , length(mec_nodes)], maybe some dictionary for action space?
        :param action: currently action means the id of cluster where to relocate
        :return: true if app has been moved, false if app stayed at the same cluster
        '''
        #check first if selected MEC is a current MEC
        currentNode = self._getMecNodeByID(self.mecApp.current_MEC.id)
        targetNode = self._getMecNodeByID(action.targetNode)

        if currentNode == targetNode:
            return False
            print("No relocation, since selected cluster is the same as a current")

        #OLD NODE
        #take care of CPU
        currentNode.cpu_available -= self.mecApp.app_req_cpu
        currentNode.cpu_utilization /= currentNode.cpu_capacity * 100

        # take care of Memory
        currentNode.memory_available -= self.mecApp.app_req_memory
        currentNode.memory_utilization /= currentNode.memory_capacity * 100

        #NEW NODE
        # take care of CPU
        targetNode.cpu_available -= self.mecApp.app_req_cpu
        targetNode.cpu_utilization /= targetNode.cpu_capacity * 100

        # take care of Memory
        targetNode.memory_available -= self.mecApp.app_req_memory
        targetNode.memory_utilization /= targetNode.memory_capacity * 100

        #Application udpate
        self.mecApp.current_MEC = targetNode
        self.mecApp.user_position = action.uePosition

        return True

    def calculateReward(self, paramWeights, check_if_relocated):
        '''
        func to calculate reward after each step
        :param paramWeights: this is struct provided by agent with weights in order to prioritize some of reward function parts
        :param check_if_relocated: this params refers to
        :return:
        '''

        ###################### MIN number of relocation #############################
        if check_if_relocated:
            min_Number_of_relocation_reward = 1
        else:
            min_Number_of_relocation_reward = 0

        ######################### LOAD BALANCING REWARD ###############################
        # Initialize an empty arrays
        np.cpu_util = []
        np.mem_util = []

        # Create a numpy array of numbers
        for mec_node in self.mec_nodes:
            np.cpu_util.append(mec_node.cpu_utilization / 100) #divided by 100, as utilization is as percentage, so we are adding in a range [0-1] e.g array = [0.1, 0.4, 0.7, 0.5, ...]
            np.mem_util.append(mec_node.memory_utilization / 100)  # divided by 100, as utilization is as percentage, so we are adding in a range [0-1]

        # Calculate the standard deviation -> for range [0-1] the max std is 0.5 (e.g., mec1 - 0, mec2 - 1 -> mean: 0.5, deviation max: 0.5), the min is 0
        std_dev_cpu = np.std(np.cpu_util)
        std_dev_mem = np.std(np.mem_util)
        LB_reward = 1 - (std_dev_cpu + std_dev_mem) # if the std is huge, e.g. 0.4 (LB is bad), the reward would be: 1 - (0.4+0.4) = 0.2

        ###################### Cost of placement REWARD #############################
        placement_cost_reward = self.mecApp.current_MEC.placement_cost

        ###################### TOTAL REWARD #####################################
        reward = paramWeights.minNumberOfRelocation *  min_Number_of_relocation_reward +  paramWeights.minNumberOfRelocation * LB_reward + paramWeights.minNumberOfRelocation * placement_cost_reward
        return reward


    def test(self):
        Trajectory1 = [5, 7, 15, 25, 27, 29, 18, 10, 8, 1, 3, 12, 22, 25, 15, 22, 24, 33]
        Trajectory2 = [8, 1, 3, 12, 22, 25, 27, 29, 18, 13, 26, 32, 35, 37, 39, 31, 29, 25, 15, 22, 24, 33]
        Trajectory3 = [4, 14, 16, 23, 26, 32, 35, 42, 41, 28, 19, 17, 20, 6, 1]
        Trajectory4 = [2, 14, 16, 23, 32, 35, 42, 41, 28, 19, 21, 20, 17, 9, 4, 7, 12, 22, 24, 33, 39]
        Trajectory5 = [6, 1, 3, 12, 22, 25, 27, 31, 29, 25, 18, 10, 8, 1, 3, 12, 22, 25, 15, 22, 24, 33]
        Trajectory6 = [2, 14, 16, 23, 26, 32, 35, 37, 42, 41, 28, 19, 17, 9, 4, 7, 12, 22, 25, 27, 31, 29, 25, 18, 13]
        Trajectory7 = [9, 4, 14, 16, 23, 32, 35, 37, 42, 41, 28, 19, 21, 17, 20, 6, 2, 11, 13, 16, 23, 32, 34, 36, 31]
        Trajectory8 = [5, 7, 15, 22, 25, 27, 31, 29, 25, 27, 12, 22, 25, 27, 31, 29, 18, 13, 26, 23, 16, 23, 32, 34, 36, 31, 29]
        Trajectory9 = [4, 7, 12, 22, 25, 15, 22, 25, 27, 29, 18, 21, 20, 6, 9, 17, 20, 21, 19, 28, 30, 19, 21, 17]
        Trajectory10 = [13, 23, 26, 32, 35, 42, 41, 28, 30, 19, 21, 20, 17, 9, 4, 7, 12, 22, 25, 27, 29, 25, 18, 13, 26, 23, 16, 14, 2, 11]

        trajectories = [Trajectory1, Trajectory2, Trajectory3, Trajectory4, Trajectory5, Trajectory6, Trajectory7, Trajectory8, Trajectory9, Trajectory10]

        #      MecApp(mecApp.app_req_cpu, mecApp.app_req_memory, mecApp.app_req_latency, 0.8, self.number_of_RANs)
        app1 = MecApp(500, 500, 15, 0.8, self.number_of_RANs)


    # Episodes:
    # For each initial load
    # FOR each trajectory
    # For each type of application [ fixed latency, fixed resources ]

#         for initial_load in range (1,10):
#             self._generateInitialLoadForTopology()
#             for trajectory in trajectories:
#                 for app in self.apps:
#
#
#
# test()
#
