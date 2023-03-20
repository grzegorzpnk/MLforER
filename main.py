import random
import requests
import numpy as np
import gym
from gym.spaces import Discrete

from mec_node import MecNode
from app import MecApp



class MECEnv(gym.Env):

    def __init__(self):

        self.mec_nodes = self._fetchMECnodesConfig()
        self.number_of_RANs = len(self.mec_nodes[0].latency_array)

        #generate inputs: iniital load, application and starting position (MEC and cell), trajectory

        #generate initial load
        self._generateInitialLoadForTopology()

        #generate trajectory
        self.mobilityStateMachine = self._generateStateMachine()
        self.trajectory = self._generateTrajectory(5, 25)

        #generateApp
        self.mecApp = self._generateMECApp()
        self.mecApp.current_MEC = self._selectStartingNode()
        if self.mecApp.current_MEC is None:
            print("Cannot find any initial cluster for app")
            #todo and what next?

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
        space_APP = gym.spaces.Tuple((gym.spaces.Box(3, shape=low_bound.shape, dtype=np.int32, low=low_bound, high=high_bound),
                                      gym.spaces.Discrete(3, start=1),
                                      gym.spaces.Discrete(3, start=1),
                                      gym.spaces.Discrete(len(self.mec_nodes), start=1),
                                      gym.spaces.Discrete(self.number_of_RANs, start=1),))

        obs_box = gym.spaces.Tuple((space_MEC, space_APP))

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
        space_APP = (self.determineStateofAppReq(self.mecApp.app_req_cpu),
                     self.determineStateofAppReq(self.mecApp.app_req_memory),
                     self.determineStateofAppLatReq(self.mecApp.app_req_latency),
                     self.mecApp.current_MEC.id,
                     self.mecApp.user_position)


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
        if reqValue == 300:
            return 1
        if reqValue == 400:
            return 2
        if reqValue == 500:
            return 3

    def determineStateofAppLatReq(self, latValue):
        if latValue == 10:
            return 1
        if latValue == 15:
            return 2
        if latValue == 25:
            return 3

    def _generateTrajectory(self, min_length, max_length):

            #generate initial UE position
            start_state = random.randint(1,self.number_of_RANs)
            trajectory = [start_state]
            current_state = start_state

            #generate length of trajectory
            trajectory_length = random.randint(min_length, max_length)

            for i in range(trajectory_length):
                next_states = self.mobilityStateMachine[current_state]
                if not next_states:
                    break
                current_state = random.choice(next_states)
                trajectory.append(current_state)

            return trajectory


    def _generateMECApp(self):

        # Generate a random integer between 500 and 1000 (inclusive)
        random_req_cpu = random.randint(500, 1000)
        random_req_mem = random.randint(500, 1000)
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

        if check_if_relocated:
            reward = 1
            return reward
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