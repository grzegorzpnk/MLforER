import random
import requests
import numpy as np
import gym
from gym.spaces import Discrete

from mec_node import MecNode
from app import MecApp



class MECEnv(gym.Env):

    def __init__(self, mecApp, initialLoad, maxNumberOfSteps):

        self.maxNumberOfSteps = maxNumberOfSteps
        self.step = 0

        # Initialize the MEC nodes part of a state
        self.mec_nodes = self._initializeMECnodes(initialLoad)
        self.mec_nodes_number = len(self.mec_nodes)

        # ran specific
        self.number_of_RANs = len(self.mec_nodes[0].latency_array)

        #app specific
        self.mecApp = MecApp(mecApp.app_req_cpu, mecApp.app_req_memory, mecApp.app_req_latency, 0.8, self.number_of_RANs)
        self.mecApp.current_MEC = self._selectStartingNode()
        if self.mecApp.current_MEC is None:
            print("Cannot find any initial cluster for app")

        # Define the action and observation space
        #self.action_space = gym.spaces.Discrete(self.mec_nodes_number)
        # self.action_space = gym.spaces.Tuple((Discrete(self.mec_nodes_number), Discrete(self.number_of_RANs))) -> its commented since the next cell value will be fixed, at agent or at env side
        self.action_space = gym.spaces.Discrete(self.mec_nodes_number)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.mec_nodes_number * 2, self.number_of_RANs),
            dtype=np.float32,
        )

        #alternatively:

        # The observation space consists of the CPU and memory utilization and capacity of each MEC node, and the RAN where the application's user is currently located
        # Each MEC node is represented by a tuple containing its CPU utilization, CPU capacity, memory utilization, memory capacity, and the cost of placement
        # The RAN is represented by a categorical variable
        self.observation_space = spaces.Tuple((spaces.Tuple([spaces.Box(low=0, high=1, shape=(1,), dtype=np.int),
                                                             spaces.Box(low=0, high=1, shape=(1,), dtype=np.int),
                                                             ),
                                                             spaces.Discrete(num_rans)))



        self.state = self._get_state()
    #
    # def _get_state(self):
    #
    #     state.number_of_RANs = self.number_of_RANs
    #
    #     return state

    def _printAllMECs(self):
        print(self.mec_nodes)

    def _getMecNodeByID(self, id):
        for node in self.mec_nodes:
            if node.id == id:
                return node
        # if the mec node with the given ID is not found, return None
        return None

    def _initializeMECnodes(self, initialLoad):
        mec_nodes = []
        url = "http://127.0.0.1:8282/v1/topology/ml/InitialState/"+initialLoad
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

#Episodes:
#For each initial load
    #FOR each trajectory
            #For each type of application [ fixed latency, fixed resources ]



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

    # def _checkRANsNumber(self):
    #     url = "http://127.0.0.1:8282/v1/topology/ml/rans"
    #     response = requests.get(url)
    #     if response.status_code == 200:
    #         response_data = response.json()
    #         return response_data
    #     else:
    #         print('Error:', response.status_code)

    def reset(self, mecApp, initialLoad):

            # Reset the MEC nodes part of a state
            self.mec_nodes = self._initializeMECnodes(initialLoad)
            self.mec_nodes_number = len(self.mec_nodes)

            # app specific
            #todo: if the paths will be defined, so the starting cell will be known as well
            self.mecApp = MecApp(mecApp.app_req_cpu, mecApp.app_req_memory, mecApp.app_req_latency, mecApp.tau, self.number_of_RANs)
            self.mecApp.current_MEC = self._selectStartingNode()
            if self.mecApp.current_MEC is None:
                print("Cannot find any initial cluster for app")

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
