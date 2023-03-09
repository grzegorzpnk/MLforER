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
        self.number_of_RANs = self._checkRANsNumber()

        #app specific
        self.mecApp = MecApp(mecApp.app_req_cpu, mecApp.app_req_memory, mecApp.app_req_latency, 0.8, self.number_of_RANs)
        self.mecApp.current_MEC = self._selectStartingNode()
        if self.mecApp.current_MEC is None:
            print("Cannot find any initial cluster for app")

        # Define the action and observation space
        #self.action_space = gym.spaces.Discrete(self.mec_nodes_number)
        self.action_space = gym.spaces.Tuple((Discrete(self.mec_nodes_number), Discrete(self.number_of_RANs)))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.mec_nodes_number * 2, self.number_of_RANs),
            dtype=np.float32,
        )

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
                    item['latency_matrix'],
                    item['placement_cost']
                )
                mec_nodes.append(mec_node)
        else:
            print('Error:', response.status_code)
        print(mec_nodes)
        return mec_nodes

    def _selectStartingNode(self):
        cnt = 0
        while True:
            randomMecId = random.randint(1, len(self.mec_nodes))
            if self.mecApp.LatencyOK(self._getMecNodeByID(randomMecId)) and self.mecApp.ResourcesOK(self._getMecNodeByID(randomMecId)):
                return self._getMecNodeByID(randomMecId)
            if cnt > 1000:
                return None
            cnt += 1

    def _checkRANsNumber(self):
        url = "http://127.0.0.1:8282/v1/topology/ml/rans"
        response = requests.get(url)
        if response.status_code == 200:
            response_data = response.json()
            return response_data
        else:
            print('Error:', response.status_code)

    def reset(self, mecApp, initialLoad):

            # Reset the MEC nodes part of a state
            self.mec_nodes = self._initializeMECnodes(random.randint(1, initialLoad))
            self.mec_nodes_number = len(self.mec_nodes)

            # reset number of rans -> it will remains the same over whole training

            # app specific
            self.mecApp = MecApp(mecApp.app_req_cpu, mecApp.app_req_memory, mecApp.app_req_latency, self.number_of_RANs)
            self.mecApp.current_MEC = self._selectStartingNode()
            if self.mecApp.current_MEC is None:
                print("Cannot find any initial cluster for app")

    def step(self, action, paramWeights):
        # We are assuming that the constraints are already checked by agent, and actions are masked -> here we need to move application only and update the state
        # I'm assuming that action is a ID of mec done where the application is relocated

        # Check that the action is within the action space
        assert self.action_space.contains(action)

        self._relocateApplication(action)

        # # Update the state of the environment
        # state = self._get_state()

        # Calculate the reward based on the new state
        # reward = self._calculate_reward(state)
        reward = self.calculateReward(paramWeights)

        # Determine whether the episode is finished
        done = False
        self.step += self.step
        if self.step > self.maxNumberOfSteps:
            done = True

        # Return the new state, the reward, and whether the episode is finished
        return state, reward, done, {}

    def _relocateApplication(self, action):
        #OLD NODE
        currentNode = self._getMecNodeByID(self.mecApp.current_MEC.id)
        #take care of CPU
        currentNode.cpu_available -= self.mecApp.app_req_cpu
        currentNode.cpu_utilization /= currentNode.cpu_capacity * 100

        # take care of Memory
        currentNode.memory_available -= self.mecApp.app_req_memory
        currentNode.memory_capacity /= currentNode.memory_capacity * 100

        #NEW NODE
        targetNode = self._getMecNodeByID(action.targetNode)
        # take care of CPU
        targetNode.cpu_available -= self.mecApp.app_req_cpu
        targetNode.cpu_utilization /= targetNode.cpu_capacity * 100

        # take care of Memory
        targetNode.memory_available -= self.mecApp.app_req_memory
        targetNode.memory_capacity /= targetNode.memory_capacity * 100

        #Application udpate
        self.mecApp.current_MEC = targetNode
        self.mecApp.user_position = action.uePosition

    def calculateReward(self, paramWeights):
        mean utlization =
        #reward = paramWeights.LatencyWeight * nLat + paramWeights.ResourcesWeight * (paramWeights.CpuUtilizationWeight * nCpu + paramWeights.MemUtilizationWeight * nMem) * staticCost
        return reward
