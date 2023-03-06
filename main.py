import random
import requests
import numpy as np
import gym
from mec_node import MecNode
from app import MecApp



class MECEnv(gym.Env):

    def __init__(self, mecApp, initialLoad, maxNumberOfSteps):

        self.maxNumberOfSteps = maxNumberOfSteps
        self.step = 0

        # Initialize the MEC nodes part of a state
        self.mec_nodes = self.initializeMECnodes(initialLoad)
        self.mec_nodes_number = len(self.mec_nodes)

        # ran specific
        self.number_of_RANs = self.checkRANsNumber()

        #app specific
        self.mecApp = MecApp(mecApp.app_req_cpu, mecApp.app_req_memory, mecApp.app_req_latency, self.number_of_RANs)
        self.mecApp.current_MEC = self.selectStartingNode(self, self.mecApp)
        if self.mecApp.current_MEC is None:
            print("Cannot find any initial cluster for app")

        # Define the action and observation space
        self.action_space = gym.spaces.Discrete(self.mec_nodes_number)
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

    def selectStartingNode(self):
        cnt = 0
        while True:
            randomMecId = random.randint(1, len(self.mec_nodes))
            randomMec = self.get_mec_node_by_id(self.mec_nodes, randomMecId)
            if self.mecApp.LatencyOK(randomMec) and self.mecApp.ResourcesOK(randomMec):
                return randomMec
            if cnt > 1000:
                return None
            cnt += 1

    def printallMECs(self):
        print(self.mec_nodes)
    def get_mec_node_by_id(mec_nodes, id):
        for node in mec_nodes:
            if node.id == id:
                return node
        # if the mec node with the given ID is not found, return None
        return None


    #number of application (as a initial load on a cluster needs to be transfered as a param, not hardcoded, but first we need to know how to pass this argument from agent
    def initializeMECnodes(self, initialLoad):
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

    def checkRANsNumber(self):
        url = "http://127.0.0.1:8282/v1/topology/ml/rans"
        response = requests.get(url)
        if response.status_code == 200:
            response_data = response.json()
            return response_data
        else:
            print('Error:', response.status_code)

    def reset(self, mecApp, initialLoad):

            # Reset the MEC nodes part of a state
            self.mec_nodes = self.initializeMECnodes(random.randint(1, initialLoad))
            self.mec_nodes_number = len(self.mec_nodes)

            # reset number of rans -> it will remains the same over whole training

            # app specific
            self.mecApp = MecApp(mecApp.app_req_cpu, mecApp.app_req_memory, mecApp.app_req_latency, self.number_of_RANs)
            self.mecApp.current_MEC = self.selectStartingNode()
            if self.mecApp.current_MEC is None:
                print("Cannot find any initial cluster for app")


    def step(self, action):
        # Take the specified action
        self._take_action(action)

        # Update the state of the environment
        state = self._get_state()

        # Calculate the reward based on the new state
        reward = self._calculate_reward(state)

        # Determine whether the episode is finished
        done = False
        self.step += self.step
        if self.step > self.maxNumberOfSteps:
            done = True

        # Return the new state, the reward, and whether the episode is finished
        return state, reward, done, {}









