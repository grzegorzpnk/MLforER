import random
import requests
import numpy as np
import gym
from mec_node import MecNode
from app import MecApp



class MECEnv(gym.Env):

    def __init__(self, mecApp):

        # Initialize the MEC nodes part of a state
        self.mec_nodes = self.initializeMECnodes()
        self.n_mec_nodes = len(self.mec_nodes)

        # ran specific
        self.n_rans = self.checkRANsNumber()

        #app specific
        self.mecApp = MecApp(mecApp.app_req_cpu,mecApp.app_req_memory,mecApp.app_req_latency, self.n_rans)
        self.mecApp.current_MEC = self.selectStartingNode(self, self.mecApp)
        if self.mecApp.current_MEC is None:
            print("Cannot find any initial cluster for app")

        # Define the action and observation space
        self.action_space = gym.spaces.Discrete(self.n_mec_nodes)  # Action space - possible action that agent can execute. it means that we can take n_mec_nodes number of actions, e.g. 1-> relocate to MEC 1, ; 2-> relocate to MEC 2
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.n_mec_nodes * 2, self.n_rans),
            dtype=np.float32,
        )

       # self.state = self._get_state()

    def selectStartingNode(self):
        cnt = 0
        while True:
            randomMec = self.get_mec_node_by_id(self.mec_nodes, random.randint(1, len(self.mec_nodes)))
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
    def initializeMECnodes(self):
        mec_nodes = []
        url = "http://127.0.0.1:8282/v1/topology/ml/InitialState/50"
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













