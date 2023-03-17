# MLforER
This repo contains Rainforcement Learning training model for Edge Relocation
Details about what Edge Relocation is can be found [here](https://ieeexplore.ieee.org/document/9779643) 


## Action Space Documentation

The action space for the custom environment in AI gym is defined as follows:

python

`self.action_space = gym.spaces.Discrete(len(self.mec_nodes))` 

The action space consists of a discrete set of values, where each value corresponds to the index of MEC node that the agent will choose to relocate application to. The number of possible actions is equal to the number of MEC nodes available in the environment.


## Observation Space

The observation space of this custom environment in OpenAI Gym is defined by the `obs_box` object, which is a tuple of two sub-spaces:

       
       ` # APP  : 1) Required mvCPU 2) required Memory 3) Required Latency 4) Current MEC 5) Current RAN
        space_APP = gym.spaces.Tuple((gym.spaces.Discrete(3, start=1),
                                      gym.spaces.Discrete(3, start=1),
                                      gym.spaces.Discrete(3, start=1),
                                      gym.spaces.Discrete(len(self.mec_nodes), start=1),
                                      gym.spaces.Discrete(self.number_of_RANs, start=1),))`

        obs_box = gym.spaces.Tuple((space_MEC, space_APP))


### MEC infra Sub-Space

` low_bound = np.zeros((len(self.mec_nodes), 5))  # initialize a 3x5 array filled with zeros
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
        space_MEC = gym.spaces.Box(shape=low_bound.shape, dtype=np.int32, low=low_bound, high=high_bound)`
        
The first sub-space represents the current state of all MEC nodes in the network. It is a `Box` space, a 2-dimensional array with a shape of `(number_of_MEC_nodes, 5)` and a data type of `np.int32`. The five dimensions of this sub-space correspond to the following attributes of each MEC node:

1.  **CPU Capacity**: The CPU capacity of the MEC node in milli-vCPU. The valid value for this attribute is 1, 2 or 3, corresponding to 4000, 8000, 12000 milli-vCPU, respectively.
2.  **CPU Utilization**: The current CPU utilization of the MEC node as a percentage of used/capacity. The valid range for this attribute is between 0 and 100, what express percentage utilization.
3.  **Memory Capacity**: The RAM memory capacity of the MEC node in Megabytes. The valid value for this attribute is 1, 2 or 3, corresponding to 4000, 8000, 12000, Megabytes, respectively.
4.  **Memory Utilization**: The current memory utilization of the MEC node as a percentage of used/capacity. The valid range for this attribute is between 0 and 100, what express percentage utilization.
5.  **Unit Cost**: The placement cost of the MEC node, expressed as a unitless value that ranges from 1 to 3. A value of 1 corresponds to a city-level MEC, 2 to a regional MEC and 3 to international level cost.

### MEC App Sub-Space

The second sub-space represents the current state of the MEC application requesting deployed at MEC infra. It is a `Tuple` space consisting of the following five discrete sub-spaces:

1.  **Required mvCPU**: The required by app a milli-vCPU at MEC Node. This is a `Discrete` space with a valid values of 1, 2, 3, corresponding to a required mvCPU resources: 300, 400, 500 milli-vCPU, respectively.
2.  **Required Memory**: The required by app a Megabytes RAM memory at MEC Node. This is a `Discrete` space with a valid values of 1, 2, 3, corresponding to a required  resources: 300, 400, 500 Megabytes RAM, respectively.
3.  **Required Latency**: The required latency of the MEC application in milliseconds to be guaranteed by selected MEC Node. This is a `Discrete` space with a valid value 1,2,3 , corresponding to required latencies of 10, 15, and 25 milliseconds, respectively.
4.  **Current MEC**: The index of the current MEC node where the MEC application is deployed. This is a `Discrete` integer space with a valid range of 1 to the number of MEC nodes in the topology.
5.  **Current RAN**: The index of the current cell - RAN (Radio Access Network) that the user of considered MEC application is located. This is a `Discrete` integer space with a valid range of 1 to the total number of RANs in the network.

The observation space of this custom environment provides an agent with all the necessary information about the current state of the MEC network and the mobile application to make informed decisions about which MEC node to connect to and allocate resources from
