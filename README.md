# MLforER
This repo contains Rainforcement Learning training model for Edge Relocation

#Observation Space

The observation space of this custom environment in OpenAI Gym is defined by the obs_box object, which is a tuple of two sub-spaces:
MEC Sub-Space

The first sub-space represents the current state of all MEC nodes in the network. It is a Box space with a shape of (number_of_MEC_nodes, 5) and a data type of np.int32. The five dimensions of this sub-space correspond to the following attributes of each MEC node:

    CPU Capacity: The maximum CPU capacity of the MEC node in milli-vCPU. The valid range for this attribute is between 1 and 3, corresponding to 4000 and 12000 milli-vCPU, respectively.
    CPU Utilization: The current CPU utilization of the MEC node as a percentage of its maximum capacity. The valid range for this attribute is between 0 and 100.
    Memory Capacity: The maximum memory capacity of the MEC node in milli-bytes. The valid range for this attribute is between 1 and 3, corresponding to 4000 and 12000 milli-bytes, respectively.
    Memory Utilization: The current memory utilization of the MEC node as a percentage of its maximum capacity. The valid range for this attribute is between 0 and 100.
    Unit Cost: The placement cost of the MEC node, expressed as a unitless value that ranges from 1 to 3. A value of 1 corresponds to a city-level cost, while a value of 3 corresponds to an international-level cost.

APP Sub-Space

The second sub-space represents the current state of the mobile application requesting resources from the MEC network. It is a Tuple space consisting of the following five discrete sub-spaces:

    Required mvCPU: The required milli-vCPU capacity of the mobile application. This is a Discrete space with a valid range of 2 to 4, corresponding to a required capacity of 8000 to 16000 milli-vCPU, respectively.
    Required Memory: The required memory capacity of the mobile application in milli-bytes. This is a Discrete space with a valid range of 1 to 3, corresponding to a required capacity of 4000 to 12000 milli-bytes, respectively.
    Required Latency: The required latency of the mobile application in milliseconds. This is a Discrete space with a valid range of 1 to 3, corresponding to required latencies of 100, 200, and 300 milliseconds, respectively.
    Current MEC: The index of the current MEC node that the mobile application is connected to. This is a Discrete space with a valid range of 1 to the number of MEC nodes in the network.
    Current RAN: The index of the current RAN (Radio Access Network) that the mobile application is connected to. This is a Discrete space with a valid range of 1 to the total number of RANs in the network.

The observation space of this custom environment provides an agent with all the necessary information about the current state of the MEC network and the mobile application to make informed decisions about which MEC node to connect to and allocate resources from
