import gym
import numpy as np


mec_node = [1,2,3,4,5,6,7]
rans = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

low_bound = np.zeros((len(mec_node), 5))  # initialize a 3x5 array filled with zeros
low_bound[:, 0] = 1  # set all rows in the first column to 1
low_bound[:, 2] = 1  # set all rows in the third column to 1
low_bound[:, 4] = 1  # set all rows in the fifth column to 0.5

high_bound = np.ones((len(mec_node), 5))  # initialize a 3x5 array filled with zeros
high_bound[:, 0] = 12000
high_bound[:, 1] = 100
high_bound[:, 2] = 12000
high_bound[:, 3] = 100
high_bound[:, 4] = 3

space_MEC = gym.spaces.Box(shape=low_bound.shape, dtype=np.int32, low=low_bound, high=high_bound)
space_APP = gym.spaces.Tuple((gym.spaces.Discrete(3, start=2),
                             gym.spaces.Discrete(3, start=1),
                             gym.spaces.Discrete(3, start=1),
                             gym.spaces.Discrete(len(mec_node), start=1),
                             gym.spaces.Discrete(len(rans), start=1),))

# Sample a value from the space
obs_box = gym.spaces.Tuple((space_MEC,space_APP))

sample = obs_box.sample()
print(sample)

# self.observation_space = gym.spaces.Dict(
#     {
#         # L'espace des observations :
#         # Infra : 1) CPU 2) ST 3) UP_CPU 4) UP_ST 5) BW  6) DEG 7) UP_ABW 8) Dikestra 9) Masque
#         # VNF   : 1) CPU 2) ST 3) Delay 4) Position 5) Users  6)  NB
#         "infra": gym.spaces.Box(shape=t_1.shape, dtype=np.float32, high=t_1, low=t_2),
#         "vnf": gym.spaces.Box(shape=vn_1.shape, dtype=np.float32, high=vn_1, low=vn_2),
#
#     }


def state():
    space_APP = (mec_app.CPUCapacity, mec_app.CPUUtilization, mec_app.memCapacity, mec_app.memCapacity, mec_app.Ran, mec_app.CurrentMEC)

    return gym.spaces.Tuple((space_MEC,space_APP))
