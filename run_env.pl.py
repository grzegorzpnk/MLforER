from gym.envs.registration import register
import gym

register(
    id='gym_examples/edge-relocation-v0',
    entry_point='gym_examples.envs:EdgeRelEnv',
    max_episode_steps=100,
)

env = gym.make('gym_examples/edge-relocation-v0', endpoint='127.0.0.1')
