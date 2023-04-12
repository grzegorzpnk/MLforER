from gym.envs.registration import register

register(
    id='gym_examples/edge-relocation-v0',
    entry_point='gym_examples.envs:EdgeRelEnv',
    max_episode_steps=300,
)