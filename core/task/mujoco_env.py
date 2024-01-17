import numpy as np
import gym
from core.task.environment import Environment
from core.task.add_time_info import AddTimeInfo
class MujocoEnv(Environment):
    """
    MujocoEnv environment.
    https://gymnasium.farama.org/environments/mujoco/
    """

    def __init__(self, name='InvertedPendulum', seed=0):
        self.name = name
        self.env = gym.make(f'{name}-v2')
        self.env = AddTimeInfo(self.env)
        self.env.seed(seed)
        self.n_states = self.env.obs_space_size
        self.n_actions = self.env.action_space_size
        self.action_space_type = 'continuous'
        super().__init__()

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)        

    def get_random_action(self):
        return self.env.action_space.sample()

    def normalize(self):
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = gym.wrappers.NormalizeObservation(self.env)
        self.env = gym.wrappers.TransformObservation(self.env, lambda obs: np.clip(obs, -10, 10))
        self.env = gym.wrappers.NormalizeReward(self.env, gamma=0.99)
        self.env = gym.wrappers.TransformReward(self.env, lambda reward: np.clip(reward, -10, 10))

    def __str__(self) -> str:
        return self.name

    def get_max_episode_steps(self):
        return self.env._max_episode_steps

if __name__ == "__main__":
    env = MujocoEnv()
    env.reset()
    for i in range(100):
        # get random action
        action = env.get_random_action()
        # perform action
        state, reward, done, _, _ = env.step(action)
        # reset if done
        if done:
            env.reset()
