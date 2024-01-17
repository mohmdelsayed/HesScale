import numpy as np
import gym

class AddTimeInfo(gym.core.Wrapper):
    '''
    This wrapper will add time information to the observation. The time information is a float between 0 and 1 that
    indicates the progress of the episode. 0 means the episode just started and 1 means the episode is over.
    '''
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.epi_time = -0.5
        self.time_limit = self.env._max_episode_steps
        self.obs_space_size = self.observation_space.shape[0] + 1
        self.action_space_size = self.action_space.shape[0]
        self._max_episode_steps = self.env._max_episode_steps
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.observation_space.shape[0] + 1,), dtype=np.float64)

    def step(self, action):
        obs, rews, terminateds, infos = self.env.step(action)
        obs = np.concatenate((obs, np.array([self.epi_time])))
        self.epi_time += 1.0 / self.time_limit
        return obs, rews, terminateds, infos
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.epi_time = -0.5
        obs = np.concatenate((obs, np.array([self.epi_time])))
        return obs
