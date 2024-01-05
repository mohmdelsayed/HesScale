import gym
from core.task.environment import Environment

class MujocoEnv(Environment):
    """
    MujocoEnv environment.
    https://gymnasium.farama.org/environments/mujoco/
    """

    def __init__(self, name='InvertedPendulum', seed=0):
        self.name = name
        self.env = gym.make(f'{name}-v2')
        self.env.seed(seed)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.action_space_type = 'continuous'
        super().__init__()

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)        

    def get_random_action(self):
        return self.env.action_space.sample()

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
