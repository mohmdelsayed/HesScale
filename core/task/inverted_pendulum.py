import gym
from core.task.environment import Environment

class InvertedPendulum(Environment):
    """
    InvertedPendulum environment.
    https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/
    """

    def __init__(self, name="inverted_pendulum"):
        self.env = gym.make('InvertedPendulum-v2')
        self.name = name
        # self.n_states = 4
        # self.n_actions = 2
        self.action_space_type = 'continuous'
        super().__init__()

    def reset(self, seed=None):
        return self.env.reset(seed=seed)
    
    def step(self, action):
        return self.env.step(action)        

    def get_random_action(self):
        return self.env.action_space.sample()

    def __str__(self) -> str:
        return self.name
    
if __name__ == "__main__":
    env = InvertedPendulum()
    env.reset()
    for i in range(100):
        # get random action
        action = env.get_random_action()
        # perform action
        state, reward, done, _, _ = env.step(action)
        # reset if done
        if done:
            env.reset()
