import gymnasium as gym

class CartPole:
    """
    Iteratable CartPole task.
    Each sample is a 4-dimensional state and the action is a number between 0 and 1.
    """

    def __init__(self, name="cartpole"):
        self.env = gym.make('CartPole-v1')
        self.name = name
        self.n_states = 4
        self.n_actions = 2
        self.action_space_type = 'discrete'
        super().__init__()

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)        

    def get_random_action(self):
        return self.env.action_space.sample()

    def __str__(self) -> str:
        return self.name
    
if __name__ == "__main__":
    env = CartPole()
    env.reset()
    for i in range(100):
        # get random action
        action = env.get_random_action()
        # perform action
        state, reward, done, _, _ = env.step(action)
        # reset if done
        if done:
            env.reset()
