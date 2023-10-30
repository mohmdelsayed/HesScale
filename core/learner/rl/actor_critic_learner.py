from backpack import extend
import torch

class ActorCriticLearner:
    def __init__(self, name, network, optimizer, optim_kwargs, extend=False):
        self.network_cls = network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optim_kwargs = optim_kwargs
        for k, v in optim_kwargs.items():
            if isinstance(v, str):
                optim_kwargs[k] = float(v)
        self.optimizer_cls = optimizer
        self.name = name
        self.extend = extend

    def __str__(self):
        return self.name
    
    def predict(self, state):
        value = self.critic(state)
        return value

    def act(self, state):
        raise NotImplementedError
    
    def sample_action(self, dist):
        raise NotImplementedError

    def setup_env(self, env):
        if self.extend:
            self.critic = extend(self.network_cls(n_obs=env.n_states, n_outputs=1).to(self.device))
            self.actor = extend(self.network_cls(n_obs=env.n_states, n_outputs=env.n_actions).to(self.device))
        else:
            self.critic = self.network_cls(n_obs=env.n_states, n_outputs=1).to(self.device)
            self.actor = self.network_cls(n_obs=env.n_states, n_outputs=env.n_actions).to(self.device)
        self.action_space_type = 'discrete' if env.action_space_type == 'discrete' else 'continous'
        self.setup_optimizer()

    def setup_optimizer(self):
        self.actor_optimizer = self.optimizer_cls(self.actor.parameters(), **self.optim_kwargs)
        self.critic_optimizer = self.optimizer_cls(self.critic.parameters(), **self.optim_kwargs)

    def update_params(self, actor_closure, critic_closure):
        self.actor_optimizer.step(actor_closure)
        self.critic_optimizer.step(critic_closure)
