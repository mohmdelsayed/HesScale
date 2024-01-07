from backpack import extend
import torch

from hesscale.core.additional_activations import Exponential

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
        self.action_space_type = 'discrete' if env.action_space_type == 'discrete' else 'continuous'
        n_out = env.n_actions
        if self.extend:
            self.critic = extend(self.network_cls(n_obs=env.n_states, n_outputs=1).to(self.device))
            self.actor = extend(self.network_cls(n_obs=env.n_states, n_outputs=n_out).to(self.device))
            if self.action_space_type == 'continuous':
                self.var = extend(torch.nn.Sequential(torch.nn.Linear(env.n_states, 64), torch.nn.Tanh(), torch.nn.Linear(64, n_out), Exponential()).to(self.device))
        else:
            self.critic = self.network_cls(n_obs=env.n_states, n_outputs=1).to(self.device)
            self.actor = self.network_cls(n_obs=env.n_states, n_outputs=n_out).to(self.device)
            if self.action_space_type == 'continuous':
                self.var = torch.nn.Sequential(torch.nn.Linear(env.n_states, 64), torch.nn.Tanh(), torch.nn.Linear(64, n_out), Exponential()).to(self.device)
        self.setup_optimizer()

    def setup_optimizer(self):
        self.actor_optimizer = self.optimizer_cls(self.actor.parameters(), **self.optim_kwargs)
        if self.action_space_type == 'continuous':
            self.var_optimizer = self.optimizer_cls(self.var.parameters(), **self.optim_kwargs)
        self.critic_optimizer = self.optimizer_cls(self.critic.parameters(), **self.optim_kwargs)

    def update_params(self, actor_closure, critic_closure, var_closure=None):
        self.actor_optimizer.step(actor_closure)
        if self.action_space_type == 'continuous':
            self.var_optimizer.step(var_closure)
        self.critic_optimizer.step(critic_closure)
