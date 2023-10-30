from backpack import extend
import torch
from core.learner.rl.actor_critic_learner import ActorCriticLearner
from core.optim.sgd import SGD

class VanillaSGD(ActorCriticLearner):
    def __init__(self, network=None, gamma=0.9, optim_kwargs={}):
        optimizer = SGD
        name='vanilla_sgd'
        self.gamma = gamma
        self.gamma_actor = gamma
        super().__init__(name, network, optimizer, optim_kwargs)

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action_logits = self.actor(state)
        self.action = self.sample_from_dist(action_logits, self.action_space_type)
        return self.action.item()

    def sample_from_dist(self, action_outputs, action_space_type):
        if action_space_type == 'discrete':
            action_outputs = torch.softmax(action_outputs, dim=-1)
            self.dist = torch.distributions.Categorical(action_outputs)
            return self.dist.sample()
        elif action_space_type == 'continous':
            self.dist = torch.distributions.Normal(loc=action_outputs[0], scale=action_outputs[1])
        else:
            raise NotImplementedError
        return self.dist.sample()

    def update(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        actor_closure, critic_closure = self.get_closures(state, action, next_state, reward, done)
        self.update_params(actor_closure, critic_closure)
    
    def get_closures(self, state, action, next_state, reward, done):                
        value = self.predict(state)
        next_value = self.predict(next_state)
        td_target = reward + self.gamma * next_value * (1 - done)
        td_error = td_target.item() - value
        critic_loss = td_error ** 2
        actor_loss = -self.gamma_actor * self.dist.log_prob(self.action) * td_error.detach()
        self.gamma_actor = self.gamma_actor * self.gamma
        def critic_closure():
            return critic_loss, value
        def actor_closure():            
            return actor_loss, action
        return actor_closure, critic_closure