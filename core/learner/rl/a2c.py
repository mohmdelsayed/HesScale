import numpy as np
from backpack import extend
import torch
from core.learner.rl.actor_critic_learner import ActorCriticLearner
from core.optim.adam import Adam

class A2C(ActorCriticLearner):
    def __init__(self, network=None, gamma=0.99, optim_kwargs={}):
        optimizer = Adam
        name='a2c'
        self.gamma = gamma
        # self.gamma_actor = gamma
        self.transitions = []
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

    def update(self, state, action, reward, next_state, terminated):
        self.transitions.append((state, action, reward, next_state, terminated))
        if terminated or len(self.transitions) == 5:
            actor_closure, critic_closure = self.get_closures()
            self.update_params(actor_closure, critic_closure)
            self.transitions = []
    
    def get_closures(self):
        ob_dim = self.transitions[0][0].shape[0] 
        obs = torch.from_numpy(np.array([tr[0] for tr in self.transitions])).float().to(self.device).view(-1, ob_dim)
        acs = torch.from_numpy(np.array([tr[1] for tr in self.transitions])).float().to(self.device).view(-1, 1)
        rs = torch.from_numpy(np.array([tr[2] for tr in self.transitions])).float().to(self.device).view(-1, 1)
        next_ob = torch.from_numpy(self.transitions[-1][3]).float().to(self.device).view(-1, ob_dim)
        termin = torch.from_numpy(np.array(self.transitions[-1][4])).float().to(self.device).view(-1, 1)

        vals = self.predict(obs).detach()
        next_val = self.predict(next_ob).detach()
        steps = len(self.transitions)
        v_rets = torch.from_numpy(np.zeros(steps)).float().to(self.device)
        v_rets[-1] = rs[-1] + (1 - termin) * self.gamma * next_val
        for t in reversed(range(steps - 1)):
            v_rets[t] = rs[t] + self.gamma * v_rets[t + 1]
        v_rets = v_rets.view(-1, 1)

        logps = self.dist.log_prob(acs)
        actor_loss = ((v_rets - vals) * -logps).mean()
        critic_loss = ((v_rets - self.predict(obs)).pow(2)).mean()

        def actor_closure():            
            return actor_loss, acs
        def critic_closure():
            return critic_loss, vals
        return actor_closure, critic_closure
