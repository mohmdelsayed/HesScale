import numpy as np
from backpack import extend
import torch
from core.learner.rl.actor_critic_learner import ActorCriticLearner
from core.optim.sgd import SGD
from core.optim.adam import Adam
from core.optim.adahesscalegn import AdaHesScaleGN, AdaHesScaleGNSqrt, AdaHesScaleGNAdamStyle
from hesscale.core.additional_losses import SoftmaxNLLLoss, GaussianNLLLossMu, GaussianNLLLossVar
from hesscale.core.losses_gn import MSELossHesScale

class A2C(ActorCriticLearner):
    def __init__(self, network=None, gamma=0.99, optim='adam', optim_kwargs={}):
        optimizer = {
            'sgd': SGD,
            'adam': Adam,
            'adahesscalegn_sqrt': AdaHesScaleGNSqrt,
            'adahesscalegn': AdaHesScaleGN,
            'adahesscalegn_adamstyle': AdaHesScaleGNAdamStyle,
        }[optim]
        name='a2c'
        self.extend = True
        self.gamma = gamma
        # self.gamma_actor = gamma
        self.transitions = []
        super().__init__(name, network, optimizer, optim_kwargs, extend=self.extend)

    def setup_losses(self, env):
        if env.action_space_type == 'discrete':
            self.ac_lossf = SoftmaxNLLLoss(reduction='mean')
        else:
            full = True
            reduction = 'none'
            self.ac_lossf_mu = GaussianNLLLossMu(full=full, reduction=reduction)
            self.ac_lossf_var = GaussianNLLLossVar(full=full, reduction=reduction)
        self.cr_lossf = torch.nn.MSELoss()

        if self.extend:
            if env.action_space_type == 'discrete':
                self.ac_lossf = extend(self.ac_lossf)
            else:
                self.ac_lossf_mu = extend(self.ac_lossf_mu)
                self.ac_lossf_var =  extend(self.ac_lossf_var)
            self.cr_lossf = extend(self.cr_lossf)

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        dist = self.dist(state)
        action = dist.sample()
        return action.detach().numpy()

    def dist(self, state):
        action_prefs = self.actor(state)
        if self.action_space_type == 'discrete':
            dist = torch.distributions.Categorical(logits=action_prefs)
        elif self.action_space_type == 'continuous':
            var = self.var(torch.ones(1))
            dist = torch.distributions.Normal(loc=action_prefs, scale=torch.sqrt(var))
        else:
            raise NotImplementedError
        return dist

    def update(self, state, action, reward, next_state, terminated):
        self.transitions.append((state, action, reward, next_state, terminated))
        if terminated or len(self.transitions) == 5:
            actor_closure, critic_closure, var_closure = self.get_closures()
            self.update_params(actor_closure, critic_closure, var_closure)
            self.transitions = []
    
    def get_closures(self):
        ob_dim = self.transitions[0][0].shape[0] 
        ac_dim = self.transitions[0][1].shape[0] if self.action_space_type == 'continuous' else 1
        obs = torch.from_numpy(np.array([tr[0] for tr in self.transitions])).float().to(self.device).view(-1, ob_dim)
        acs = torch.from_numpy(np.array([tr[1] for tr in self.transitions])).float().to(self.device).view(-1, ac_dim)
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
        v_rets = v_rets.view(-1, 1).detach()

        action_prefs = self.actor(obs)
        if self.action_space_type == 'discrete':
            acs_onehot = torch.nn.functional.one_hot(acs[:, 0].type(torch.int64), num_classes=action_prefs.shape[1]).float()
            target = acs_onehot * (v_rets - vals)
            actor_loss = self.ac_lossf(action_prefs, target)
        else:
            var = self.var(torch.ones(acs.shape[0], 1))
            actor_loss = ((v_rets - vals) * self.ac_lossf_mu(action_prefs, var, acs)).mean()
            var_loss = ((v_rets - vals) * self.ac_lossf_var(var, action_prefs, acs)).mean()

        critic_loss = self.cr_lossf(self.predict(obs), v_rets)

        def actor_closure():            
            return actor_loss, acs
        def critic_closure():
            return critic_loss, vals
        var_closure = None
        if self.action_space_type == 'continuous':
            def v_clsr():
                return var_loss, var
            var_closure = v_clsr
        return actor_closure, critic_closure, var_closure
