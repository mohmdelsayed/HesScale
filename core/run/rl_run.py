import torch, sys
import gym
from core.utils import networks, learners
from core.logger import Logger
from backpack import extend
import signal
import traceback
import time
from functools import partial

def signal_handler(msg, signal, frame):
    print('Exit signal: ', signal)
    cmd, learner = msg
    with open(f'timeout_{learner}.txt', 'a') as f:
        f.write(f"{cmd} \n")
    exit(0)

class Run:
    def __init__(self, name='rl_run', n_steps=10000, env='CartPole-v1', exp_name='exp1', learner='q_learning', save_path="logs", seed=0, network='mlp', **kwargs):
        self.name = name
        self.n_steps = int(n_steps)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = gym.make(env)

        self.exp_name = exp_name
        self.learner = learners[learner](networks[network](self.env.observation_space.shape[0], self.env.action_space.n), kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)

    def start(self):
        torch.manual_seed(self.seed)
        rewards_per_step = []

        state = self.env.reset()
        for step in range(self.n_steps):
            action = self.learner.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.learner.update(state, action, reward, next_state, done)
            state = next_state

            if done:
                state = self.env.reset()

            rewards_per_step.append(reward)

        logging_data = {
                'rewards': rewards_per_step,
                'exp_name': self.exp_name,
                'env': self.env.spec.id,
                'learner': self.learner.name,
                'network': self.learner.network.name,
                'optimizer_hps': self.learner.optim_kwargs,
                'n_steps': self.n_steps,
                'seed': self.seed,
        }

        self.logger.log(**logging_data)


    def __str__(self) -> str:
        return self.name

if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = Run(**args)
    cmd = f"python3 {' '.join(sys.argv)}"
    signal.signal(signal.SIGUSR1, partial(signal_handler, (cmd, args['learner'])))
    current_time = time.time()
    try:
        run.start()
        with open(f"finished_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} time_elapsed: {time.time()-current_time} \n")
    except Exception as e:
        with open(f"failed_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} \n")
        with open(f"failed_{args['learner']}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")