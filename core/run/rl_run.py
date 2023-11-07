import torch, sys
from core.utils import environments
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

class RLRun:
    def __init__(self, name='rl_run', n_samples=10000, task='cartpole', exp_name='exp1', learner='vanilla_sgd', save_path="logs", seed=0, network='fcn_relu', **kwargs):
        self.name = name
        self.n_samples = int(n_samples)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = environments[task]()

        self.exp_name = exp_name
        self.learner = learners[learner](networks[network], optim_kwargs=kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)

    def start(self):
        torch.manual_seed(self.seed)
        return_per_episode = []
        self.learner.setup_env(self.env)
        state, _ = self.env.reset()
        episodic_return = 0.0
        for _ in range(self.n_samples):
            action = self.learner.act(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.learner.update(state, action, reward, next_state, done)
            state = next_state
            episodic_return += reward
            if done:
                state, _ = self.env.reset()
                return_per_episode.append(episodic_return)
                print(episodic_return)
                episodic_return = 0.0

        logging_data = {
                'returns': return_per_episode,
                'exp_name': self.exp_name,
                'task': self.env.name,
                'learner': self.learner.name,
                'network': self.learner.actor.name,
                'optimizer_hps': self.learner.optim_kwargs,
                'n_samples': self.n_samples,
                'seed': self.seed,
        }

        self.logger.log(**logging_data)


    def __str__(self) -> str:
        return self.name

if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = RLRun(**args)
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