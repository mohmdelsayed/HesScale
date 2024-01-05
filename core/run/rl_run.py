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
    def __init__(self, name='rl_run', n_samples=10000, task='cartpole', exp_name='exp1', learner='vanilla_sgd', save_path="logs", seed=0, network='fcn_relu', optim='adam', **kwargs):
        self.name = name
        self.n_samples = int(n_samples)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = int(seed)
        if task == 'cartpole':
            self.env = environments[task](seed=self.seed)
        else:
            self.env = environments['mujoco_env'](name=task, seed=self.seed)

        self.exp_name = exp_name
        self.optim = optim
        if learner == 'a2c':
            self.learner = learners[learner](networks[network], optim=optim, optim_kwargs=kwargs)
        else:
            self.learner = learners[learner](networks[network], optim_kwargs=kwargs)
        self.logger = Logger(save_path)

    def start(self):
        torch.manual_seed(self.seed)
        ts = []
        return_per_episode = []
        self.learner.setup_env(self.env)
        self.learner.setup_losses(self.env)
        state = self.env.reset()
        episodic_return = 0.0
        epi_t0 = 0
        for t in range(self.n_samples):
            action = self.learner.act(state)
            next_state, reward, done, _ = self.env.step(action)
            terminated = done and (t - epi_t0 < self.env.get_max_episode_steps())
            self.learner.update(state, action, reward, next_state, terminated)
            state = next_state
            episodic_return += reward
            if done:
                state = self.env.reset()
                ts.append(t)
                return_per_episode.append(episodic_return)
                print(f'{t} ( {t - epi_t0} ) {episodic_return}')
                episodic_return = 0.0
                epi_t0 = t

        logging_data = {
                'ts': ts,
                'returns': return_per_episode,
                'exp_name': self.exp_name,
                'task': self.env.name,
                'learner': self.learner.name,
                'network': self.learner.actor.name,
                'optimizer': self.optim,
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
    # run.start()

    try:
        run.start()
        with open(f"finished_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} time_elapsed: {time.time()-current_time} \n")
    except Exception as e:
        print(e)
        with open(f"failed_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} \n")
        with open(f"failed_{args['learner']}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")
