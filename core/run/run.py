import torch, sys, os
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from backpack import backpack, extend
sys.path.insert(1, os.getcwd())
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
    name = 'run'
    def __init__(self, n_samples=10000, task=None, learner=None, save_path="logs", seed=0, network=None, **kwargs):
        self.n_samples = int(n_samples)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = tasks[task]()
        self.task_name = task
        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)

    def start(self):
        torch.manual_seed(self.seed)
        losses_per_step_size = []

        if self.task.criterion == 'cross_entropy':
            accuracy_per_step_size = []
        self.learner.set_task(self.task)
        criterion = extend(criterions[self.task.criterion]()) if self.learner.extend else criterions[self.task.criterion]()
        optimizer = self.learner.optimizer(
            self.learner.parameters, **self.learner.optim_kwargs
        )

        for _ in range(self.n_samples):
            input, target = next(self.task)
            input, target = input.to(self.device), target.to(self.device)
            def closure():
                output = self.learner.predict(input)
                loss = criterion(output, target)
                return loss, output
            loss, output = optimizer.step(closure=closure)
            losses_per_step_size.append(loss.item())
            if self.task.criterion == 'cross_entropy':
                accuracy_per_step_size.append((output.argmax(dim=1) == target).float().mean().item())

        if self.task.criterion == 'cross_entropy':
            self.logger.log(losses=losses_per_step_size,
                            accuracies=accuracy_per_step_size,
                            task=self.task_name, 
                            learner=self.learner.name,
                            network=self.learner.network.name,
                            optimizer_hps=self.learner.optim_kwargs,
                            n_samples=self.n_samples,
                            seed=self.seed,
            )
        else:
            self.logger.log(losses=losses_per_step_size,
                            task=self.task_name,
                            learner=self.learner.name,
                            network=self.learner.network.name,
                            optimizer_hps=self.learner.optim_kwargs,
                            n_samples=self.n_samples,
                            seed=self.seed,
            )


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