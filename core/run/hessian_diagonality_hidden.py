import torch, sys
from curvlinops.experimental import ActivationHessianLinearOperator
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from backpack import extend
import signal
import traceback
import time
import numpy as np
from functools import partial

def signal_handler(msg, signal, frame):
    print('Exit signal: ', signal)
    cmd, learner = msg
    with open(f'timeout_{learner}.txt', 'a') as f:
        f.write(f"{cmd} \n")
    exit(0)

def hidden_hessian(loss_func, data, model, batch_size, feature_dim):
    hessian = ActivationHessianLinearOperator(model, loss_func, ("1", "input", 0), data)
    H_mat = (hessian @ np.eye(hessian.shape[1])).reshape(batch_size, feature_dim, batch_size, feature_dim)
    idx = np.arange(batch_size)
    hessian_matrix = np.abs(H_mat[idx, :, idx, :])
    return hessian_matrix

class RunHessianHiddenDiagonality:
    def __init__(self, name='hessian_diagonality_hidden', n_samples=10000, task='stationary_mnist', exp_name='exp1', learner='sgd', save_path="logs", seed=0, network='fcn_relu', **kwargs):
        self.name = name
        self.n_samples = int(n_samples)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_batch_size_32 = tasks[task](batch_size=32)
        self.task_batch_size_1000 = tasks[task](batch_size=1000)
        self.exp_name = exp_name
        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)

    def start(self):
        torch.manual_seed(self.seed)
        losses_per_step_size = []
        if self.task_batch_size_32.criterion == 'cross_entropy':
            accuracy_per_step_size = []
        self.learner.setup_task(self.task_batch_size_32)
        criterion = extend(criterions[self.task_batch_size_32.criterion]()) if self.learner.extend else criterions[self.task_batch_size_32.criterion]()

        def closure():
            output = self.learner.predict(input)
            loss = criterion(output, target)
            return loss, output

        # Hessian before training
        input, target = next(self.task_batch_size_1000)
        input, target = input.to(self.device), target.to(self.device)
        H_before = hidden_hessian(criterion, [(input, target)], self.learner.network, self.task_batch_size_1000.batch_size, self.learner.network.n_hidden_units).mean(axis=0)
        print(H_before.shape)
        # training
        for _ in range(self.n_samples):
            input, target = next(self.task_batch_size_32)
            input, target = input.to(self.device), target.to(self.device)
            loss, output = self.learner.update_params(closure=closure)
            losses_per_step_size.append(loss.item())
            if self.task_batch_size_32.criterion == 'cross_entropy':
                accuracy_per_step_size.append((output.argmax(dim=1) == target).float().mean().item())

        # Hessian after training
        input, target = next(self.task_batch_size_1000)
        input, target = input.to(self.device), target.to(self.device)
        H_after = hidden_hessian(criterion, [(input, target)], self.learner.network, self.task_batch_size_1000.batch_size, self.learner.network.n_hidden_units).mean(axis=0)
        print(H_after.shape)

        logging_data = {
                'losses': losses_per_step_size,
                'exp_name': self.exp_name,
                'task': self.task_batch_size_32.name,
                'learner': self.learner.name,
                'network': self.learner.network.name,
                'optimizer_hps': self.learner.optim_kwargs,
                'n_samples': self.n_samples,
                'seed': self.seed,
                'H_before': H_before.tolist(),
                'H_after': H_after.tolist(),
        }

        if self.task_batch_size_32.criterion == 'cross_entropy':
            logging_data['accuracies'] = accuracy_per_step_size

        self.logger.log(**logging_data)

    def __str__(self) -> str:
        return self.name

if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = RunHessianHiddenDiagonality(**args)
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