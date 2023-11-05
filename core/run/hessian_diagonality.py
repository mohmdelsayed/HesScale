import torch, sys
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from torch.nn.utils import parameters_to_vector, vector_to_parameters
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

def hessian(loss, params):
    # Calculate Hessian using Hessian-vector product with one-hot vectors
    grads = torch.autograd.grad(loss, params, create_graph=True)

    vs = []
    for p in params:
        v = torch.zeros_like(p)
        vs.append(v)
    vectorized_v = parameters_to_vector(vs)
    hvps = []
    for i in range(vectorized_v.shape[0]):
        vectorized_v[i] = 1
        vector_to_parameters(vectorized_v, vs)
        hvp = torch.autograd.grad(grads, params, vs, retain_graph=True)
        vectorized_v = parameters_to_vector(vs)
        vectorized_v[i] = 0
        hvp = torch.cat([e.flatten() for e in hvp])
        hvps.append(hvp)
    hessian_matrix = torch.stack(hvps)
    return hessian_matrix

class RunHessianDiagonality:
    def __init__(self, name='hessian_diagonality', n_samples=10000, task='stationary_mnist', exp_name='exp1', learner='sgd', save_path="logs", seed=0, network='fcn_relu', **kwargs):
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
        loss_before, output = closure()
        H_before = hessian(loss_before, self.learner.parameters[1:])

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
        loss_after, output = closure()
        H_after = hessian(loss_after, self.learner.parameters[1:])

        logging_data = {
                'losses': losses_per_step_size,
                'exp_name': self.exp_name,
                'task': self.task_batch_size_32.name,
                'learner': self.learner.name,
                'network': self.learner.network.name,
                'optimizer_hps': self.learner.optim_kwargs,
                'n_samples': self.n_samples,
                'seed': self.seed,
                'H_before': H_before.cpu().numpy().tolist(),
                'H_after': H_after.cpu().numpy().tolist(),
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
    run = RunHessianDiagonality(**args)
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