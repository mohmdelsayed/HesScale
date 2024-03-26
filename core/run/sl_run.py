import statistics
import torch, sys
from core.utils import tasks, networks, learners, criterions
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

class SLRun:
    def __init__(self, name='sl_run', torch_threads=0, n_epochs=200, batch_size=128, task='cifar-100', exp_name='exp1', learner='sgd', save_path="logs", seed=0, network='3c3d', **kwargs):
        self.name = name
        self.torch_threads = int(torch_threads)
        self.n_epochs = int(n_epochs)
        batch_size = int(batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = tasks[task](train=True, batch_size=batch_size)
        self.task_test = tasks[task](train=False, batch_size=batch_size)

        self.exp_name = exp_name
        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)

    def start(self):
        if self.torch_threads > 0:
            torch.set_num_threads(self.torch_threads)
        torch.manual_seed(self.seed)

        self.learner.setup_task(self.task)
        criterion = extend(criterions[self.task.criterion]()) if self.learner.extend else criterions[self.task.criterion]()
        logging_data = {
            'exp_name': self.exp_name,
            'task': self.task.name,
            'learner': self.learner.name,
            'network': self.learner.network.name,
            'optimizer': self.learner.name,
            'optimizer_hps': self.learner.optim_kwargs,
            'n_samples': self.task.n_samples,
            'n_epochs': self.n_epochs,
            'seed': self.seed,
        }

        l_train, a_train = [], []
        l_test, a_test = [], []
        times = []
        try:
            t0 = time.time()
            for e in range(self.n_epochs):
                self.task.reset()
                b = 0
                l_epoch, a_epoch = [], []
                while True:
                    input, target = next(self.task)
                    if input is None:
                        break
                    input, target = input.to(self.device), target.to(self.device)
                    def closure():
                        output = self.learner.predict(input)
                        loss = criterion(output, target)
                        return loss, output
                    loss, output = self.learner.update_params(closure=closure)
                    # check if loss is nan
                    if torch.isnan(loss):
                        raise ValueError("Loss is nan")
                    l_epoch.append(loss.item())
                    a_epoch.append((output.argmax(dim=1) == target).float().mean().item())

                    # print(b)
                    b += 1

                l_train.append(statistics.mean(l_epoch))
                a_train.append(statistics.mean(a_epoch))

                self.task_test.reset()
                bt = 0
                l_epoch, a_epoch = [], []
                while True:
                    input, target = next(self.task_test)
                    if input is None:
                        break
                    input, target = input.to(self.device), target.to(self.device)
                    output = self.learner.predict(input)
                    loss = criterion(output, target)
                    # check if loss is nan
                    if torch.isnan(loss):
                        raise ValueError("Loss is nan")
                    l_epoch.append(loss.item())
                    a_epoch.append((output.argmax(dim=1) == target).float().mean().item())

                    # print(bt)
                    bt += 1

                l_test.append(statistics.mean(l_epoch))
                a_test.append(statistics.mean(a_epoch))
                times.append(time.time() - t0)

                print(f'epoch {e} - train acc {a_train[-1]:.3f} - test acc {a_test[-1]:.3f}')

        except (ValueError) as err:
            print(err)
            logging_data.update({'diverged': True})

        logging_data.update({
            'loss_train': l_train,
            'loss_test': l_test,
            'accuracies_train': a_train,
            'accuracies_test': a_test,
            'times': times,
        })

        self.logger.log(**logging_data)


    def __str__(self) -> str:
        return self.name

if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = SLRun(**args)
    cmd = f"python3 {' '.join(sys.argv)}"
    signal.signal(signal.SIGUSR1, partial(signal_handler, (cmd, args['learner'])))
    current_time = time.time()
    # run.start()

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
