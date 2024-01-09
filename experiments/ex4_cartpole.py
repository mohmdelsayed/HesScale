from pathlib import Path

from core.grid_search import GridSearch
import core.learner.rl as rl
from core.network.fcn_tanh import FCNTanh, FCNTanhSmall
from core.runner import Runner
from core.run.rl_run import RLRun
from core.task.cartpole import CartPole
from core.task.mujoco_env import MujocoEnv
from core.utils import create_script_generator, create_script_runner

# env = CartPole()

# n_samples = 200000
# task = 'InvertedPendulum'
# time = '0:20:00'
# memory = '1G'
n_samples = 1000000
task = 'Ant'
time = '1:58:00'
memory = '1G'
env = MujocoEnv(name=task)

# exp_name = f'exp4_{task}3'
# account = 'rrg-ashique'
# optim = [
#     'adahesscale_scaled', 'adahesscale_sqrt_scaled', 'adahesscale_adamstyle_scaled',
#     'adahesscalegn_scaled', 'adahesscalegn_sqrt_scaled', 'adahesscalegn_adamstyle_scaled',
#     'adam_scaled', 'adam_scaled_sqrt',
# ]

exp_name = f'exp4_{task}4'
account = 'def-ashique'
optim = [
    'adahesscale', 'adahesscale_sqrt', 'adahesscale_adamstyle',
    'adahesscalegn', 'adahesscalegn_sqrt', 'adahesscalegn_adamstyle',
    'sgd', 'adam',
]

a2c_grid = GridSearch(
                seed=[i for i in range(0, 10)],
                optim=optim,
                # lr=[.000003, .00001, .00003, .0001, .0003, .001, .003, .01, .03, .1, .3, 1.],
                lr=[.000003, .00001, .00003, .0001, .0003, .001, .003, .01],
                network=[FCNTanhSmall()],
                n_samples=[n_samples],
    )

grids = [a2c_grid]

learners = [
    rl.A2C(),
]

hes_dir = Path(__file__).parent / '..'
save_dir = Path(__file__).parent / '../generated_cmds'
env_dir = Path(__file__).parent / '../envhes'
for learner, grid in zip(learners, grids):
    runner = Runner(RLRun(), learner, env, grid, exp_name)
    num_jobs = runner.write_cmd(save_dir)
    create_script_generator(hes_dir, save_dir, exp_name, env_dir, learner.name, num_jobs, time=time, memory=memory, account=account)
create_script_runner(save_dir, exp_name)
