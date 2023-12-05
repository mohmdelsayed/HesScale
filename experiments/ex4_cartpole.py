from pathlib import Path

from core.grid_search import GridSearch
from core.learner.rl.vanilla_sgd import VanillaSGD
from core.learner.rl.a2c import A2C
from core.network.fcn_tanh import FCNTanh, FCNTanhSmall
from core.runner import Runner
from core.run.rl_run import RLRun
from core.task.cartpole import CartPole
from core.utils import create_script_generator, create_script_runner

exp_name = "exp4"
env = CartPole()

a2c_grid = GridSearch(
                seed=[i for i in range(0, 10)],
                optim=['adam', 'adahesscalegn_sqrt'],
                lr=[.000003, .00001, .00003, .0001, .0003, .001, .003, .01],
                network=[FCNTanhSmall()],
                n_samples=[200000],
    )

grids = [a2c_grid]

learners = [
    A2C(),
]

hes_dir = Path(__file__).parent / '..'
save_dir = Path(__file__).parent / '../generated_cmds'
env_dir = Path(__file__).parent / '../envhes'
for learner, grid in zip(learners, grids):
    runner = Runner(RLRun(), learner, env, grid, exp_name)
    num_jobs = runner.write_cmd(save_dir)
    create_script_generator(hes_dir, save_dir, exp_name, env_dir, learner.name, num_jobs, time="0:58:00", memory="1G")
create_script_runner(save_dir, exp_name)
