from core.grid_search import GridSearch
from core.learner.rl.vanilla_sgd import VanillaSGD
from core.network.fcn_tanh import FCNTanh
from core.runner import Runner
from core.run.rl_run import RLRun
from core.task.cartpole import CartPole
from core.utils import create_script_generator, create_script_runner

exp_name = "exp4"
env = CartPole()

adam_grid = GridSearch(
                seed=[i for i in range(0, 20)],
                lr=[0.0003],
                network=[FCNTanh()],
                n_samples=[1000000],
    )

grids = [adam_grid]

learners = [
    VanillaSGD(),
]

save_dir = "generated_cmds"
for learner, grid in zip(learners, grids):
    runner = Runner(RLRun(), learner, env, grid, exp_name)
    num_jobs = runner.write_cmd(save_dir)
    create_script_generator(save_dir, exp_name, learner.name, num_jobs, time="1:00:00", memory="1G")
create_script_runner(save_dir, exp_name)