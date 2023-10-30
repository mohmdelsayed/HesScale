from core.grid_search import GridSearch
from core.learner.rl.vanilla_sgd import VanillaSGD
from core.network.fcn_tanh import FCNTanh
from core.runner import Runner
from core.run.rl_run import Run
from core.utils import create_script_generator, create_script_runner, environments

exp_name = "exp4"
env = environments["cartpole"]()

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
for learner, grid in zip(learners, grids):
    runner = Runner(Run(), learner, env, grid, exp_name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")