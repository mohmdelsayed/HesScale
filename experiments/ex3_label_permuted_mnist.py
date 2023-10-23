from core.grid_search import GridSearch
from core.learner.sgd import SGDLearner
from core.learner.adam import AdamLearner
from core.network.fcn_relu import FullyConnectedReLU
from core.runner import Runner
from core.run.run import Run
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex3_label_permuted_emnist"
task = tasks[exp_name]()

sgd_grid = GridSearch(
               seed=[i for i in range(0, 20)],
               lr=[10 ** -i for i in range(1, 5)],
               network=[FullyConnectedReLU()],
               n_samples=[1000000],
    )

adam_grid = GridSearch(
                seed=[i for i in range(0, 20)],
                lr=[10 ** -i for i in range(1, 5)],
                network=[FullyConnectedReLU()],
                n_samples=[1000000],
    )



grids = [sgd_grid] + [adam_grid]

learners = [
    SGDLearner(),
    AdamLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")