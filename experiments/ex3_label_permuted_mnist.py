from core.grid_search import GridSearch
from core.learner.sl.sgd import SGDLearner
from core.learner.sl.adam import AdamLearner
from core.learner.sl.adahesscalegn import AdaHesScaleGNLearner, AdaHesScaleGNSqrtLearner, AdaHesScaleGNAdamStyleLearner
from core.network.fcn_relu import FCNReLU
from core.runner import Runner
from core.run.sl_run import Run
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "exp3"
task = tasks["label_permuted_emnist"]()

sgd_grid = GridSearch(
               seed=[i for i in range(0, 20)],
               lr=[10 ** -i for i in range(1, 5)],
               network=[FCNReLU()],
               n_samples=[1000000],
    )

adam_grid = GridSearch(
                seed=[i for i in range(0, 20)],
                lr=[10 ** -i for i in range(1, 5)],
                network=[FCNReLU()],
                n_samples=[1000000],
    )



adahesscale_grid = GridSearch(
                seed=[i for i in range(0, 20)],
                lr=[10 ** -i for i in range(1, 5)],
                network=[FCNReLU()],
                n_samples=[1000000],
    )

grids = [sgd_grid] + [adam_grid] + [adahesscale_grid] + [adahesscale_grid for _ in range(3)]

learners = [
    SGDLearner(),
    AdamLearner(),
    AdaHesScaleGNLearner(),
    AdaHesScaleGNSqrtLearner(),
    AdaHesScaleGNAdamStyleLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run(), learner, task, grid, exp_name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")