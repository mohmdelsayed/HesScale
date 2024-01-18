from core.grid_search import GridSearch
from core.learner.sl.adam import AdamLearner
from core.network.fcn_relu import FCNReLUSmallWithNoBias
from core.network.fcn_tanh import FCNTanhSmallWithNoBias
from core.runner import Runner
from core.task.stationary_mnist import StationaryMNIST
from core.run.hessian_diagonality_hidden import RunHessianHiddenDiagonality
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "exp5"
task = StationaryMNIST()

adam_grid_relu = GridSearch(
               seed=[i for i in range(0, 100)],
               lr=[0.001],
               network=[FCNReLUSmallWithNoBias()],
               n_samples=[10000],
    )

adam_grid_tanh = GridSearch(
               seed=[i for i in range(0, 100)],
               lr=[0.001],
               network=[FCNTanhSmallWithNoBias()],
               n_samples=[10000],
    )

grids = [
        adam_grid_relu,
        adam_grid_tanh
]

learners = [
    AdamLearner(),
    AdamLearner(),
]

mode = 'w'
save_dir = "generated_cmds"
for learner, grid in zip(learners, grids):
    runner = Runner(RunHessianHiddenDiagonality(), learner, task, grid, exp_name)
    num_jobs = runner.write_cmd(save_dir)
    mode = 'a'
    create_script_generator(save_dir, exp_name, learner.name, num_jobs, time="1:00:00", memory="1G")
create_script_runner(save_dir, exp_name)