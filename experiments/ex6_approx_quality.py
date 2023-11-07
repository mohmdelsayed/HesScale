from core.grid_search import GridSearch
from core.learner.sl.adam import AdamLearner
from core.network.fcn_relu import FCNReLUSmall
from core.network.fcn_tanh import FCNTanhSmall
from core.runner import Runner
from core.task.stationary_mnist import StationaryMNIST
from core.run.approx_quality import RunApproxQuality
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "exp6"
task = StationaryMNIST()

adam_grid_relu = GridSearch(
               seed=[i for i in range(0, 40)],
               lr=[0.001],
               network=[FCNReLUSmall()],
               n_samples=[10000],
               n_eval_samples=[100],
    )

adam_grid_tanh = GridSearch(
               seed=[i for i in range(0, 40)],
               lr=[0.001],
               network=[FCNTanhSmall()],
               n_samples=[10000],
               n_eval_samples=[100],
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
    runner = Runner(RunApproxQuality(), learner, task, grid, exp_name)
    num_jobs = runner.write_cmd(save_dir)
    mode = 'a'
    create_script_generator(save_dir, exp_name, learner.name, num_jobs, time="1:00:00", memory="1G")
create_script_runner(save_dir, exp_name)