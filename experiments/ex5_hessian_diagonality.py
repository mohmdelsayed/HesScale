from core.grid_search import GridSearch
from core.learner.sl.adam import AdamLearner
from core.network.fcn_relu import FCNReLUSmallWithNoBias
from core.network.fcn_tanh import FCNTanhSmallWithNoBias
from core.runner import Runner
from core.run.hessian_diagonality import RunHessianDiagonality
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "exp5"
task = tasks["stationary_mnist"]()

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


grids = [adam_grid_relu] + [adam_grid_tanh]

learners = [
    AdamLearner(),
    AdamLearner(),
]

mode = 'w'
for learner, grid in zip(learners, grids):
    runner = Runner(RunHessianDiagonality(), learner, task, grid, exp_name)
    runner.write_cmd("generated_cmds", mode=mode)
    mode = 'a'
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")