from core.task.stationary_mnist import StationaryMNIST
from core.task.label_permuted_emnist import LabelPermutedMNIST
from core.task.input_permuted_mnist import InputPermutedMNIST

from core.network.fcn_leakyrelu import FCNLeakyReLU
from core.network.fcn_relu import FCNReLU
from core.network.fcn_tanh import FCNTanh
from core.network.cnn_relu import CNNReLU

from core.learner.sl.sgd import SGDLearner
from core.learner.sl.adam import AdamLearner
from core.learner.sl.adahesscale import AdaHesScaleLearner
from core.learner.sl.adahesscalegn import AdaHesScaleGNLearner
from core.learner.sl.adahessian import AdaHessianLearner
from core.learner.sl.adaggnmc import AdaGGNMCLearner
import torch


tasks = {
    "stationary_mnist" : StationaryMNIST,
    "input_permuted_mnist": InputPermutedMNIST,
    "label_permuted_emnist" : LabelPermutedMNIST,

}

networks = {
    "fcn_relu": FCNReLU,
    "fcn_leakyrelu": FCNLeakyReLU,
    "fcn_tanh": FCNTanh,
    "cnn_relu": CNNReLU,
}

learners = {
    "sgd": SGDLearner,
    "adam": AdamLearner,
    "adahesscale": AdaHesScaleLearner,
    "adahesscalegn": AdaHesScaleGNLearner,
    "adahessian": AdaHessianLearner,
    "adaggnmc": AdaGGNMCLearner,
}

criterions = {
    "mse": torch.nn.MSELoss,
    "cross_entropy": torch.nn.CrossEntropyLoss,
}


def create_script_generator(path, exp_name):
    cmd=f'''#!/bin/bash
for f in *.txt
do
echo \"#!/bin/bash\" > ${{f%.*}}.sh
echo -e \"#SBATCH --signal=USR1@90\" >> ${{f%.*}}.sh
echo -e \"#SBATCH --job-name=\"${{f%.*}}\"\\t\\t\\t# single job name for the array\" >> ${{f%.*}}.sh
echo -e \"#SBATCH --mem=2G\\t\\t\\t# maximum memory 100M per job\" >> ${{f%.*}}.sh
echo -e \"#SBATCH --time=01:00:00\\t\\t\\t# maximum wall time per job in d-hh:mm or hh:mm:ss\" >> ${{f%.*}}.sh
echo \"#SBATCH --array=1-240\" >> ${{f%.*}}.sh
echo -e \"#SBATCH --account=def-ashique\" >> ${{f%.*}}.sh

echo "cd \"../../\"" >> ${{f%.*}}.sh
echo \"FILE=\\"\$SCRATCH/GT-learners/generated_cmds/{exp_name}/${{f%.*}}.txt\\"\"  >> ${{f%.*}}.sh
echo \"SCRIPT=\$(sed -n \\"\${{SLURM_ARRAY_TASK_ID}}p\\" \$FILE)\"  >> ${{f%.*}}.sh
echo \"module load python/3.7.9\" >> ${{f%.*}}.sh
echo \"source \$SCRATCH/GT-learners/.gt-learners/bin/activate\" >> ${{f%.*}}.sh
echo \"srun \$SCRIPT\" >> ${{f%.*}}.sh
done'''

    with open(f"{path}/create_scripts.bash", "w") as f:
        f.write(cmd)

    
def create_script_runner(path):
    cmd='''#!/bin/bash
for f in *.sh
do sbatch $f
done'''
    with open(f"{path}/run_all_scripts.bash", "w") as f:
        f.write(cmd)
